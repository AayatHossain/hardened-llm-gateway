"""Tests for confidence intervals and paired comparisons of eval runs."""

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.evals.stats import (
    _resampled_sum,
    compare,
    load_run,
    main,
    mcnemar,
    mean_interval,
    sign_flip,
    wilson,
)

# --- 1 run ---


@pytest.mark.parametrize(
    ("passed", "n", "low", "high"),
    [
        (8, 10, 0.4902, 0.9433),  # textbook values
        (5, 5, 0.5655, 1.0),  # all passed: still not certain
        (0, 10, 0.0, 0.2775),
        (500, 1000, 0.4691, 0.5309),
    ],
)
def test_wilson(passed: int, n: int, low: float, high: float) -> None:
    got_low, got_high = wilson(passed, n)
    assert got_low == pytest.approx(low, abs=1e-4)
    assert got_high == pytest.approx(high, abs=1e-4)


def test_pass_fail_scores_use_wilson() -> None:
    estimate = mean_interval([1.0] * 8 + [0.0] * 2)
    assert estimate is not None
    assert (estimate.mean, estimate.n, estimate.method) == (0.8, 10, "wilson")
    assert (estimate.low, estimate.high) == wilson(8, 10)
    assert mean_interval([]) is None


def test_ratings_use_a_bootstrap_matching_the_normal_approximation() -> None:
    rng = random.Random(1)
    ratings = [rng.choice([0.2, 0.5, 0.7, 0.8, 0.9, 1.0]) for _ in range(400)]
    estimate = mean_interval(ratings)
    assert estimate is not None and estimate.method == "bootstrap"
    half = 1.96 * statistics.pstdev(ratings) / math.sqrt(len(ratings))
    assert estimate.low == pytest.approx(estimate.mean - half, abs=0.005)
    assert estimate.high == pytest.approx(estimate.mean + half, abs=0.005)
    assert mean_interval(ratings) == estimate  # fixed seed: the same every time
    narrower = mean_interval(ratings, confidence=0.5)
    assert narrower is not None and narrower.high < estimate.high


def test_a_resample_has_the_right_mean_and_spread() -> None:
    """1 multinomial draw over distinct values behaves like n draws with replacement."""
    values = [0.0] * 50 + [0.5] * 30 + [1.0] * 20
    counts = [(0.0, 50), (0.5, 30), (1.0, 20)]
    rng = random.Random(7)
    sums = [_resampled_sum(counts, 100, rng) for _ in range(20_000)]
    assert statistics.fmean(sums) == pytest.approx(sum(values), rel=0.01)
    expected_sd = statistics.pstdev(values) * math.sqrt(100)
    assert statistics.pstdev(sums) == pytest.approx(expected_sd, rel=0.03)
    assert _resampled_sum([(0.7, 10)], 10, rng) == pytest.approx(7.0)


# --- 2 runs ---


def test_mcnemar() -> None:
    differences = [1.0] * 10 + [-1.0] * 2 + [0.0] * 50
    assert mcnemar(differences) == pytest.approx(158 / 4096)  # 2 * P(X <= 2), X ~ Bin(12, 0.5)
    assert mcnemar([0.0] * 5) == 1.0
    assert mcnemar([1.0, -1.0]) == 1.0


def test_sign_flip() -> None:
    assert sign_flip([0.0, 0.0]) == 1.0
    assert sign_flip([0.3] * 30) < 0.001  # every item better: very unlikely by chance
    assert sign_flip([0.2, -0.2] * 15) > 0.9  # balanced: no sign of a difference


def test_pairing_finds_what_separate_intervals_miss() -> None:
    """B passes 12 more items and never fails one that A passes."""
    a = [1.0] * 140 + [0.0] * 60
    b = [1.0] * 152 + [0.0] * 48
    a_ci, b_ci = mean_interval(a), mean_interval(b)
    assert a_ci is not None and b_ci is not None
    assert b_ci.low < a_ci.high  # the 2 intervals overlap...
    result = compare(a, b)
    assert result is not None
    assert result.test == "mcnemar"
    assert result.p_value < 0.001  # ...but the paired test is clear
    assert (result.a, result.b) == (0.7, 0.76)
    assert result.difference.mean == pytest.approx(0.06)
    assert result.difference.low > 0
    assert result.p_value == 2 / 2**12  # 12 items differ, all in B's favour


def test_compare_ratings_and_errors() -> None:
    result = compare([0.5, 0.6, 0.7] * 20, [0.6, 0.7, 0.8] * 20)
    assert result is not None
    assert result.test == "sign-flip"
    assert result.difference.mean == pytest.approx(0.1)
    assert result.p_value < 0.001
    assert compare([], []) is None
    with pytest.raises(ValueError, match="differ in length"):
        compare([1.0], [1.0, 0.0])


# --- Runs on disk ---


def write_run(folder: Path, scores: dict[str, tuple[str, float | None]], **info: Any) -> Path:
    folder.mkdir(parents=True)
    run_info = {"model": "groq/m", "judge": "groq/j", "judge_prompt_version": 1} | info
    (folder / "run.json").write_text(json.dumps(run_info), encoding="utf-8")
    lines = [
        json.dumps({"id": item_id, "task": task, "score": score})
        for item_id, (task, score) in scores.items()
    ]
    (folder / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def test_load_run_keeps_graded_items(tmp_path: Path) -> None:
    folder = write_run(tmp_path / "r", {"a": ("maths", 1.0), "b": ("chat", None)})
    run = load_run(folder)
    assert run.scores == {"a": ("maths", 1.0)}
    assert run.label == "groq/m (r)"


def test_command_describes_1_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scores: dict[str, tuple[str, float | None]] = {
        f"m{i}": ("maths", float(i < 8)) for i in range(10)
    }
    scores |= {"c1": ("chat", 0.8), "c2": ("chat", 0.6)}
    assert main([str(write_run(tmp_path / "r", scores))]) == 0
    out = capsys.readouterr().out
    assert "maths          10   0.800  [0.490, 0.943]  wilson" in out
    assert "chat            2   0.700  [0.600, 0.800]  bootstrap" in out
    assert "total          12" in out


def test_command_compares_2_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a_scores: dict[str, tuple[str, float | None]] = {
        f"m{i}": ("maths", float(i < 140)) for i in range(200)
    }
    b_scores: dict[str, tuple[str, float | None]] = {
        f"m{i}": ("maths", float(i < 152)) for i in range(200)
    }
    b_scores["extra"] = ("maths", 1.0)  # only in B: left out
    a = write_run(tmp_path / "a", a_scores)
    b = write_run(tmp_path / "b", b_scores, model="gemini/m", judge="gemini/j")
    assert main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "A: groq/m (a)\nB: gemini/m (b)" in out
    assert "note: 200 items graded in both runs (A has 200, B has 201)" in out
    assert "warning: different judge (groq/j vs gemini/j)" in out
    assert "maths         200   0.700   0.760   +0.060  [+0.030, +0.095]    0.0005 * mcnemar" in out


def test_incomplete_runs_are_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scores: dict[str, tuple[str, float | None]] = {"m1": ("maths", 1.0)}
    stopped = write_run(tmp_path / "stopped", scores, complete=False)
    whole = write_run(tmp_path / "whole", scores, complete=True)
    warning = "warning: stopped is incomplete (stopped early)"
    assert main([str(stopped)]) == 0
    assert warning in capsys.readouterr().out
    assert main([str(whole)]) == 0
    assert "incomplete" not in capsys.readouterr().out
    assert main([str(whole), str(stopped)]) == 0
    assert warning in capsys.readouterr().out


def test_command_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["a", "b", "c"])
    with pytest.raises(SystemExit):
        main(["a", "--confidence", "95"])
    assert main([str(tmp_path / "missing")]) == 1
    assert "cannot read run" in capsys.readouterr().err
