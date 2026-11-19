"""Tests for the baseline report."""

import json
from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.evals.baseline import (
    EvalSetInfo,
    latest_runs,
    main,
    problems,
    read_eval_set,
    read_run,
    render,
)

FINGERPRINT = "047f881e4b3b"
EVAL_SET = EvalSetInfo(FINGERPRINT, 10, {"maths": 6, "chat": 4})


def task(mean: float, ci: tuple[float, float], items: int, **fields: Any) -> dict[str, Any]:
    return {
        "items": items,
        "mean_score": mean,
        "score_ci": list(ci),
        "errors": 0,
        "ungraded": 0,
        "prompt_tokens": 100 * items,
        "completion_tokens": 50 * items,
        "cost_usd": 0.0002 * items,
        "unpriced": 0,
        "cut_off": 0,
        "p50_ms": 1234.0,
    } | fields


def run_info(model: str, total: float = 0.7, **changes: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "model": model,
        "started_at": "2026-11-20T10:00:00+00:00",
        "finished_at": "2026-11-20T11:00:00+00:00",
        "git_commit": "7e4df5d",
        "eval_set": {
            "items": 10,
            "set_fingerprint": FINGERPRINT,
            "split": "test",
            "tasks": None,
            "limit": None,
        },
        "options": {"temperature": 0.0, "max_tokens": 4096},
        "judge": "groq/openai/gpt-oss-120b",
        "judge_prompt_version": 1,
        "complete": True,
        "summary": {
            "maths": task(0.5, (0.19, 0.81), 6),
            "chat": task(0.8, (0.7, 0.9), 4),
            "total": task(total, (0.4, 0.89), 10),
        },
    }
    for key, value in changes.items():
        if key in info["eval_set"]:
            info["eval_set"][key] = value
        else:
            info[key] = value
    return info


def write_run(root: Path, name: str, info: dict[str, Any]) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    return folder


def write_manifest(data: Path) -> Path:
    data.mkdir()
    manifest = {
        "fingerprint": FINGERPRINT,
        "files": {
            "maths.jsonl": {"task": "maths", "dev": 3, "test": 6},
            "chat.jsonl": {"task": "chat", "dev": 2, "test": 4},
        },
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data


# --- Which runs count ---


def test_a_full_test_split_run_counts(tmp_path: Path) -> None:
    run = read_run(write_run(tmp_path, "r", run_info("groq/a")))
    assert problems(run, EVAL_SET) == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"complete": False}, "it did not finish (stopped early, or still running)"),
        ({"summary": None}, "it did not finish (stopped early, or still running)"),
        ({"items": 11}, "not every item finished"),
        ({"split": "dev"}, "split is 'dev', not 'test'"),
        ({"tasks": ["maths"]}, "only some tasks (maths)"),
        ({"limit": 20}, "a sample of 20 items"),
        ({"set_fingerprint": "old"}, f"eval set old, not today's {FINGERPRINT}"),
    ],
)
def test_other_runs_do_not(tmp_path: Path, changes: dict[str, Any], reason: str) -> None:
    run = read_run(write_run(tmp_path, "r", run_info("groq/a", **changes)))
    assert problems(run, EVAL_SET) == [reason]


def test_latest_runs_takes_each_models_newest_baseline_run(tmp_path: Path) -> None:
    write_run(tmp_path, "1", run_info("groq/a", total=0.1, started_at="2026-11-18"))
    write_run(tmp_path, "2", run_info("groq/a", total=0.2, started_at="2026-11-19"))
    write_run(tmp_path, "3", run_info("groq/a", started_at="2026-11-21", complete=False))
    write_run(tmp_path, "4", run_info("gemini/b"))
    (tmp_path / "cache").mkdir()  # not a run
    (write_run(tmp_path, "5", {}) / "run.json").write_text("{broken", encoding="utf-8")

    runs = {r.model: r.folder.name for r in latest_runs(tmp_path, EVAL_SET)}
    assert runs == {"groq/a": "2", "gemini/b": "4"}  # "3" is newer but incomplete
    assert latest_runs(tmp_path / "missing", EVAL_SET) == []


def test_read_eval_set(tmp_path: Path) -> None:
    eval_set = read_eval_set(write_manifest(tmp_path / "data"))
    assert eval_set == EVAL_SET


# --- The report ---


def test_report(tmp_path: Path) -> None:
    weaker = run_info("groq/a", total=0.6)
    weaker["summary"]["total"] |= {"unpriced": 2, "cut_off": 3, "errors": 1, "ungraded": 1}
    stronger = run_info("gemini/b", total=0.75)
    del stronger["summary"]["chat"]
    stronger["summary"]["total"]["cost_usd"] = None
    runs = [read_run(write_run(tmp_path, n, i)) for n, i in [("ra", weaker), ("rb", stronger)]]

    report = render(runs, EVAL_SET)
    assert report.startswith("# Baseline\n")
    assert "the 10 test items of the eval set (fingerprint `047f881e4b3b`: 6 maths, 4 chat)" in (
        report
    )
    assert "graded by groq/openai/gpt-oss-120b (prompt version 1)." in report
    assert "not comparable" not in report

    rows = [line for line in report.splitlines() if line.startswith("| ")]
    assert rows[0] == "| Model | Total | Maths | Chat |"
    assert rows[1] == "| gemini/b | 0.75 [0.40, 0.89] | 0.50 [0.19, 0.81] | - |"  # best first
    assert rows[2] == "| groq/a | 0.60 [0.40, 0.89] | 0.50 [0.19, 0.81] | 0.80 [0.70, 0.90] |"
    assert "| groq/a | $0.20 (2 replies unpriced) | 100 / 50 | 1.2 s | 3 | 1 | 1 |" in rows
    assert "| gemini/b | unknown | 100 / 50 | 1.2 s | 0 | 0 | 0 |" in rows
    assert "| groq/a | `ra` | 2026-11-20 | 0.0 | 4096 | `7e4df5d` |" in rows


def test_report_warns_about_different_judges(tmp_path: Path) -> None:
    infos = [run_info("groq/a"), run_info("gemini/b", judge="gemini/j")]
    runs = [read_run(write_run(tmp_path, str(n), info)) for n, info in enumerate(infos)]
    report = render(runs, EVAL_SET)
    assert "gemini/j (prompt version 1) and groq/openai/gpt-oss-120b (prompt version 1)" in report
    assert "used different judges, so their judged scores are not comparable" in report


# --- Command line ---


def test_command_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = write_manifest(tmp_path / "data")
    runs = tmp_path / "runs"
    write_run(runs, "r1", run_info("groq/a"))
    out = tmp_path / "evals" / "BASELINE.md"
    args = ["--runs-dir", str(runs), "--data", str(data), "--out", str(out)]

    assert main(args) == 0
    assert "| groq/a | 0.70 [0.40, 0.89]" in out.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert f"wrote {out}: 1 model(s)" in printed
    assert "groq/a: 0.70 [0.40, 0.89]  (r1)" in printed

    assert main([str(runs / "r1"), *args]) == 0  # the same run, named by hand


def test_command_refuses_runs_that_do_not_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_manifest(tmp_path / "data")
    runs = tmp_path / "runs"
    good = write_run(runs, "good", run_info("groq/a"))
    again = write_run(runs, "again", run_info("groq/a"))
    dev = write_run(runs, "dev", run_info("groq/b", split="dev", limit=20))
    out = tmp_path / "BASELINE.md"
    common = ["--data", str(data), "--out", str(out)]

    assert main([str(good), str(dev), str(tmp_path / "nowhere"), *common]) == 1
    err = capsys.readouterr().err
    assert "dev: not a baseline run: split is 'dev', not 'test'; a sample of 20 items" in err
    assert "nowhere: cannot read run.json" in err

    assert main([str(good), str(again), *common]) == 1
    assert "more than 1 run of groq/a" in capsys.readouterr().err

    assert main(["--runs-dir", str(tmp_path / "empty"), *common]) == 1
    assert "no full test-split runs" in capsys.readouterr().err
    assert main(["--data", str(tmp_path / "no-data"), "--out", str(out)]) == 1
    assert "cannot read the eval set's manifest" in capsys.readouterr().err
    assert not out.exists()
