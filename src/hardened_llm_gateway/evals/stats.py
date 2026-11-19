"""How sure we can be about eval numbers: confidence intervals and paired comparisons.

    uv run python -m hardened_llm_gateway.evals.stats runs/<run>              # 1 run
    uv run python -m hardened_llm_gateway.evals.stats runs/<run A> runs/<run B>  # A vs B

**1 run.** Each task's mean score gets a 95% confidence interval:
- pass/fail scores use the Wilson interval, which stays sensible near 0% and 100%
  (a bootstrap would say [100%, 100%] after 5 passes out of 5);
- other scores (judge ratings, or a total that mixes them) use a percentile bootstrap.

**2 runs.** Items are paired by id, using only items graded in both runs. Pairing
removes the item-to-item difficulty, so it detects much smaller differences than
comparing 2 separate intervals. The difference B - A gets a paired bootstrap
interval and a two-sided p-value:
- pass/fail: the exact McNemar test (only items where the runs disagree count);
- other scores: a sign-flip permutation test.

Bootstrap and permutation samples are drawn without numpy. Scores take few distinct
values (0 and 1, or ratings / 10), so a resample is a handful of binomial draws, 1 per
distinct value, rather than n random picks. The seed is fixed, so results repeat.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from hardened_llm_gateway.evals.dataset import TASKS
from hardened_llm_gateway.report import percentile

CONFIDENCE = 0.95
RESAMPLES = 10_000
SEED = 0


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int
    method: Literal["wilson", "bootstrap"]


@dataclass(frozen=True)
class Comparison:
    n: int  # items graded in both runs
    a: float  # mean of run A on those items
    b: float
    difference: Estimate  # of B - A
    p_value: float
    test: Literal["mcnemar", "sign-flip"]


def _is_pass_fail(values: Sequence[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values)


def mean_interval(
    values: Sequence[float], confidence: float = CONFIDENCE, seed: int = SEED
) -> Estimate | None:
    """The mean of `values` with its confidence interval; None if there are no values."""
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    if _is_pass_fail(values):
        low, high = wilson(sum(values), n, confidence)
        return Estimate(mean, low, high, n, "wilson")
    low, high = _bootstrap(values, confidence, seed)
    return Estimate(mean, low, high, n, "bootstrap")


def wilson(passed: float, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """The Wilson score interval for a pass rate of passed / n."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def compare(
    a: Sequence[float], b: Sequence[float], confidence: float = CONFIDENCE, seed: int = SEED
) -> Comparison | None:
    """Paired comparison of 2 runs' scores on the same items, in the same order."""
    if len(a) != len(b):
        raise ValueError("compare needs paired scores: a and b differ in length")
    if not a:
        return None
    n = len(a)
    differences = [y - x for x, y in zip(a, b, strict=True)]
    mean = sum(differences) / n
    low, high = _bootstrap(differences, confidence, seed)
    difference = Estimate(mean, low, high, n, "bootstrap")
    a_mean, b_mean = sum(a) / n, sum(b) / n
    if _is_pass_fail(a) and _is_pass_fail(b):
        return Comparison(n, a_mean, b_mean, difference, mcnemar(differences), "mcnemar")
    return Comparison(n, a_mean, b_mean, difference, sign_flip(differences, seed), "sign-flip")


def mcnemar(differences: Sequence[float]) -> float:
    """Exact two-sided McNemar p-value from paired pass/fail differences (-1, 0 or 1)."""
    better = sum(d > 0 for d in differences)
    worse = sum(d < 0 for d in differences)
    disagree = better + worse
    if disagree == 0:
        return 1.0
    tail = sum(math.comb(disagree, k) for k in range(min(better, worse) + 1))
    outcomes: int = 2**disagree  # exact integers: no float overflow for large counts
    return min(1.0, 2 * tail / outcomes)


def sign_flip(differences: Sequence[float], seed: int = SEED, resamples: int = RESAMPLES) -> float:
    """Two-sided p-value that the mean difference is 0, by randomly flipping signs.

    If the 2 runs were equally good, each difference would be as likely to be positive
    as negative. The p-value is how often random signs give a mean at least as far
    from 0 as the real one.
    """
    observed = abs(sum(differences))
    sizes = sorted(Counter(abs(d) for d in differences if d != 0).items())
    if not sizes:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        total = sum(size * (2 * rng.binomialvariate(count, 0.5) - count) for size, count in sizes)
        extreme += abs(total) >= observed - 1e-9
    return (1 + extreme) / (1 + resamples)


def _bootstrap(
    values: Sequence[float], confidence: float, seed: int, resamples: int = RESAMPLES
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean."""
    n = len(values)
    counts = sorted(Counter(values).items())
    rng = random.Random(seed)
    means = sorted(_resampled_sum(counts, n, rng) / n for _ in range(resamples))
    tail = (1 - confidence) / 2 * 100
    low, high = percentile(means, tail), percentile(means, 100 - tail)
    assert low is not None and high is not None
    return low, high


def _resampled_sum(counts: Sequence[tuple[float, int]], n: int, rng: random.Random) -> float:
    """Sum of n values drawn with replacement, as 1 multinomial draw over distinct values."""
    remaining, share_left, total = n, 1.0, 0.0
    for value, count in counts[:-1]:
        share = count / n
        drawn = rng.binomialvariate(remaining, min(1.0, share / share_left)) if remaining else 0
        total += value * drawn
        remaining -= drawn
        share_left -= share
    return total + counts[-1][0] * remaining


# --- Runs ---


@dataclass(frozen=True)
class Run:
    folder: Path
    info: dict[str, Any]  # run.json
    scores: dict[str, tuple[str, float]]  # item id -> (task, score), graded items only

    @property
    def label(self) -> str:
        return f"{self.info.get('model', '?')} ({self.folder.name})"


def load_run(folder: Path) -> Run:
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    scores = {}
    for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("score") is not None:
                scores[record["id"]] = (record["task"], float(record["score"]))
    return Run(folder, info, scores)


def _by_task[T](rows: dict[str, list[T]], all_rows: list[T]) -> dict[str, list[T]]:
    ordered: dict[str, list[T]] = {task: rows[task] for task in TASKS if task in rows}
    ordered["total"] = all_rows
    return ordered


def describe(run: Run, confidence: float = CONFIDENCE) -> str:
    groups: dict[str, list[float]] = {}
    for task, score in run.scores.values():
        groups.setdefault(task, []).append(score)
    every = [score for _, score in run.scores.values()]
    lines = [run.label, "", *_incomplete([run])]
    lines.append(f"{'task':<10}{'graded':>7}{'score':>8}  {_ci_header(confidence)}")
    for name, values in _by_task(groups, every).items():
        estimate = mean_interval(values, confidence)
        if estimate is not None:
            lines.append(
                f"{name:<10}{estimate.n:>7}{estimate.mean:>8.3f}  "
                f"[{estimate.low:.3f}, {estimate.high:.3f}]  {estimate.method}"
            )
    return "\n".join(lines)


def describe_comparison(a: Run, b: Run, confidence: float = CONFIDENCE) -> str:
    shared = sorted(a.scores.keys() & b.scores.keys())
    pairs: dict[str, list[tuple[float, float]]] = {}
    for item_id in shared:
        pairs.setdefault(a.scores[item_id][0], []).append(
            (a.scores[item_id][1], b.scores[item_id][1])
        )
    every = [(a.scores[i][1], b.scores[i][1]) for i in shared]
    lines = [f"A: {a.label}", f"B: {b.label}", ""]
    lines += _warnings(a, b, len(shared))
    lines.append(
        f"{'task':<10}{'paired':>7}{'A':>8}{'B':>8}{'B - A':>9}  "
        f"{_ci_header(confidence):<18}{'p':>8}"
    )
    for name, rows in _by_task(pairs, every).items():
        result = compare([x for x, _ in rows], [y for _, y in rows], confidence)
        if result is None:
            continue
        d = result.difference
        mark = "*" if result.p_value < 1 - confidence else ""
        lines.append(
            f"{name:<10}{result.n:>7}{result.a:>8.3f}{result.b:>8.3f}{d.mean:>+9.3f}  "
            f"[{d.low:+.3f}, {d.high:+.3f}]  {_p(result.p_value):>8} {mark} {result.test}"
        )
    lines.append(
        f"\n* p < {1 - confidence:.2f}. With several tasks, expect an occasional * by "
        "chance; the total row is the main result."
    )
    return "\n".join(lines)


def _incomplete(runs: Sequence[Run]) -> list[str]:
    """A warning for each run that stopped early (see run.QuotaExhausted)."""
    notes = [
        f"warning: {run.folder.name} is incomplete (stopped early): its items are not "
        "the full selection"
        for run in runs
        if run.info.get("complete") is False
    ]
    return [*notes, ""] if notes else []


def _warnings(a: Run, b: Run, shared: int) -> list[str]:
    notes = [line for line in _incomplete([a, b]) if line]
    if shared < max(len(a.scores), len(b.scores)):
        notes.append(
            f"note: {shared} items graded in both runs "
            f"(A has {len(a.scores)}, B has {len(b.scores)}); only those are compared"
        )
    for key in ("judge", "judge_prompt_version"):
        if a.info.get(key) != b.info.get(key):
            notes.append(
                f"warning: different {key} ({a.info.get(key)} vs {b.info.get(key)}): "
                "judged items are not scored the same way"
            )
    return [*notes, ""] if notes else []


def _ci_header(confidence: float) -> str:
    return f"{confidence:.0%} CI"


def _p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hardened_llm_gateway.evals.stats",
        description="Confidence intervals for 1 eval run, or a paired comparison of 2.",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="1 or 2 run folders (runs/...)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    args = parser.parse_args(argv)
    if len(args.runs) > 2:
        parser.error("give 1 run, or 2 to compare")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1, e.g. 0.95")
    try:
        runs = [load_run(folder) for folder in args.runs]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read run: {exc}", file=sys.stderr)
        return 1
    if len(runs) == 1:
        print(describe(runs[0], args.confidence))
    else:
        print(describe_comparison(runs[0], runs[1], args.confidence))
    return 0


if __name__ == "__main__":
    sys.exit(main())
