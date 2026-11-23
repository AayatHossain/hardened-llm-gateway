"""The baseline report: each model's score on the test split, with cost and speed.

    uv run python -m hardened_llm_gateway.evals.baseline              # writes evals/BASELINE.md
    uv run python -m hardened_llm_gateway.evals.baseline runs/a runs/b

With no run folders, it takes each model's newest run in runs/ that counts as a
baseline. A run counts only if it:
- finished every item (not stopped early by a quota);
- covered the whole test split: every task, no --limit sample;
- used the eval set the manifest describes now (the same fingerprint);
- was graded by today's grading rules (`GRADING_VERSION`).

Run folders given by hand must count too; otherwise the command says why not. So
every number in the report is a full test-split result on today's eval set.

The scores and intervals are the ones the runner saved in each run.json. Whether 2
models really differ is a paired question: compare their runs with `evals.stats`.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hardened_llm_gateway.evals.dataset import TASKS
from hardened_llm_gateway.evals.run import GRADING_VERSION

DATA = Path("evals/data")
RUNS = Path("runs")
OUT = Path("evals/BASELINE.md")
TITLES = {
    "maths": "Maths",
    "code": "Code",
    "facts": "Facts",
    "reasoning": "Reasoning",
    "chat": "Chat",
}


class RunError(Exception):
    """A run folder that cannot be read."""


@dataclass(frozen=True)
class EvalSetInfo:
    fingerprint: str
    test_items: int
    test_by_task: dict[str, int]


@dataclass(frozen=True)
class BaselineRun:
    folder: Path
    info: dict[str, Any]  # run.json

    @property
    def model(self) -> str:
        return str(self.info["model"])

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = self.info["summary"]
        return summary


def read_eval_set(data: Path = DATA) -> EvalSetInfo:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    by_task = {entry["task"]: entry["test"] for entry in manifest["files"].values()}
    return EvalSetInfo(manifest["fingerprint"], sum(by_task.values()), by_task)


def read_run(folder: Path) -> BaselineRun:
    try:
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"{folder}: cannot read run.json: {exc}") from exc
    if not isinstance(info, dict) or "model" not in info:
        raise RunError(f"{folder}: run.json is not a run")
    return BaselineRun(folder, info)


def problems(run: BaselineRun, eval_set: EvalSetInfo) -> list[str]:
    """Why `run` is not a baseline run; empty if it is one."""
    info = run.info
    selection = info.get("eval_set") or {}
    reasons = []
    if info.get("complete") is False or info.get("summary") is None:
        reasons.append("it did not finish (stopped early, or still running)")
    elif info["summary"]["total"]["items"] != selection.get("items"):
        reasons.append("not every item finished")
    if selection.get("split") != "test":
        reasons.append(f"split is {selection.get('split')!r}, not 'test'")
    if selection.get("tasks"):
        reasons.append(f"only some tasks ({', '.join(selection['tasks'])})")
    if selection.get("limit") is not None:
        reasons.append(f"a sample of {selection['limit']} items")
    if selection.get("set_fingerprint") != eval_set.fingerprint:
        reasons.append(
            f"eval set {selection.get('set_fingerprint')}, not today's {eval_set.fingerprint}"
        )
    graded = info.get("grading_version", 1)  # runs from before the field are version 1
    if graded != GRADING_VERSION:
        reasons.append(
            f"graded by older rules (version {graded}, now {GRADING_VERSION}); "
            "rerun it: replies and verdicts come from the cache"
        )
    return reasons


def latest_runs(root: Path, eval_set: EvalSetInfo) -> list[BaselineRun]:
    """Each model's newest baseline run in `root`."""
    newest: dict[str, BaselineRun] = {}
    folders = sorted(root.iterdir()) if root.is_dir() else []
    for folder in folders:
        if not (folder / "run.json").is_file():
            continue
        try:
            run = read_run(folder)
        except RunError:
            continue
        if problems(run, eval_set):
            continue
        current = newest.get(run.model)
        if current is None or _started(run) >= _started(current):
            newest[run.model] = run
    return list(newest.values())


def _started(run: BaselineRun) -> str:
    return str(run.info.get("started_at", ""))


# --- The report ---


def render(runs: Sequence[BaselineRun], eval_set: EvalSetInfo) -> str:
    """The report as Markdown, best total score first."""
    ranked = sorted(runs, key=lambda r: (-_mean(r.summary["total"]), r.model))
    judges = sorted({_judge(r) for r in ranked})
    tasks = [t for t in TASKS if t in eval_set.test_by_task]
    counts = ", ".join(f"{eval_set.test_by_task[t]} {t}" for t in tasks)

    lines = [
        "# Baseline",
        "",
        f"Each model on the {eval_set.test_items} test items of the eval set "
        f"(fingerprint `{eval_set.fingerprint}`: {counts}), sent through the gateway.",
        "",
        "Scores run from 0 to 1. Pass/fail tasks score 1 per pass; chat items score the "
        "judge's rating out of 10. Brackets are 95% confidence intervals. Overlapping "
        "intervals do not mean 2 models are equal: compare their runs item by item with "
        "`evals.stats` for that.",
        "",
        f"Judged items (facts without a numeric answer, and chat) were graded by "
        f"{' and '.join(judges)}.",
    ]
    if len(judges) > 1:
        lines.append(
            "**These runs used different judges, so their judged scores are not comparable.**"
        )

    lines += ["", "## Scores", ""]
    lines += _table(
        ["Model", "Total", *(TITLES.get(t, t) for t in tasks)],
        [
            [r.model, _score(r.summary["total"]), *(_task_score(r, t) for t in tasks)]
            for r in ranked
        ],
    )

    lines += ["", "## Cost and speed", ""]
    lines += _table(
        [
            "Model",
            "List price per 1,000 items",
            "Tokens per item (in / out)",
            "Median latency",
            "Cut off",
            "Errors",
            "Ungraded",
        ],
        [_cost_row(r) for r in ranked],
    )
    lines += [
        "",
        "- **List price:** what the replies would cost at the provider's paid prices "
        "(`prices.toml`); the runs themselves used free tiers. It leaves out the judge.",
        "- **Cut off:** replies that hit the token limit, often a reasoning model that "
        "never got to answer. They are graded as they are.",
        "- **Errors:** items the model or provider failed to answer; they score 0.",
        "- **Ungraded:** items the judge could not grade; they are left out of the score.",
    ]

    lines += ["", "## Runs", ""]
    lines += _table(
        ["Model", "Run", "Finished", "Temperature", "Max tokens", "Gateway commit"],
        [_run_row(r) for r in ranked],
    )
    lines += [
        "",
        "Generated by `uv run python -m hardened_llm_gateway.evals.baseline` from the "
        "runs above (kept locally in `runs/`).",
        "",
    ]
    return "\n".join(lines)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _mean(summary: Mapping[str, Any]) -> float:
    mean = summary.get("mean_score")
    return float(mean) if mean is not None else -1.0


def _score(summary: Mapping[str, Any] | None) -> str:
    if summary is None or summary.get("mean_score") is None:
        return "-"
    text = f"{summary['mean_score']:.2f}"
    if summary.get("score_ci") is not None:
        low, high = summary["score_ci"]
        text += f" [{low:.2f}, {high:.2f}]"
    return text


def _task_score(run: BaselineRun, task: str) -> str:
    return _score(run.summary.get(task))


def _judge(run: BaselineRun) -> str:
    judge = run.info.get("judge") or "no judge"
    return f"{judge} (prompt version {run.info.get('judge_prompt_version', '?')})"


def _cost_row(run: BaselineRun) -> list[str]:
    total = run.summary["total"]
    items = total["items"]
    if total.get("cost_usd") is None:
        cost = "unknown"
    else:
        cost = f"${total['cost_usd'] / items * 1000:.2f}"
        if total.get("unpriced"):
            cost += f" ({total['unpriced']} replies unpriced)"
    tokens = f"{total['prompt_tokens'] / items:,.0f} / {total['completion_tokens'] / items:,.0f}"
    latency = "-" if total.get("p50_ms") is None else f"{total['p50_ms'] / 1000:.1f} s"
    return [
        run.model,
        cost,
        tokens,
        latency,
        str(total.get("cut_off", 0)),
        str(total["errors"]),
        str(total["ungraded"]),
    ]


def _run_row(run: BaselineRun) -> list[str]:
    options = run.info.get("options") or {}
    finished = str(run.info.get("finished_at", "-"))[:10]
    return [
        run.model,
        f"`{run.folder.name}`",
        finished,
        str(options.get("temperature", "-")),
        str(options.get("max_tokens", "-")),
        f"`{run.info.get('git_commit') or '-'}`",
    ]


# --- Command line ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hardened_llm_gateway.evals.baseline",
        description="Write the baseline report from full test-split eval runs.",
    )
    parser.add_argument(
        "runs", nargs="*", type=Path, help="run folders (default: each model's newest)"
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS, help="where to look for runs")
    parser.add_argument("--data", type=Path, default=DATA, help="the eval set's folder")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    try:
        eval_set = read_eval_set(args.data)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: cannot read the eval set's manifest: {exc}", file=sys.stderr)
        return 1

    if args.runs:
        runs, rejected = [], []
        for folder in args.runs:
            try:
                run = read_run(folder)
            except RunError as exc:
                rejected.append(str(exc))
                continue
            if reasons := problems(run, eval_set):
                rejected.append(f"{folder}: not a baseline run: {'; '.join(reasons)}")
            runs.append(run)
        models = [r.model for r in runs]
        if duplicates := sorted({m for m in models if models.count(m) > 1}):
            rejected.append(f"more than 1 run of {', '.join(duplicates)}")
        if rejected:
            print("error: " + "\nerror: ".join(rejected), file=sys.stderr)
            return 1
    else:
        runs = latest_runs(args.runs_dir, eval_set)
        if not runs:
            print(
                f"error: no full test-split runs in {args.runs_dir}; run e.g.\n"
                "  uv run python -m hardened_llm_gateway.evals.run "
                "groq/openai/gpt-oss-20b --split test",
                file=sys.stderr,
            )
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(runs, eval_set), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(runs)} model(s)")
    for run in sorted(runs, key=lambda r: r.model):
        print(f"  {run.model}: {_score(run.summary['total'])}  ({run.folder.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
