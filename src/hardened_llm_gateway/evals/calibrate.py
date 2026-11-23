"""Judge calibration: how often the judge agrees with a person.

    uv run python -m hardened_llm_gateway.evals.calibrate sample runs/<a> runs/<b>
    uv run python -m hardened_llm_gateway.evals.calibrate label
    uv run python -m hardened_llm_gateway.evals.calibrate score

1. **sample** picks judged items (facts and chat) from dev-split runs into
   evals/calibration/labels.jsonl: the question, reference, reply and the judge's
   verdict. Only dev items are used, so tuning the judge on these labels never
   touches the test split. Items are spread evenly over the runs' models and over
   the judge's pass and fail verdicts, so both kinds of judge mistake show up.
2. **label** shows each item in the terminal, without the judge's verdict, and asks
   for yours: c / i / n for facts (the same 3 grades the judge uses), a rating from 1
   to 10 for chat. Each answer is saved at once; quit and resume at any time.
3. **score** compares the 2:
   - facts: agreement and Cohen's kappa over the 3 grades, and pass/fail agreement;
   - chat: pass/fail agreement (pass at 7 or more), the mean rating difference with
     a paired test, and the rank correlation;
   - per model: the judge's pass rate against yours. A judge that grades its own
     model family more kindly than others shows up here.

Because the sample is balanced over the judge's verdicts, the overall agreement is
not the agreement on a typical run. The per-verdict rows are what generalise: "when
the judge says pass, the person agrees X% of the time".

Pass/fail rows score the pipeline as it grades now: a reply cut off at the token
limit fails whatever the judge said (the runner no longer asks the judge about it).
The "judge alone" row shows the agreement without that rule. Grades and ratings are
always the judge's own.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hardened_llm_gateway.evals.dataset import EvalFileError, EvalItem, load
from hardened_llm_gateway.evals.judge import PASS_RATING
from hardened_llm_gateway.evals.run import CUT_OFF, default_files
from hardened_llm_gateway.evals.stats import compare, mean_interval

LABELS = Path("evals/calibration/labels.jsonl")
FACT_GRADES = {"c": "CORRECT", "i": "INCORRECT", "n": "NOT_ATTEMPTED"}
SKIPPED = "SKIPPED"


class CalibrationError(Exception):
    """The runs or labels cannot be used."""


# --- Sampling ---


def judged_records(folder: Path) -> list[dict[str, Any]]:
    """The judged dev items of a run, each with the run's model and judge."""
    try:
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        lines = (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"{folder}: cannot read run: {exc}") from exc
    records = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("grader") == "judge"
            and record.get("split") == "dev"
            and record.get("passed") is not None
            and record.get("error") is None
            and record.get("extracted") != CUT_OFF  # failed without a verdict to check
        ):
            records.append(
                record
                | {
                    "model": info["model"],
                    "judge_model": info.get("judge"),
                    "judge_prompt_version": info.get("judge_prompt_version"),
                }
            )
    return records


def choose(
    records: Sequence[dict[str, Any]], n: int, facts_share: float, seed: str
) -> list[dict[str, Any]]:
    """About n records, facts_share of them facts, spread evenly over model and verdict."""
    n_facts = round(n * facts_share)
    chosen: list[dict[str, Any]] = []
    for task, quota in (("facts", n_facts), ("chat", n - n_facts)):
        cells: dict[tuple[str, bool], list[dict[str, Any]]] = {}
        for record in records:
            if record["task"] == task:
                cells.setdefault((record["model"], record["passed"]), []).append(record)
        queues = [sorted(cell, key=lambda r: _rank(seed, r)) for _, cell in sorted(cells.items())]
        taken = 0
        while taken < quota and any(queues):
            for queue in queues:  # round robin: an empty cell's share goes to the others
                if queue and taken < quota:
                    chosen.append(queue.pop(0))
                    taken += 1
    return sorted(chosen, key=lambda r: _rank(seed + "order", r))  # mix tasks and models


def _rank(seed: str, record: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{record['model']}|{record['id']}".encode()).hexdigest()


def to_label(record: dict[str, Any], item: EvalItem) -> dict[str, Any]:
    judge: str | int = record["extracted"]
    if record["task"] == "chat":
        judge = int(judge)
    return {
        "id": record["id"],
        "task": record["task"],
        "model": record["model"],
        "question": "\n\n".join(m.content for m in item.messages if m.role == "user"),
        "reference": item.answer,
        "reply": record.get("reply", ""),
        "cut_off": record.get("finish_reason") == "length",
        "judge": judge,
        "judge_model": record["judge_model"],
        "judge_prompt_version": record["judge_prompt_version"],
        "human": None,
    }


# --- Label file ---


def read_labels(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CalibrationError(f"cannot read {path}: {exc}") from exc
    return [json.loads(line) for line in lines if line.strip()]


def write_labels(path: Path, labels: Sequence[dict[str, Any]]) -> None:
    """Write via a temporary file, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    text = "".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels)
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


# --- Labelling ---

FACTS_HELP = """\
Grade the reply against the reference answer, as the judge does:
  c  CORRECT        it gives the reference answer (other wording or extra detail is
                    fine) and does not contradict it
  i  INCORRECT      it gives a different answer, or contradicts the reference
  n  NOT_ATTEMPTED  it does not commit to an answer (e.g. "I don't know")"""

CHAT_HELP = f"""\
Rate the reply from 1 to 10: helpfulness, relevance, accuracy, depth, creativity and
detail, as MT-Bench asks. If there is a reference answer, correctness matters most.
{PASS_RATING} or more counts as a pass."""

Ask = Callable[[str], str]
Show = Callable[[str], None]


def label(path: Path, ask: Ask = input, show: Show = print) -> int:
    """Ask for each unlabelled item's verdict; return how many are labelled now."""
    labels = read_labels(path)
    todo = [label for label in labels if label["human"] is None]
    show(f"{len(labels) - len(todo)} of {len(labels)} labelled. s = skip, q = quit.\n")
    for count, item in enumerate(todo, start=len(labels) - len(todo) + 1):
        show(_card(item, count, len(labels)))
        answer = _ask_until_valid(item["task"], ask, show)
        if answer is None:
            break
        item["human"] = answer
        write_labels(path, labels)
    done = sum(label["human"] is not None for label in labels)
    show(f"\n{done} of {len(labels)} labelled; saved in {path}")
    return done


def _card(item: dict[str, Any], count: int, total: int) -> str:
    rule = "-" * 78
    lines = [rule, f"[{count}/{total}] {item['task']} · {item['id']}", rule]
    lines += ["QUESTION", item["question"], ""]
    if item.get("reference"):
        lines += ["REFERENCE ANSWER", item["reference"], ""]
    lines += ["REPLY", item["reply"] or "(empty reply)", ""]
    lines.append(FACTS_HELP if item["task"] == "facts" else CHAT_HELP)
    return "\n".join(lines)


def _ask_until_valid(task: str, ask: Ask, show: Show) -> str | int | None:
    prompt = "grade [c/i/n] > " if task == "facts" else "rating [1-10] > "
    while True:
        try:
            text = ask(prompt).strip().lower()
        except EOFError:
            return None
        if text == "q":
            return None
        if text == "s":
            return SKIPPED
        if task == "facts" and text in FACT_GRADES:
            return FACT_GRADES[text]
        if task == "chat" and text.isdigit() and 1 <= int(text) <= 10:
            return int(text)
        show("  not understood; " + ("type c, i or n" if task == "facts" else "type 1 to 10"))


# --- Scoring ---


@dataclass(frozen=True)
class Agreement:
    n: int
    agreed: int
    low: float  # 95% Wilson interval of the agreement rate
    high: float

    @property
    def rate(self) -> float:
        return self.agreed / self.n


def agreement(pairs: Sequence[tuple[bool, bool]]) -> Agreement | None:
    if not pairs:
        return None
    hits = [float(a == b) for a, b in pairs]
    estimate = mean_interval(hits)
    assert estimate is not None
    return Agreement(len(pairs), int(sum(hits)), estimate.low, estimate.high)


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Agreement beyond chance: 1 is perfect, 0 is what chance alone would give."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in left) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _passes(task: str, verdict: str | float) -> bool:
    if task == "facts":
        return verdict == "CORRECT"
    return int(verdict) >= PASS_RATING


def _judge_passes(label: dict[str, Any]) -> bool:
    """The pipeline's verdict: the judge's, except that a cut-off reply always fails."""
    return _passes(label["task"], label["judge"]) and not label.get("cut_off", False)


def _person_passes(label: dict[str, Any]) -> bool:
    return _passes(label["task"], label["human"])


def score(labels: Sequence[dict[str, Any]]) -> str:
    done = [x for x in labels if x["human"] not in (None, SKIPPED)]
    lines = [
        f"{len(done)} labelled items ({len(labels) - len(done)} unlabelled or skipped)",
        *_judges(done),
    ]
    if cut := sum(bool(x.get("cut_off")) for x in done):
        lines.append(
            f"{cut} replies were cut off at the token limit: they fail whatever the judge "
            "said, as in the runner"
        )
    facts = [x for x in done if x["task"] == "facts"]
    chat = [x for x in done if x["task"] == "chat"]
    if facts:
        lines += ["", "FACTS", *_facts_report(facts)]
    if chat:
        lines += ["", "CHAT", *_chat_report(chat)]
    if done:
        lines += ["", "BY MODEL (pass rates; judge - person > 0 means the judge is kinder)"]
        lines += _model_report(done)
    return "\n".join(lines)


def _judges(done: Sequence[dict[str, Any]]) -> list[str]:
    judges = Counter(
        f"{x['judge_model']} (prompt version {x['judge_prompt_version']})" for x in done
    )
    return [f"judge: {judge}, {n} items" for judge, n in sorted(judges.items())]


def _facts_report(facts: Sequence[dict[str, Any]]) -> list[str]:
    grades = [(x["judge"], x["human"]) for x in facts]
    exact = agreement([(a == b, True) for a, b in grades])
    passing = agreement([(_judge_passes(x), _person_passes(x)) for x in facts])
    kappa = cohen_kappa(grades)
    assert exact is not None and passing is not None and kappa is not None
    lines = [
        f"  same grade:      {_agree(exact)}  Cohen's kappa {kappa:.2f}",
        f"  same pass/fail:  {_agree(passing)}",
        *_judge_alone(facts),
        *_per_verdict(facts),
        "",
        "  judge (rows) by person (columns):",
    ]
    order = list(FACT_GRADES.values())
    counts = Counter(grades)
    lines.append("  " + " " * 15 + "".join(f"{g:>15}" for g in order))
    for row in order:
        lines.append(f"  {row:<15}" + "".join(f"{counts[(row, col)]:>15}" for col in order))
    return lines


def _chat_report(chat: Sequence[dict[str, Any]]) -> list[str]:
    judge = [float(x["judge"]) for x in chat]
    person = [float(x["human"]) for x in chat]
    passing = agreement([(_judge_passes(x), _person_passes(x)) for x in chat])
    assert passing is not None
    lines = [f"  same pass/fail:  {_agree(passing)}", *_judge_alone(chat), *_per_verdict(chat)]
    gap = statistics.fmean(abs(a - b) for a, b in zip(judge, person, strict=True))
    lines.append(f"  mean |judge - person| rating: {gap:.2f}")
    result = compare([p / 10 for p in person], [j / 10 for j in judge])
    if result is not None:
        d = result.difference
        lines.append(
            f"  judge - person rating: {_signed(d.mean * 10)} "
            f"[{_signed(d.low * 10)}, {_signed(d.high * 10)}], p = {result.p_value:.4f}"
        )
    if len(chat) >= 3 and len(set(judge)) > 1 and len(set(person)) > 1:
        rho = statistics.correlation(judge, person, method="ranked")
        lines.append(f"  rank correlation (Spearman): {rho:.2f}")
    return lines


def _judge_alone(items: Sequence[dict[str, Any]]) -> list[str]:
    """Pass/fail agreement without the cut-off rule, when the rule changed anything."""
    if not any(x.get("cut_off") for x in items):
        return []
    alone = agreement([(_passes(x["task"], x["judge"]), _person_passes(x)) for x in items])
    assert alone is not None
    return [f"  judge alone:     {_agree(alone)}  (without the cut-off rule)"]


def _per_verdict(items: Sequence[dict[str, Any]]) -> list[str]:
    lines = []
    for says, word in ((True, "pass"), (False, "fail")):
        group = [x for x in items if _judge_passes(x) == says]
        result = agreement([(says, _person_passes(x)) for x in group])
        if result is not None:
            lines.append(f"  judge says {word}: person agrees {_agree(result)}")
    return lines


def _model_report(done: Sequence[dict[str, Any]]) -> list[str]:
    lines = [f"  {'model':<30}{'items':>6}{'person':>8}{'judge':>8}{'diff':>8}"]
    for model in sorted({x["model"] for x in done}):
        group = [x for x in done if x["model"] == model]
        person = statistics.fmean(_person_passes(x) for x in group)
        judge = statistics.fmean(_judge_passes(x) for x in group)
        lines.append(
            f"  {model:<30}{len(group):>6}{person:>8.0%}{judge:>8.0%}{judge - person:>+8.0%}"
        )
    return lines


def _signed(value: float) -> str:
    return f"{round(value, 2) + 0.0:+.2f}"  # + 0.0 turns -0.0 into 0.0


def _agree(result: Agreement) -> str:
    return f"{result.agreed}/{result.n} = {result.rate:.0%} [{result.low:.0%}, {result.high:.0%}]"


# --- Command line ---


def main(argv: list[str] | None = None, ask: Ask = input) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hardened_llm_gateway.evals.calibrate",
        description="Measure how often the judge agrees with a person.",
    )
    parser.add_argument("--labels", type=Path, default=LABELS)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="pick judged dev items from runs to label")
    sample.add_argument("runs", nargs="+", type=Path)
    sample.add_argument("--n", type=int, default=100)
    sample.add_argument("--facts-share", type=float, default=0.6)
    sample.add_argument("--seed", default="calibration-v1")
    sample.add_argument("--data", nargs="+", type=Path, help="eval files (default: the set)")
    sample.add_argument("--force", action="store_true", help="replace an existing label file")
    commands.add_parser("label", help="label the sampled items in the terminal")
    commands.add_parser("score", help="compare your labels with the judge's")
    args = parser.parse_args(argv)

    try:
        if args.command == "sample":
            return _sample(args)
        if args.command == "label":
            label(args.labels, ask)
            return 0
        print(score(read_labels(args.labels)))
        return 0
    except (CalibrationError, EvalFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _sample(args: argparse.Namespace) -> int:
    if args.labels.exists() and not args.force:
        raise CalibrationError(
            f"{args.labels} exists and may hold your labels; pass --force to replace it"
        )
    if not 0 <= args.facts_share <= 1 or args.n < 1:
        raise CalibrationError("--n must be at least 1 and --facts-share between 0 and 1")
    records = [r for folder in args.runs for r in judged_records(folder)]
    if not records:
        raise CalibrationError("no judged dev items in these runs (run them with --split dev)")
    files = args.data or default_files()
    items = {item.id: item for item in load(files).items}
    chosen = choose(records, args.n, args.facts_share, args.seed)
    labels = [to_label(record, items[record["id"]]) for record in chosen]
    write_labels(args.labels, labels)
    tally = Counter((x["task"], x["model"]) for x in labels)
    print(f"wrote {len(labels)} items to {args.labels}:")
    for (task, model), n in sorted(tally.items()):
        print(f"  {task:<6}{model:<35}{n:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
