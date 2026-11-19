"""Eval datasets: JSON Lines files of prompts with known good answers.

Each line is 1 item:

    {"id": "gsm8k-0001", "task": "maths", "split": "test", "grader": "numeric",
     "messages": [{"role": "user", "content": "..."}], "answer": "18",
     "source": "gsm8k (MIT)"}

- `task` is the kind of work (facts, maths, code, reasoning, chat), used to report
  quality per task type.
- `split` is fixed in the file: "dev" for trying ideas, "test" for reported numbers.
- `grader` says how a reply is scored, and decides which fields are required:
  exact and numeric need `answer`; code needs `tests` (Python asserts run against the
  reply's code); judge (an LLM judge) may use `answer` as a reference.
- `source` names where the item came from, with its licence.
- `meta` holds anything else (difficulty, original id, ...).

Unknown fields are an error, so a typo such as "anwser" is caught instead of ignored.

Check a file with `uv run python -m hardened_llm_gateway.evals.dataset <file.jsonl>`.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hardened_llm_gateway.providers import ChatMessage

Task = Literal["facts", "maths", "code", "reasoning", "chat"]
Split = Literal["dev", "test"]
Grader = Literal["exact", "numeric", "code", "judge"]

TASKS: tuple[Task, ...] = ("facts", "maths", "code", "reasoning", "chat")
SPLITS: tuple[Split, ...] = ("dev", "test")


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=100)
    task: Task
    split: Split
    grader: Grader
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    answer: str | None = None
    tests: str | None = None
    source: str = Field(min_length=1)
    meta: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fields_match_grader(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be from the user")
        if self.grader in ("exact", "numeric") and not self.answer:
            raise ValueError(f"grader {self.grader!r} needs an answer")
        if self.grader == "numeric" and self.answer and parse_number(self.answer) is None:
            raise ValueError(f"grader 'numeric' needs a number as answer, got {self.answer!r}")
        if self.grader == "code" and not self.tests:
            raise ValueError("grader 'code' needs tests")
        return self


_PLAIN_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)")


def parse_number(text: str) -> float | None:
    """'1,234.5' -> 1234.5; None if `text` is not a plain number.

    Only digits, thousands commas, a point and a sign: float() alone would also
    accept "Infinity" and "nan", which are answers to some questions.
    """
    text = text.strip()
    if not _PLAIN_NUMBER.fullmatch(text):
        return None
    return float(text.replace(",", ""))


class EvalFileError(ValueError):
    """An eval file could not be read. The message names the file and line."""


@dataclass(frozen=True)
class EvalSet:
    items: tuple[EvalItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def fingerprint(self) -> str:
        """A short hash of the content, independent of line order and formatting.

        Reports record it, so 2 results are only compared when they used the same items.
        """
        digest = hashlib.sha256()
        for item in sorted(self.items, key=lambda i: i.id):
            digest.update(item.model_dump_json(exclude_defaults=True).encode())
            digest.update(b"\n")
        return digest.hexdigest()[:12]

    def select(self, split: Split | None = None, tasks: Iterable[Task] | None = None) -> "EvalSet":
        wanted = set(tasks) if tasks is not None else None
        return EvalSet(
            tuple(
                item
                for item in self.items
                if (split is None or item.split == split)
                and (wanted is None or item.task in wanted)
            )
        )

    def sample(self, n: int, seed: str = "") -> "EvalSet":
        """`n` items chosen by a hash of `seed` and each id, in a stable order.

        The same seed always picks the same items, and adding items to the file
        changes the pick as little as possible (unlike shuffling).
        """

        def rank(item: EvalItem) -> str:
            return hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest()

        return EvalSet(tuple(sorted(self.items, key=rank)[:n]))

    def counts(self) -> Counter[tuple[str, str]]:
        """Items per (task, split)."""
        return Counter((item.task, item.split) for item in self.items)


def load(paths: Path | str | Sequence[Path | str]) -> EvalSet:
    """Read 1 or more .jsonl files. Blank lines are skipped; ids must be unique."""
    if isinstance(paths, Path | str):
        paths = [paths]
    items: list[EvalItem] = []
    seen: dict[str, str] = {}
    for path in map(Path, paths):
        for item, where in _read(path):
            if item.id in seen:
                raise EvalFileError(f"{where}: id {item.id!r} is already used at {seen[item.id]}")
            seen[item.id] = where
            items.append(item)
    return EvalSet(tuple(items))


def _read(path: Path) -> Iterable[tuple[EvalItem, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalFileError(f"{path}: {exc.strerror or exc}") from None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalFileError(f"{where}: not valid JSON ({exc.msg})") from None
        try:
            yield EvalItem.model_validate(data), where
        except ValidationError as exc:
            problems = "; ".join(_describe(error) for error in exc.errors())
            raise EvalFileError(f"{where}: {problems}") from None


def _describe(error: object) -> str:
    assert isinstance(error, dict)
    location = ".".join(str(part) for part in error["loc"])
    message = str(error["msg"]).removeprefix("Value error, ")
    return f"{location}: {message}" if location else message


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m hardened_llm_gateway.evals.dataset <file.jsonl> ...",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        evals = load(args)
    except EvalFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    counts = evals.counts()
    print(f"{len(evals)} items, fingerprint {evals.fingerprint}\n")
    print(f"{'task':<10}" + "".join(f"{s:>7}" for s in SPLITS) + f"{'total':>7}")
    for task in TASKS:
        row = [counts[(task, split)] for split in SPLITS]
        print(f"{task:<10}" + "".join(f"{n:>7}" for n in row) + f"{sum(row):>7}")
    graders = Counter(item.grader for item in evals.items)
    print("\ngraders: " + ", ".join(f"{name} {n}" for name, n in sorted(graders.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
