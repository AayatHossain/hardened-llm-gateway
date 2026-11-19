"""Graders that score a model's reply to an eval item without calling a model.

- **exact:** the reply's answer equals the item's answer, ignoring case, spacing,
  markdown and end punctuation. Letter answers (multiple choice) also accept forms
  such as "(B)", "B. text" or "The answer is B".
- **numeric:** the last number in the reply equals the answer, so "$1,234.50" matches
  "1234.5" and a worked solution ending in the answer passes.
- **code:** the reply's code runs against the item's tests in a separate Python
  process (see `grade_code` for what that does and does not protect against).

The LLM judge lives in its own module because it calls a model.
"""

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hardened_llm_gateway.evals.dataset import EvalItem, parse_number


@dataclass(frozen=True)
class Grade:
    passed: bool
    extracted: str | None  # what was read from the reply, for failure analysis
    detail: str = ""  # why it failed, e.g. "expected 72, got 70", or the judge's reason
    score: float | None = None  # 0-1 for graders with a scale (judge ratings), else None

    @property
    def value(self) -> float:
        """The score, or 1.0 / 0.0 for pass/fail graders."""
        return self.score if self.score is not None else float(self.passed)


# --- exact ---

_DECORATION = " \t\r\n*_`\"'“”‘’"
_ANSWER_IS = re.compile(r"answer\s*(?:is|:)\s*:?\s*(.+)", re.IGNORECASE)
_LETTER_ANSWER_IS = re.compile(r"(?i:answer\s*(?:is|:))\s*:?\s*[*(\[]*([A-Z])\b(?![a-z'])")
# "(B) text", "[B]", "B. text", "**B:** text" or just "B"; not "A good question".
_LEADING_LETTER = re.compile(r"^\**(?:[(\[]([A-Z])[)\]]|([A-Z])\**(?:[.):]|$))")


def _normalise(text: str) -> str:
    text = text.strip(_DECORATION).rstrip(".!").strip(_DECORATION)
    return " ".join(text.casefold().split())


def _lines(reply: str) -> list[str]:
    return [line.strip() for line in reply.splitlines() if line.strip()]


def extract_letter(reply: str) -> str | None:
    """The chosen option's letter, or None.

    Checked in order:
    1. "answer is X" (the last one);
    2. a reply that is only a letter;
    3. lines that start with a letter and a delimiter ("B. safer", "(B)"), so
       "A good question" is not read as option A. If only 1 line does, that is the
       answer. If several do, the reply probably repeats the options, so only a
       last line counts.
    """
    if matches := _LETTER_ANSWER_IS.findall(reply):
        return str(matches[-1])
    if len(bare := _normalise(reply).strip("()[]")) == 1 and bare.isalpha():
        return bare.upper()
    lines = _lines(reply)
    letters = [(m.group(1) or m.group(2), i) for i, line in enumerate(lines)
               if (m := _LEADING_LETTER.match(line))]  # fmt: skip
    if len(letters) == 1:
        return letters[0][0]
    if letters and letters[-1][1] == len(lines) - 1:
        return letters[-1][0]
    return None


def grade_exact(item: EvalItem, reply: str) -> Grade:
    answer = item.answer or ""
    if re.fullmatch(r"[A-Z]", answer):
        letter = extract_letter(reply)
        if letter is None:
            return Grade(False, None, "no option letter found")
        passed = letter == answer
        return Grade(passed, letter, "" if passed else f"expected {answer}, got {letter}")

    lines = _lines(reply)
    explicit = _ANSWER_IS.findall(lines[-1]) if lines else []
    candidates = [explicit[-1]] if explicit else [reply, *lines[-1:]]
    for candidate in candidates:
        if _normalise(candidate) == _normalise(answer):
            return Grade(True, candidate.strip())
    got = candidates[0].strip()
    return Grade(False, got, f"expected {answer!r}, got {_shorten(got)!r}")


# --- numeric ---

# A minus sign only counts when it is not joined to a word: "3-4" is 3 and 4.
_NUMBER = re.compile(r"(?:(?<!\w)-)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)")


def extract_number(reply: str) -> str | None:
    """The last number in the reply, as written ("1,234.50"), or None."""
    matches = _NUMBER.findall(reply.replace("−", "-"))  # unicode minus
    return str(matches[-1]) if matches else None


def grade_numeric(item: EvalItem, reply: str) -> Grade:
    expected = parse_number(item.answer or "")
    assert expected is not None, "checked when the item was loaded"
    text = extract_number(reply)
    if text is None:
        return Grade(False, None, "no number found")
    got = parse_number(text)
    assert got is not None
    passed = abs(got - expected) <= 1e-6 * max(1.0, abs(expected))
    return Grade(passed, text, "" if passed else f"expected {item.answer}, got {text}")


# --- code ---

CODE_TIMEOUT = 10.0
_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Runs in the child process before the reply's code.
_DRIVER = """\
import socket


def _no_network(*args, **kwargs):
    raise OSError("network access is disabled in the code grader")


class _NoSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        _no_network()


socket.socket = _NoSocket
socket.create_connection = _no_network
socket.getaddrinfo = _no_network

namespace = {"__name__": "solution"}  # so `if __name__ == "__main__":` demos do not run
for name in ("solution.py", "tests.py"):
    with open(name, encoding="utf-8") as file:
        exec(compile(file.read(), name, "exec"), namespace)
"""

# The child gets only what Python needs to start, never the gateway's own settings.
_KEPT_ENV = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")


def extract_code(reply: str) -> str:
    """The code in a reply.

    Fenced blocks that define something (`def` or `class`) are used, so a trailing
    "example usage" block is left out. With no fences, the whole reply is the code.
    """
    blocks = _FENCE.findall(reply)
    if not blocks:
        return reply.strip()
    defining = [b for b in blocks if re.search(r"^\s*(?:async\s+)?(?:def|class)\s", b, re.M)]
    return "\n\n".join(block.strip() for block in defining or blocks)


def grade_code(item: EvalItem, reply: str, timeout: float = CODE_TIMEOUT) -> Grade:
    """Run the reply's code and then the item's tests; pass if nothing raises.

    The code runs in a separate `python -I` process, in an empty temporary folder,
    with a time limit, stdin closed, output discarded, a minimal environment (so no
    API keys) and socket creation blocked.

    This guards against mistakes, not attacks: the code can still read files and
    start processes. It is only meant for replies from the models we evaluate,
    on a machine we control.
    """
    code = extract_code(reply)
    with tempfile.TemporaryDirectory(prefix="grade-", ignore_cleanup_errors=True) as folder:
        work = Path(folder)
        (work / "driver.py").write_text(_DRIVER, encoding="utf-8")
        (work / "solution.py").write_text(code, encoding="utf-8")
        (work / "tests.py").write_text(item.tests or "", encoding="utf-8")
        errors = work / "stderr.txt"
        try:
            with errors.open("wb") as stderr:
                result = subprocess.run(
                    [sys.executable, "-I", "-B", "driver.py"],
                    cwd=work,
                    env={k: os.environ[k] for k in _KEPT_ENV if k in os.environ},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return Grade(False, _shorten(code), f"timed out after {timeout:g} s")
        if result.returncode == 0:
            return Grade(True, _shorten(code))
        last = _lines(errors.read_bytes()[-4000:].decode("utf-8", "replace"))[-1:]
        return Grade(False, _shorten(code), last[0] if last else f"exit code {result.returncode}")


def _shorten(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- dispatch ---


def grade(item: EvalItem, reply: str) -> Grade:
    """Score `reply` with the item's grader. Judge items need `judge.Judge` instead."""
    if item.grader == "exact":
        return grade_exact(item, reply)
    if item.grader == "numeric":
        return grade_numeric(item, reply)
    if item.grader == "code":
        return grade_code(item, reply)
    raise ValueError(f"{item.id}: grader {item.grader!r} needs a model; use the LLM judge")
