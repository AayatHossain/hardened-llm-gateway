"""Tests for the exact, numeric and code graders."""

from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.evals.dataset import EvalItem, load
from hardened_llm_gateway.evals.graders import (
    extract_code,
    extract_letter,
    extract_number,
    grade,
    grade_code,
)


def make(grader: str, **fields: Any) -> EvalItem:
    return EvalItem.model_validate(
        {
            "id": "t-1",
            "task": "code" if grader == "code" else "facts",
            "split": "test",
            "grader": grader,
            "messages": [{"role": "user", "content": "?"}],
            "source": "tests",
            **fields,
        }
    )


# --- exact: letters ---


@pytest.mark.parametrize(
    ("reply", "letter"),
    [
        ("B", "B"),
        ("b", "B"),
        ("B.", "B"),
        ("**B**", "B"),
        ("(B)", "B"),
        ("(B) buildings will be made safer", "B"),
        ("B. buildings will be made safer", "B"),
        ("**B:** safer", "B"),
        ("The answer is B.", "B"),
        ("Answer: (B)", "B"),
        ("Option A looks right at first, but the answer is B.", "B"),
        ("A good question.\n\nB", "B"),  # "A good" is not option A
        ("Testing changes the design.\nB) safer", "B"),
        ("Let me think.\n\nC. cheaper", "C"),
        ("A. faster\nB. safer\nC. nicer\n\nAll could happen.", None),  # a restated list
        ("I am not sure.", None),
        ("The answer is a planet.", None),
    ],
)
def test_extract_letter(reply: str, letter: str | None) -> None:
    assert extract_letter(reply) == letter


def test_exact_letter_grade() -> None:
    item = make("exact", answer="B")
    assert grade(item, "The answer is B.").passed
    wrong = grade(item, "C")
    assert (wrong.passed, wrong.extracted, wrong.detail) == (False, "C", "expected B, got C")
    assert grade(item, "No idea").detail == "no option letter found"


# --- exact: words ---


@pytest.mark.parametrize(
    ("reply", "passed"),
    [
        ("Dhaka", True),
        ("dhaka.", True),
        ("**Dhaka**", True),
        ("  Dhaka!  ", True),
        ('"Dhaka"', True),
        ("Let me think.\nDhaka", True),  # the last line
        ("Final answer: Dhaka", True),
        ("Dhaka, Bangladesh", False),
        ("It is Dhaka.", False),
        ("Final answer: Chittagong.\n(Not Dhaka.)", False),
    ],
)
def test_exact_words(reply: str, passed: bool) -> None:
    assert grade(make("exact", answer="Dhaka"), reply).passed is passed


def test_exact_yes_no_uses_the_explicit_answer_first() -> None:
    item = make("exact", answer="yes")
    assert grade(item, "Yes.").passed
    assert not grade(item, "Yes, at first.\nThe answer is no").passed
    failed = grade(item, "No")
    assert (failed.extracted, failed.detail) == ("No", "expected 'yes', got 'No'")


# --- numeric ---


@pytest.mark.parametrize(
    ("reply", "number"),
    [
        ("84", "84"),
        ("7 pens cost 7 * 12 = 84 taka.\n\n84", "84"),
        ("The total is $1,234.50.", "1,234.50"),
        ("It drops to -3 degrees", "-3"),
        ("It drops to −3 degrees", "-3"),
        ("Pages 3-4", "4"),
        ("\\boxed{18}", "18"),
        ("Done on 2024-11-19", "19"),
        ("No numbers here", None),
    ],
)
def test_extract_number(reply: str, number: str | None) -> None:
    assert extract_number(reply) == number


@pytest.mark.parametrize(
    ("answer", "reply", "passed"),
    [
        ("72", "Speed = 180 / 2.5 = 72 km/h. **72**", True),
        ("72", "72.0", True),
        ("1234.5", "$1,234.50", True),
        ("1,000", "1000", True),
        ("0.1", "0.10000000001", True),
        ("72", "70", False),
        ("18", "She makes 18 dollars, plus 2 for tips", False),  # the last number counts
    ],
)
def test_numeric_grade(answer: str, reply: str, passed: bool) -> None:
    assert grade(make("numeric", answer=answer), reply).passed is passed


def test_numeric_explains_failures() -> None:
    item = make("numeric", answer="72")
    assert grade(item, "about 70").detail == "expected 72, got 70"
    assert grade(item, "I cannot tell").detail == "no number found"


# --- code: extraction ---


def test_extract_code_prefers_fenced_definitions() -> None:
    reply = (
        "Here it is:\n```python\ndef f(x):\n    return x + 1\n```\n"
        "Example:\n```python\nprint(f(1))\n```\n"
    )
    assert extract_code(reply) == "def f(x):\n    return x + 1"
    assert extract_code("```\nx = 1\n```") == "x = 1"  # no definitions: every block
    assert extract_code("def f():\n    return 1\n") == "def f():\n    return 1"  # no fences


# --- code: running ---

TESTS = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"


def code_item(tests: str = TESTS) -> EvalItem:
    return make("code", tests=tests)


def test_code_passes_when_the_tests_pass() -> None:
    reply = "```python\ndef add(a, b):\n    return a + b\n```"
    result = grade(code_item(), reply)
    assert (result.passed, result.detail) == (True, "")


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        ("def add(a, b):\n    return a - b", "AssertionError"),
        ("def add(a, b)\n    return a + b", "SyntaxError: expected ':'"),
        ("def plus(a, b):\n    return a + b", "NameError: name 'add' is not defined"),
    ],
)
def test_code_failures_name_the_error(reply: str, detail: str) -> None:
    result = grade(code_item(), reply)
    assert not result.passed
    assert result.detail.startswith(detail)


def test_code_has_a_time_limit() -> None:
    reply = "def add(a, b):\n    while True:\n        pass"
    result = grade_code(code_item(), reply, timeout=1)
    assert (result.passed, result.detail) == (False, "timed out after 1 s")


def test_code_cannot_open_sockets() -> None:
    reply = (
        "import socket\n"
        "def add(a, b):\n"
        "    socket.create_connection(('example.com', 80), timeout=1)\n"
        "    return a + b"
    )
    result = grade(code_item(), reply)
    assert not result.passed
    assert "network access is disabled" in result.detail


def test_code_does_not_see_the_gateway_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_SECRET_FOR_TEST", "leak")
    reply = "import os\ndef add(a, b):\n    return a + b\n"
    tests = "assert 'GATEWAY_SECRET_FOR_TEST' not in os.environ\n" + TESTS
    assert grade(code_item(tests), reply).passed


def test_code_demo_blocks_and_input_do_not_hang() -> None:
    reply = (
        "def add(a, b):\n    return a + b\n\n"
        "if __name__ == '__main__':\n    print(add(int(input()), 1))\n"
        "print('lots of output ' * 10000)\n"
    )
    assert grade(code_item(), reply).passed


# --- the shipped eval set ---


def test_every_shipped_answer_passes_its_own_grader() -> None:
    data = Path(__file__).parents[1] / "evals" / "data"
    items = load(sorted(data.glob("*.jsonl"))).items
    checked = [i for i in items if i.grader in ("exact", "numeric")]
    assert len(checked) > 500
    failures = [i.id for i in checked if not grade(i, i.answer or "").passed]
    assert failures == []


# --- dispatch ---


def test_judge_items_are_not_graded_here() -> None:
    with pytest.raises(ValueError, match="grader 'judge' needs a model"):
        grade(make("judge", answer="ref"), "anything")
