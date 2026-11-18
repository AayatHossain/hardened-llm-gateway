"""Tests for the eval dataset format and loader."""

import json
from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.evals.dataset import (
    EvalFileError,
    EvalSet,
    load,
    main,
    parse_number,
)

SAMPLE = Path(__file__).parents[1] / "evals" / "data" / "sample.jsonl"


def item(n: int = 1, **changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": f"t-{n}",
        "task": "maths",
        "split": "test",
        "grader": "numeric",
        "messages": [{"role": "user", "content": "2 + 2?"}],
        "answer": "4",
        "source": "tests",
    }
    fields.update(changes)
    return {k: v for k, v in fields.items() if v is not None}


def write(path: Path, *rows: dict[str, Any] | str) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_shipped_sample_is_valid() -> None:
    evals = load(SAMPLE)
    assert len(evals) == 10
    assert {i.task for i in evals.items} == {"facts", "maths", "code", "reasoning", "chat"}
    assert all(n == 1 for n in evals.counts().values())  # 1 per task and split


def test_load_reads_every_field(tmp_path: Path) -> None:
    path = write(
        tmp_path / "a.jsonl",
        item(1, meta={"difficulty": "easy", "original_id": 7}),
        "",  # blank lines are fine
        item(2, grader="code", task="code", answer=None, tests="assert f() == 1"),
    )
    first, second = load(path).items
    assert first.messages[0].content == "2 + 2?"
    assert first.meta == {"difficulty": "easy", "original_id": 7}
    assert (second.grader, second.answer, second.tests) == ("code", None, "assert f() == 1")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (item(anwser="4"), "anwser: Extra inputs are not permitted"),
        (item(answer=None), "grader 'numeric' needs an answer"),
        (item(answer="four"), "needs a number as answer"),
        (item(grader="exact", answer=""), "grader 'exact' needs an answer"),
        (item(grader="code", task="code"), "grader 'code' needs tests"),
        (item(task="poetry"), "task: Input should be"),
        (item(split="train"), "split: Input should be 'dev' or 'test'"),
        (item(id="Has Spaces"), "id: String should match pattern"),
        (item(messages=[]), "messages: Tuple should have at least 1 item"),
        (item(messages=[{"role": "assistant", "content": "hi"}]), "last message must be"),
        (item(source=""), "source: String should have at least 1 character"),
    ],
)
def test_bad_items_name_the_line_and_the_problem(
    tmp_path: Path, row: dict[str, Any], message: str
) -> None:
    path = write(tmp_path / "bad.jsonl", item(0), row)
    with pytest.raises(EvalFileError, match=r"bad\.jsonl:2: ") as error:
        load(path)
    assert message in str(error.value)


def test_judge_needs_no_answer(tmp_path: Path) -> None:
    path = write(tmp_path / "j.jsonl", item(grader="judge", task="chat", answer=None))
    assert load(path).items[0].answer is None


def test_bad_json_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(EvalFileError, match=r"x\.jsonl:1: not valid JSON"):
        load(write(tmp_path / "x.jsonl", "{not json"))
    with pytest.raises(EvalFileError, match="nope.jsonl"):
        load(tmp_path / "nope.jsonl")


def test_ids_are_unique_across_files(tmp_path: Path) -> None:
    a = write(tmp_path / "a.jsonl", item(1))
    b = write(tmp_path / "b.jsonl", item(2), item(1))
    with pytest.raises(
        EvalFileError, match=r"b\.jsonl:2: id 't-1' is already used at .*a\.jsonl:1"
    ):
        load([a, b])


def test_fingerprint_follows_content_not_layout(tmp_path: Path) -> None:
    one = load(write(tmp_path / "1.jsonl", item(1), item(2)))
    reordered = load(write(tmp_path / "2.jsonl", item(2), "", json.dumps(item(1), indent=None)))
    changed = load(write(tmp_path / "3.jsonl", item(1), item(2, answer="5")))
    assert one.fingerprint == reordered.fingerprint
    assert one.fingerprint != changed.fingerprint
    assert len(one.fingerprint) == 12


def test_select_by_split_and_task() -> None:
    evals = load(SAMPLE)
    assert len(evals.select(split="test")) == 5
    maths_and_code = evals.select(split="dev", tasks=["maths", "code"])
    assert sorted(i.id for i in maths_and_code.items) == ["sample-code-1", "sample-maths-1"]


def test_sample_is_stable_and_grows_gently() -> None:
    items = load(SAMPLE).items
    evals = EvalSet(items)
    picked = [i.id for i in evals.sample(4, seed="ci").items]
    assert picked == [i.id for i in evals.sample(4, seed="ci").items]
    assert picked != [i.id for i in evals.sample(4, seed="other").items]
    # Fewer items in the pool: the ones still there keep their place in the pick.
    smaller = EvalSet(tuple(i for i in items if i.id != picked[0]))
    assert [i.id for i in smaller.sample(3, seed="ci").items] == picked[1:]
    assert len(evals.sample(50)) == 10


@pytest.mark.parametrize(
    ("text", "number"), [("84", 84.0), (" 1,234.5 ", 1234.5), ("-3", -3.0), ("7 pens", None)]
)
def test_parse_number(text: str, number: float | None) -> None:
    assert parse_number(text) == number


def test_command_summarises_a_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(SAMPLE)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("10 items, fingerprint ")
    assert "maths           1      1      2" in out
    assert "graders: code 2, exact 3, judge 2, numeric 3" in out


def test_command_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(write(tmp_path / "bad.jsonl", "{"))]) == 1
    assert "bad.jsonl:1: not valid JSON" in capsys.readouterr().err
    assert main([]) == 2
