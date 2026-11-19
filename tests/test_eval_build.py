"""Tests for the eval set build, and for the shipped files it produced. No network."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from hardened_llm_gateway.evals import build
from hardened_llm_gateway.evals.build import (
    SOURCES,
    Source,
    SourceChangedError,
    arc_items,
    choose,
    fetch,
    gsm8k_items,
    mbpp_items,
    mtbench_items,
    simpleqa_items,
)
from hardened_llm_gateway.evals.dataset import TASKS, EvalItem, load

DATA = Path(__file__).parents[1] / "evals" / "data"


def lines(*rows: dict[str, Any]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def prompt(item: EvalItem) -> str:
    return item.messages[-1].content


# --- The shipped files ---


def test_shipped_set_matches_its_manifest() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    assert set(files) == {f"{task}.jsonl" for task in TASKS}

    every = load([DATA / name for name in files])
    assert len(every) == manifest["items"] == 1000
    assert every.fingerprint == manifest["fingerprint"]
    for name, entry in files.items():
        part = load(DATA / name)
        assert {i.task for i in part.items} == {entry["task"]}
        assert part.fingerprint == entry["fingerprint"], f"{name} was edited by hand"
        assert (len(part), len(part.select("dev")), len(part.select("test"))) == (
            entry["items"],
            entry["dev"],
            entry["test"],
        )


def test_manifest_records_the_pinned_sources() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for source in SOURCES:
        entry = manifest["files"][f"{source.task}.jsonl"]
        assert (entry["source_sha256"], entry["licence"]) == (source.sha256, source.licence)
    assert sum(s.size for s in SOURCES) == 1000


# --- Converters ---


def test_gsm8k_takes_the_answer_after_the_marks() -> None:
    data = lines(
        {"question": "Q1?", "answer": "3 + 4 = <<3+4=7>>7\n#### 7"},
        {"question": " Q2? ", "answer": "work\n#### 1,234"},
    )
    first, second = gsm8k_items(data, "gsm8k (MIT)")
    assert (first.id, first.answer, second.id, second.answer) == (
        "gsm8k-0000",
        "7",
        "gsm8k-0001",
        "1,234",
    )
    assert prompt(second).startswith("Q2?\n\nSolve the problem.")
    assert (second.grader, second.meta, second.source) == (
        "numeric",
        {"original_index": 1},
        "gsm8k (MIT)",
    )


def test_mbpp_keeps_the_test_split_and_shows_1_test() -> None:
    def row(task_id: int, imports: list[str]) -> dict[str, Any]:
        tests = [f"assert f({n}) == {n}" for n in range(3)]
        return {
            "task_id": task_id,
            "prompt": "Write f.",
            "test_imports": imports,
            "test_list": tests,
        }

    data = json.dumps([row(2, []), row(11, ["import math"]), row(511, [])]).encode()
    (item,) = mbpp_items(data, "mbpp (CC BY 4.0)")
    assert item.id == "mbpp-11"
    assert "assert f(0) == 0" in prompt(item)
    assert "assert f(1)" not in prompt(item)  # the rest stay hidden
    assert item.tests == "import math\nassert f(0) == 0\nassert f(1) == 1\nassert f(2) == 2"


def test_simpleqa_grades_plain_numbers_as_numbers() -> None:
    data = (
        b"metadata,problem,answer\n"
        b"\"{'topic': 'Art', 'answer_type': 'Date', 'urls': ['x']}\",In what year?,1985\n"
        b"\"{'topic': 'Music', 'answer_type': 'Person', 'urls': []}\",Who won?,Ann Lee\n"
        b"\"{'topic': 'Other', 'answer_type': 'Number', 'urls': []}\",How tall?,390 metres\n"
    )
    year, person, height = simpleqa_items(data, "SimpleQA (MIT)")
    assert (year.grader, prompt(year)) == (
        "numeric",
        "In what year?\n\nAnswer with the number only.",
    )
    assert (person.grader, prompt(person), person.answer) == ("judge", "Who won?", "Ann Lee")
    assert height.grader == "judge"
    assert year.meta == {"original_index": 0, "topic": "Art", "answer_type": "Date"}


def test_arc_letters_the_options_and_cleans_ids() -> None:
    rows = [
        {
            "id": "AKDE&ED_2008_4_25",
            "question": "Which?",
            "choices": {"text": ["one", "two", "three"], "label": ["1", "2", "3"]},
            "answerKey": "2",
        }
    ]
    (item,) = arc_items(json.dumps(rows).encode(), "ARC (CC BY-SA 4.0)")
    assert item.id == "arc-akde-ed_2008_4_25"
    assert prompt(item) == (
        "Which?\n\nA. one\nB. two\nC. three\n\nAnswer with the letter of the correct option only."
    )
    assert (item.answer, item.meta) == ("B", {"original_id": "AKDE&ED_2008_4_25"})


def test_mtbench_uses_the_first_turn_and_its_reference() -> None:
    data = lines(
        {"question_id": 81, "category": "writing", "turns": ["Write.", "Rewrite."]},
        {
            "question_id": 101,
            "category": "reasoning",
            "turns": ["Q?", "Q2?"],
            "reference": ["A", "B"],
        },
    )
    writing, reasoning = mtbench_items(data, "MT-Bench (Apache 2.0)")
    assert (len(writing.messages), prompt(writing), writing.answer) == (1, "Write.", None)
    assert (reasoning.answer, reasoning.meta) == (
        "A",
        {"original_id": 101, "category": "reasoning"},
    )


# --- Picking and splitting ---


def numbered(count: int) -> list[EvalItem]:
    return gsm8k_items(
        lines(*({"question": f"Q{n}", "answer": f"#### {n}"} for n in range(count))), "t"
    )


def test_choose_picks_by_id_and_keeps_source_order() -> None:
    items = numbered(100)
    picked = choose(items, 40)
    assert len(picked) == 40
    assert sum(i.split == "dev" for i in picked) == 12  # 30%
    assert [i.id for i in picked] == sorted(i.id for i in picked)
    assert picked == choose(list(reversed(items)), 40)[::-1]  # file order does not matter
    assert picked != choose(items, 40, seed="other")
    with pytest.raises(ValueError, match="only 100 items, 101 wanted"):
        choose(items, 101)


# --- Downloads ---


def source(data: bytes, download: Any = None) -> Source:
    return Source(
        task="maths",
        size=2,
        name="tiny",
        origin="https://example.test/tiny.jsonl",
        licence="MIT",
        sha256=hashlib.sha256(data).hexdigest(),
        download=download or (lambda client: data),
        convert=gsm8k_items,
    )


def offline() -> httpx.Client:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.Client(transport=httpx.MockTransport(refuse))


def test_fetch_checks_the_hash_and_caches(tmp_path: Path) -> None:
    data = b"pinned"
    calls: list[str] = []

    def download(client: httpx.Client) -> bytes:
        calls.append("download")
        return data

    pinned = source(data, download)
    assert fetch(pinned, tmp_path, offline()) == data
    assert fetch(pinned, tmp_path, offline()) == data
    assert calls == ["download"]  # the 2nd read came from the cache

    (tmp_path / "maths.source").write_bytes(b"stale")  # a wrong cached copy is replaced
    assert fetch(pinned, tmp_path, offline()) == data
    assert calls == ["download", "download"]


def test_fetch_refuses_a_changed_source(tmp_path: Path) -> None:
    changed = source(b"pinned", lambda client: b"changed")
    with pytest.raises(SourceChangedError, match="tiny: expected SHA-256"):
        fetch(changed, tmp_path / "cache", offline())
    assert not (tmp_path / "cache").exists()


def test_arc_rows_are_paged_and_429_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", waits.append)
    total = 150
    limited = iter([True, False, False])

    def handler(request: httpx.Request) -> httpx.Response:
        if next(limited):
            return httpx.Response(429)
        offset = int(request.url.params["offset"])
        end = min(offset + int(request.url.params["length"]), total)
        rows = [{"row": {"id": f"q{n}"}} for n in range(offset, end)]
        return httpx.Response(200, json={"rows": rows, "num_rows_total": total})

    data = build.get_arc_rows(httpx.Client(transport=httpx.MockTransport(handler)))
    assert [row["id"] for row in json.loads(data)] == [f"q{n}" for n in range(total)]
    assert waits == [5.0]


def test_build_writes_files_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = lines(*({"question": f"Q{n}", "answer": f"#### {n}"} for n in range(5)))
    monkeypatch.setattr(build, "SOURCES", (source(data),))
    manifest = build.build(tmp_path / "cache", tmp_path, offline())

    written = load(tmp_path / "maths.jsonl")
    assert len(written) == 2
    assert manifest["files"]["maths.jsonl"]["fingerprint"] == written.fingerprint
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
