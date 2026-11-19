"""Tests for the CI eval gate: recording runs and replaying them offline."""

import json
from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.evals.cache import JsonlCache
from hardened_llm_gateway.evals.dataset import EvalItem
from hardened_llm_gateway.evals.gate import ReplayProvider, main, offline_settings
from hardened_llm_gateway.evals.judge import Judge

MODEL = "groq/openai/gpt-oss-20b"
JUDGE = "groq/openai/gpt-oss-120b"


def item(id: str, task: str, grader: str, **fields: Any) -> EvalItem:
    return EvalItem.model_validate(
        {
            "id": id,
            "task": task,
            "split": "dev",
            "grader": grader,
            "messages": [{"role": "user", "content": f"Question {id}?"}],
            "source": "tests",
            **fields,
        }
    )


ITEMS = [
    item("pick", "reasoning", "exact", answer="B"),
    item("sum", "maths", "numeric", answer="84"),
    item("add", "code", "code", tests="assert add(2, 3) == 5"),
    item("who", "facts", "judge", answer="Shakespeare"),
    item("poem", "chat", "judge"),
    item("unjudged", "facts", "judge", answer="Paris"),  # verdict not in the cache
    item("failed", "maths", "numeric", answer="1"),  # the model call failed
]
REPLIES = {
    "pick": "The answer is B.",
    "sum": "7 * 12 = 84",
    "add": "```python\ndef add(a, b):\n    return a + b\n```",
    "who": "William Shakespeare",
    "poem": "Rain on the roof.",
    "unjudged": "Paris",
}
VERDICTS = {
    "who": '{"reason": "Same person.", "grade": "CORRECT"}',
    "poem": '{"reason": "Short.", "rating": 5}',
}


def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An eval file, a run of it, and the judge cache that run filled."""
    data = tmp_path / "set.jsonl"
    data.write_text("".join(i.model_dump_json() + "\n" for i in ITEMS), encoding="utf-8")
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    info = {
        "model": MODEL,
        "judge": JUDGE,
        "options": {"temperature": 0.0, "max_tokens": 512},
    }
    (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
    results = [
        {"id": i, "reply": reply, "finish_reason": "stop", "prompt_tokens": 10,
         "completion_tokens": 5, "error": None}
        for i, reply in REPLIES.items()
    ]  # fmt: skip
    results.append({"id": "failed", "error": "HTTP 502: provider 'groq' failed"})
    (run / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8"
    )
    cache = tmp_path / "runs" / "cache"
    verdicts = JsonlCache(cache / "judge.jsonl")
    judge = Judge({"gateway": ReplayProvider("gateway", {})}, f"gateway/{JUDGE}")
    by_id = {i.id: i for i in ITEMS}
    for item_id, content in VERDICTS.items():
        request = judge.request_for(by_id[item_id], REPLIES[item_id])
        verdicts.put(judge.cache_key(request), {"content": content})
    return data, run, cache


def gate(tmp_path: Path, data: Path, *args: str) -> int:
    paths = ["--recording", str(tmp_path / "rec.json"), "--expected", str(tmp_path / "exp.json")]
    return main([*paths, "--data", str(data), *args])


def test_record_then_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, run, cache = setup(tmp_path)
    assert gate(tmp_path, data, "record", str(run), "--cache", str(cache)) == 0
    assert "recorded 5 items from 1 run(s)" in capsys.readouterr().out

    recording = json.loads((tmp_path / "rec.json").read_text(encoding="utf-8"))
    assert recording["runs"] == [
        {
            "model": MODEL,
            "judge": JUDGE,
            "temperature": 0.0,
            "max_tokens": 512,
            "from_run": "r1",
            "items": ["pick", "sum", "add", "who", "poem"],  # no verdict / failed: left out
        }
    ]
    assert len(recording["exchanges"]) == 7  # 5 replies and 2 verdicts

    expected = json.loads((tmp_path / "exp.json").read_text(encoding="utf-8"))
    grades = {i: (g["passed"], g["score"], g["extracted"]) for i, g in expected[MODEL].items()}
    assert grades == {
        "add": (True, 1.0, "def add(a, b):\n    return a + b"),
        "pick": (True, 1.0, "B"),
        "poem": (False, 0.5, "5"),
        "sum": (True, 1.0, "84"),
        "who": (True, 1.0, "CORRECT"),
    }

    assert gate(tmp_path, data) == 0
    assert "ok: 5 items replayed, every grade as expected" in capsys.readouterr().out


def test_a_changed_grade_fails_the_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, run, cache = setup(tmp_path)
    gate(tmp_path, data, "record", str(run), "--cache", str(cache))
    expected_file = tmp_path / "exp.json"
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    expected[MODEL]["sum"]["passed"] = False  # as if the numeric grader had changed
    expected_file.write_text(json.dumps(expected), encoding="utf-8")
    capsys.readouterr()

    assert gate(tmp_path, data) == 1
    err = capsys.readouterr().err
    assert "FAIL: 1 of 5 grades changed" in err
    assert f"{MODEL} sum: expected passed=False" in err and "got passed=True" in err

    assert gate(tmp_path, data, "--update") == 0  # accepted on purpose
    assert gate(tmp_path, data) == 0


def test_a_changed_prompt_is_a_miss(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, run, cache = setup(tmp_path)
    gate(tmp_path, data, "record", str(run), "--cache", str(cache))
    changed = [i.model_copy(update={"answer": "Marlowe"}) if i.id == "who" else i for i in ITEMS]
    data.write_text("".join(i.model_dump_json() + "\n" for i in changed), encoding="utf-8")
    capsys.readouterr()

    # The judge's prompt includes the answer, so its request no longer matches.
    assert gate(tmp_path, data) == 1
    err = capsys.readouterr().err
    assert "FAIL: 1 requests have no recorded reply" in err
    assert "openai/gpt-oss-120b" in err


def test_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, run, cache = setup(tmp_path)
    assert gate(tmp_path, data) == 1
    assert "cannot read" in capsys.readouterr().err
    assert gate(tmp_path, data, "record", str(tmp_path / "nowhere")) == 1
    assert "cannot read run" in capsys.readouterr().err

    gate(tmp_path, data, "record", str(run), "--cache", str(cache))
    (tmp_path / "exp.json").unlink()
    capsys.readouterr()
    assert gate(tmp_path, data) == 1  # no expected grades: every item counts as changed
    assert "FAIL: 5 of 5 grades changed" in capsys.readouterr().err


def test_offline_settings_ignore_dot_env(tmp_path: Path) -> None:
    Path(".env").write_text(  # conftest runs each test in its own folder
        "GATEWAY_REQUEST_LOG_URL=sqlite+aiosqlite:///x.db\n", encoding="utf-8"
    )
    settings = offline_settings()
    assert settings.request_log_url == ""
    assert settings.tracing == "off"
