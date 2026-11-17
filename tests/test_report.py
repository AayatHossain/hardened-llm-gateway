"""Tests for the cost and speed report."""

import asyncio
import datetime
from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.config import get_settings
from hardened_llm_gateway.report import load_rows, main, percentile, summarize, total
from hardened_llm_gateway.requestlog import RequestLog, RequestRecord

NOW = datetime.datetime.now(datetime.UTC)


def row(**changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": "req-1",
        "created_at": NOW,
        "client": "web",
        "requested_model": "groq/m1",
        "target": "groq/m1",
        "provider": "groq",
        "stream": False,
        "status_code": 200,
        "error_type": None,
        "attempts": 1,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost_usd": 0.001,
        "latency_ms": 500.0,
        "first_chunk_ms": None,
    }
    fields.update(changes)
    return fields


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7
    assert percentile([300.0, 100.0, 200.0], 50) == 200
    assert percentile([], 50) is None


def test_lines_per_model() -> None:
    rows = [
        row(latency_ms=400.0),
        row(latency_ms=600.0),
        row(latency_ms=900.0, stream=True, first_chunk_ms=150.0),
        row(target="gemini/m2", provider="gemini", latency_ms=1000.0, cost_usd=0.002),
    ]
    groq, gemini = summarize(rows)
    assert (groq.group, groq.requests, groq.failures) == ("groq/m1", 3, 0)
    assert (groq.prompt_tokens, groq.completion_tokens) == (300, 150)
    assert groq.cost_usd == pytest.approx(0.003)
    assert (groq.p50_ms, groq.p95_ms, groq.first_chunk_p50_ms) == (600, 900, 150)
    assert (gemini.group, gemini.requests, gemini.p50_ms) == ("gemini/m2", 1, 1000)


def test_failures_count_but_do_not_skew_times() -> None:
    rows = [
        row(latency_ms=500.0),
        # Fast rejections and slow provider failures are not "response times".
        row(status_code=400, error_type="invalid_request_error", latency_ms=1.0,
            prompt_tokens=None, completion_tokens=None, cost_usd=None),
        row(status_code=502, error_type="provider_error", latency_ms=30_000.0,
            prompt_tokens=None, completion_tokens=None, cost_usd=None),
        # Failed mid-stream: the client got 200 but the reply broke off.
        row(stream=True, error_type="provider_error", latency_ms=2000.0),
        # The app hung up: logged, but not a failure.
        row(stream=True, error_type="client_disconnected", latency_ms=3000.0),
    ]  # fmt: skip
    (line,) = summarize(rows)
    assert (line.requests, line.failures) == (5, 3)
    assert (line.p50_ms, line.p95_ms) == (500, 500)


def test_grouping_by_client_and_provider() -> None:
    rows = [
        row(client="web"),
        row(client="cli"),
        row(client="cli", requested_model="nope/x", target=None, provider=None,
            status_code=400, error_type="invalid_request_error"),
    ]  # fmt: skip
    assert [(x.group, x.requests) for x in summarize(rows, "client")] == [("cli", 2), ("web", 1)]
    by_provider = summarize(rows, "provider")
    assert [(x.group, x.requests) for x in by_provider] == [("groq", 2), ("nope", 1)]
    by_model = summarize(rows, "model")
    assert [x.group for x in by_model] == ["groq/m1", "nope/x"]
    with pytest.raises(ValueError, match="by must be one of"):
        summarize(rows, "day")


def test_unpriced_replies_are_counted_not_treated_as_free() -> None:
    rows = [row(), row(target="groq/new", cost_usd=None)]
    overall = total(rows)
    assert overall.cost_usd == pytest.approx(0.001)
    assert overall.unpriced == 1


def write_log(path: Path, records: list[dict[str, Any]]) -> str:
    url = f"sqlite+aiosqlite:///{path}"

    async def go() -> None:
        log = RequestLog(url)
        await log.start()
        for n, fields in enumerate(records):
            log.record(RequestRecord(**{**fields, "id": f"req-{n}"}))
        await log.close()

    asyncio.run(go())
    return url


def test_load_rows_filters_by_time(tmp_path: Path) -> None:
    old = NOW - datetime.timedelta(days=10)
    url = write_log(tmp_path / "log.db", [row(), row(created_at=old)])

    assert len(asyncio.run(load_rows(url))) == 2
    assert len(asyncio.run(load_rows(url, NOW - datetime.timedelta(days=1)))) == 1


def test_command_prints_the_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = write_log(
        tmp_path / "log.db",
        [row(), row(target="gemini/m2", provider="gemini", cost_usd=None)],
    )
    assert main(["--url", url]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split()[:3] == ["model", "requests", "failed"]
    assert any(line.startswith("groq/m1") and "0.001000" in line for line in lines)
    assert any(line.startswith("total") and "0.001000*" in line for line in lines)
    assert "1 replies used a model with no known price" in out

    assert main(["--url", url, "--by", "client"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("client")


def test_command_with_no_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "none.db"
    assert main(["--url", f"sqlite+aiosqlite:///{missing}"]) == 1
    assert "No request log yet" in capsys.readouterr().out
    assert not missing.exists()  # the report must not create an empty database


def test_command_with_nothing_in_range(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old = NOW - datetime.timedelta(days=10)
    url = write_log(tmp_path / "log.db", [row(created_at=old)])
    assert main(["--url", url, "--days", "1"]) == 0
    assert "No requests logged in that time." in capsys.readouterr().out


def test_command_with_log_turned_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GATEWAY_REQUEST_LOG_URL", "")

    get_settings.cache_clear()
    try:
        assert main([]) == 1
    finally:
        get_settings.cache_clear()
    assert "turned off" in capsys.readouterr().out
