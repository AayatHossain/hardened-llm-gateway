"""Tests for the request log, against a real SQLite file in each test's own folder."""

import datetime
import logging
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, select

from hardened_llm_gateway.requestlog import RequestLog, RequestRecord, requests_table

pytestmark = pytest.mark.anyio


def entry(n: int = 1, **changes: Any) -> RequestRecord:
    fields: dict[str, Any] = {
        "id": f"req-{n}",
        "created_at": datetime.datetime(2026, 11, 17, 12, 0, tzinfo=datetime.UTC),
        "client": "web",
        "requested_model": "groq/openai/gpt-oss-20b",
        "target": "groq/openai/gpt-oss-20b",
        "provider": "groq",
        "stream": False,
        "status_code": 200,
        "error_type": None,
        "attempts": 1,
        "prompt_tokens": 81,
        "completion_tokens": 70,
        "cost_usd": 0.00002708,
        "latency_ms": 603.0,
    }
    fields.update(changes)
    return RequestRecord(**fields)


def read_rows(path: Path) -> list[dict[str, Any]]:
    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as connection:
        rows = [dict(r._mapping) for r in connection.execute(select(requests_table))]
    engine.dispose()
    return rows


async def test_rows_are_written_and_flushed_on_close(tmp_path: Path) -> None:
    path = tmp_path / "log.db"
    log = RequestLog(f"sqlite+aiosqlite:///{path}")
    await log.start()
    for n in range(250):  # more than 1 batch
        log.record(entry(n))
    log.record(entry(999, status_code=502, error_type="provider_error", target=None))
    await log.close()

    rows = read_rows(path)
    assert len(rows) == 251
    first = next(r for r in rows if r["id"] == "req-0")
    assert first["cost_usd"] == pytest.approx(0.00002708)
    assert first["stream"] is False
    failed = next(r for r in rows if r["id"] == "req-999")
    assert (failed["status_code"], failed["error_type"], failed["target"]) == (
        502,
        "provider_error",
        None,
    )


async def test_full_queue_drops_rows_instead_of_blocking(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log = RequestLog(f"sqlite+aiosqlite:///{tmp_path / 'log.db'}", max_queued=2)
    # Not started: nothing drains the queue.
    log.record(entry(1))
    log.record(entry(2))
    with caplog.at_level(logging.WARNING):
        log.record(entry(3))
    assert "dropped the row for req-3" in caplog.text


async def test_database_errors_are_logged_not_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "log.db"
    log = RequestLog(f"sqlite+aiosqlite:///{path}")
    await log.start()
    with caplog.at_level(logging.ERROR):
        log.record(entry(1))
        log.record(entry(1))  # same id again: the batch insert fails
        await log.close()
    assert "could not write 2 request log rows" in caplog.text
