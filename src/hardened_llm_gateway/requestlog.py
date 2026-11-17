"""The request log: 1 row per chat request, with numbers only (no prompt or reply text).

Rows are queued and written in batches by 1 background task, so logging never slows
down a request, and a database failure is logged instead of reaching the client.

The table is plain SQLAlchemy Core, so the same code runs on SQLite (the default file
gateway.db) and on Postgres: only GATEWAY_REQUEST_LOG_URL changes.
"""

import asyncio
import contextlib
import dataclasses
import datetime
import logging
from typing import Protocol

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    insert,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("client", String(100), nullable=False),
    Column("requested_model", String(200), nullable=False),
    Column("target", String(200)),  # the "<provider>/<model>" that answered, or tried last
    Column("provider", String(50)),
    Column("stream", Boolean, nullable=False),
    Column("status_code", Integer, nullable=False),  # what the client got
    Column("error_type", String(50)),
    Column("attempts", Integer, nullable=False),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("cost_usd", Float),
    Column("latency_ms", Float, nullable=False),  # whole request, as the client saw it
    Column("first_chunk_ms", Float),  # streamed replies only
)


@dataclasses.dataclass(frozen=True)
class RequestRecord:
    id: str
    created_at: datetime.datetime
    client: str
    requested_model: str
    target: str | None
    provider: str | None
    stream: bool
    status_code: int
    error_type: str | None
    attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    latency_ms: float
    first_chunk_ms: float | None = None


class RequestRecorder(Protocol):
    def record(self, entry: RequestRecord) -> None: ...


class NullRequestLog:
    """Used when the log is turned off (GATEWAY_REQUEST_LOG_URL is empty)."""

    def record(self, entry: RequestRecord) -> None:
        pass


class RequestLog:
    _BATCH = 100

    def __init__(self, url: str, max_queued: int = 10_000) -> None:
        self._url = url
        self._queue: asyncio.Queue[RequestRecord | None] = asyncio.Queue(max_queued)
        self._engine: AsyncEngine | None = None
        self._writer: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create the table if needed and start the background writer."""
        self._engine = create_async_engine(self._url)
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        self._writer = asyncio.create_task(self._write_forever())

    def record(self, entry: RequestRecord) -> None:
        """Queue a row. Never blocks; if the queue is full, the row is dropped and logged."""
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            logger.warning("request log queue is full: dropped the row for %s", entry.id)

    async def close(self, timeout_seconds: float = 10.0) -> None:
        """Write what is still queued, then stop."""
        if self._writer is not None:
            await self._queue.put(None)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._writer, timeout_seconds)
        if self._engine is not None:
            await self._engine.dispose()

    async def _write_forever(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                return
            batch = [first]
            stop = False
            while len(batch) < self._BATCH and not self._queue.empty():
                entry = self._queue.get_nowait()
                if entry is None:
                    stop = True
                    break
                batch.append(entry)
            await self._write(batch)
            if stop:
                return

    async def _write(self, batch: list[RequestRecord]) -> None:
        assert self._engine is not None
        rows = [dataclasses.asdict(entry) for entry in batch]
        try:
            async with self._engine.begin() as connection:
                await connection.execute(insert(requests_table), rows)
        except Exception:
            logger.exception("could not write %d request log rows", len(rows))
