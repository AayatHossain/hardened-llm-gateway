"""Cost and speed report, read from the request log.

Run `uv run python -m hardened_llm_gateway.report` for 1 line per model with its
requests, failure rate, tokens, cost and response times. Options:

    --by model|client|provider   how to group the lines (default: model)
    --days N                     only the last N days (default: everything)
    --url URL                    another log database (default: GATEWAY_REQUEST_LOG_URL)

Times are percentiles over replies that finished normally: p50 is the typical time,
p95 the time that 95% of replies beat. "First word" is the p50 time to the first
streamed chunk, which is what a user waiting on a streamed reply feels.
"""

import argparse
import asyncio
import datetime
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from hardened_llm_gateway.config import get_settings
from hardened_llm_gateway.requestlog import requests_table

GROUPINGS = ("model", "client", "provider")

# The app hung up mid-stream: its own choice, not a gateway or provider failure.
_NOT_A_FAILURE = "client_disconnected"


@dataclass(frozen=True)
class ReportLine:
    group: str
    requests: int
    failures: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    unpriced: int  # replies with tokens but no known price: not in cost_usd
    p50_ms: float | None
    p95_ms: float | None
    first_chunk_p50_ms: float | None


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Nearest-rank percentile: the smallest value that `percent`% of values are <= to."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(percent / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def _group_key(row: Mapping[str, Any], by: str) -> str:
    if by == "client":
        return str(row["client"])
    if by == "provider":
        # No provider means no target was tried, e.g. an unknown provider name.
        return str(row["provider"] or row["requested_model"].partition("/")[0])
    return str(row["target"] or row["requested_model"])


def summarize(rows: Iterable[Mapping[str, Any]], by: str = "model") -> list[ReportLine]:
    """1 line per group, busiest first. `rows` are request log rows as mappings."""
    if by not in GROUPINGS:
        raise ValueError(f"by must be one of {', '.join(GROUPINGS)}")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row, by), []).append(row)

    lines = [_line(name, members) for name, members in groups.items()]
    return sorted(lines, key=lambda line: (-line.requests, line.group))


def _line(group: str, rows: list[Mapping[str, Any]]) -> ReportLine:
    finished = [r for r in rows if r["error_type"] is None]
    latencies = [r["latency_ms"] for r in finished]
    first_chunks = [r["first_chunk_ms"] for r in finished if r["first_chunk_ms"] is not None]
    return ReportLine(
        group=group,
        requests=len(rows),
        failures=sum(1 for r in rows if r["error_type"] not in (None, _NOT_A_FAILURE)),
        prompt_tokens=sum(r["prompt_tokens"] or 0 for r in rows),
        completion_tokens=sum(r["completion_tokens"] or 0 for r in rows),
        cost_usd=sum(r["cost_usd"] or 0.0 for r in rows),
        unpriced=sum(1 for r in rows if r["cost_usd"] is None and r["prompt_tokens"] is not None),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        first_chunk_p50_ms=percentile(first_chunks, 50),
    )


def total(rows: Iterable[Mapping[str, Any]]) -> ReportLine:
    """1 line over every row (percentiles cannot be added up from the group lines)."""
    return _line("total", list(rows))


async def load_rows(url: str, since: datetime.datetime | None = None) -> list[dict[str, Any]]:
    """Every request log row (or those from `since` on). Empty if there is no log yet."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            exists = await connection.run_sync(
                lambda sync: inspect(sync).has_table(requests_table.name)
            )
            if not exists:
                return []
            query = select(requests_table)
            if since is not None:
                query = query.where(requests_table.c.created_at >= since)
            result = await connection.execute(query)
            return [dict(row._mapping) for row in result]
    finally:
        await engine.dispose()


def format_report(lines: Sequence[ReportLine], overall: ReportLine, by: str) -> str:
    headers = [by, "requests", "failed", "tokens in", "tokens out", "cost (USD)"]
    headers += ["p50", "p95", "first word"]
    table = [headers] + [_cells(line) for line in [*lines, overall]]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def render(row: list[str]) -> str:
        first = row[0].ljust(widths[0])
        rest = (cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True))
        return "  ".join([first, *rest]).rstrip()

    out = [render(headers), "  ".join("-" * w for w in widths)]
    out += [render(_cells(line)) for line in lines]
    out += ["  ".join("-" * w for w in widths), render(_cells(overall))]
    if overall.unpriced:
        out.append(
            f"\n* {overall.unpriced} replies used a model with no known price; "
            "their cost is not included."
        )
    return "\n".join(out)


def _cells(line: ReportLine) -> list[str]:
    failed = f"{line.failures} ({line.failures / line.requests:.0%})" if line.requests else "0"
    cost = f"{line.cost_usd:.6f}" + ("*" if line.unpriced else "")
    return [
        line.group,
        str(line.requests),
        failed,
        f"{line.prompt_tokens:,}",
        f"{line.completion_tokens:,}",
        cost,
        _ms(line.p50_ms),
        _ms(line.p95_ms),
        _ms(line.first_chunk_p50_ms),
    ]


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f} ms"


def _sqlite_file_missing(url: str) -> Path | None:
    parsed = make_url(url)
    database = parsed.database
    if parsed.get_backend_name() != "sqlite" or not database or database == ":memory:":
        return None
    path = Path(database)
    return None if path.exists() else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hardened_llm_gateway.report",
        description="Cost and speed report from the gateway's request log.",
    )
    parser.add_argument(
        "--by", choices=GROUPINGS, default="model", help="1 line per (default: model)"
    )
    parser.add_argument("--days", type=float, help="only the last DAYS days")
    parser.add_argument("--url", help="request log database (SQLAlchemy URL)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.days is not None and args.days <= 0:
        parser.error("--days must be more than 0")
    url = args.url or get_settings().request_log_url
    if not url:
        print("The request log is turned off (GATEWAY_REQUEST_LOG_URL is empty).")
        return 1
    missing = _sqlite_file_missing(url)
    if missing is not None:
        print(f"No request log yet: {missing} does not exist. Send a request first.")
        return 1

    since = None
    if args.days is not None:
        since = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=args.days)
    rows = asyncio.run(load_rows(url, since))
    if not rows:
        print("No requests logged" + (" in that time." if since else " yet."))
        return 0

    lines = summarize(rows, args.by)
    print(format_report(lines, total(rows), args.by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
