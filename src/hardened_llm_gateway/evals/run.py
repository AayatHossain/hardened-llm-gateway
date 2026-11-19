"""Run the eval set against a model, through the gateway, and grade every reply.

    uv run python -m hardened_llm_gateway.evals.run groq/openai/gpt-oss-20b --split dev

- **Through the gateway.** Each item goes to POST /v1/chat/completions of a gateway
  started inside this process: the same app as uvicorn, with the settings in .env.
  So eval calls are retried, priced, logged and traced like any other traffic. The
  runner signs in as the client "eval-runner" with a one-off key.
- **Grading.** Replies are graded by `graders`, and judge items by `judge`. The
  judge's calls also go through the gateway.
- **Rate limits.** Calls to each provider are spaced to stay under its free-tier
  requests per minute (`PROVIDER_RPM`), shared by model and judge calls. Rate limits
  (429) and timeouts (503, 504) are retried, waiting up to 60 s.
- **Cache.** Model replies and judge verdicts are cached in runs/cache/, so a crashed
  or repeated run only pays for what is new.

Each run writes to runs/<time>-<model>/:
- `run.json`: what was run (model, settings, the eval set's fingerprint) and a summary;
- `results.jsonl`: 1 line per item with the reply, grade, tokens, cost and latency,
  written as items finish.
"""

import argparse
import asyncio
import json
import re
import secrets
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from hardened_llm_gateway import __version__
from hardened_llm_gateway.app import create_app
from hardened_llm_gateway.config import ClientConfig, Settings, get_settings
from hardened_llm_gateway.evals.cache import JsonlCache, request_key
from hardened_llm_gateway.evals.dataset import SPLITS, TASKS, EvalFileError, EvalItem, EvalSet, load
from hardened_llm_gateway.evals.graders import grade
from hardened_llm_gateway.evals.judge import DEFAULT_JUDGE, PROMPT_VERSION, Judge, JudgeError
from hardened_llm_gateway.evals.stats import mean_interval
from hardened_llm_gateway.fallback import RetryPolicy
from hardened_llm_gateway.keys import hash_key
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
)
from hardened_llm_gateway.report import percentile

EVAL_CLIENT = "eval-runner"
DATA = Path("evals/data")

# Free-tier requests per minute, a little under the published limits.
PROVIDER_RPM: dict[str, float] = {"groq": 25, "gemini": 8, "nvidia": 35, "openrouter": 15}
DEFAULT_RPM = 10.0

RUN_RETRY = RetryPolicy(max_retries=5, backoff_seconds=2.0, max_wait_seconds=60.0)
# Rate limits and timeouts are worth waiting for. A 502 is not: the gateway has already
# retried the provider, and it also means a bad key or a broken reply.
_RETRY_STATUSES = (429, 503, 504)

Sleep = Callable[[float], Awaitable[None]]


class Pacer:
    """Spaces calls evenly so they stay under `per_minute`."""

    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = float("-inf")

    async def wait(self) -> None:
        now = self._clock()
        start = max(now, self._next)
        self._next = start + self._interval
        if start > now:
            await self._sleep(start - now)


class Pacers:
    """1 pacer per provider, shared by everything that calls it."""

    def __init__(self, overrides: Mapping[str, float] | None = None, sleep: Sleep = asyncio.sleep):
        self._rpm = {**PROVIDER_RPM, **(overrides or {})}
        self._sleep = sleep
        self._pacers: dict[str, Pacer] = {}

    def for_target(self, target: str) -> Pacer:
        provider = target.partition("/")[0]
        if provider not in self._pacers:
            rpm = self._rpm.get(provider, DEFAULT_RPM)
            self._pacers[provider] = Pacer(rpm, sleep=self._sleep)
        return self._pacers[provider]


class PacedProvider:
    """A provider whose chat calls wait for a pacer first (used for the judge)."""

    def __init__(self, inner: Provider, pacer: Pacer) -> None:
        self._inner = inner
        self._pacer = pacer

    @property
    def name(self) -> str:
        return self._inner.name

    async def chat(self, request: ChatRequest) -> ChatResponse:
        await self._pacer.wait()
        return await self._inner.chat(request)

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        return self._inner.stream(request)

    async def list_models(self) -> list[str]:
        return await self._inner.list_models()


# --- The gateway, in this process ---


@dataclass(frozen=True)
class Gateway:
    client: httpx.AsyncClient  # already signed in as EVAL_CLIENT
    key: SecretStr
    providers: Mapping[str, Provider]  # the providers the gateway enabled


@asynccontextmanager
async def in_process_gateway(
    settings: Settings, providers: Mapping[str, Provider] | None = None
) -> AsyncIterator[Gateway]:
    """Start the gateway app inside this process, with a one-off client for the runner."""
    key = "gw-eval-" + secrets.token_urlsafe(32)
    clients = {
        **settings.clients,
        EVAL_CLIENT: ClientConfig(key_sha256=hash_key(key), requests_per_minute=1_000_000),
    }
    app = create_app(settings.model_copy(update={"clients": clients}), providers=providers)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            headers={"Authorization": f"Bearer {key}"},
            timeout=300,
        ) as client,
    ):
        yield Gateway(client, SecretStr(key), app.state.providers)


# --- Asking the model ---


@dataclass(frozen=True)
class RunOptions:
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    concurrency: int = 4


class ModelCallError(Exception):
    """The gateway did not return a reply, even after retrying."""


async def ask_model(
    client: httpx.AsyncClient,
    item: EvalItem,
    options: RunOptions,
    pacer: Pacer,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any]:
    """The model's reply to `item`, with what the gateway measured about it."""
    body = {
        "model": options.model,
        "messages": [m.model_dump() for m in item.messages],
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
    }
    retry_number = 0
    while True:
        await pacer.wait()
        response = await client.post("/v1/chat/completions", json=body)
        if response.status_code == 200:
            return _reply(response)
        message = f"HTTP {response.status_code}: {_error_message(response)}"
        error = ProviderError(
            "gateway",
            message,
            status_code=response.status_code,
            retryable=response.status_code in _RETRY_STATUSES,
            retry_after_seconds=_float(response.headers.get("retry-after")),
        )
        retry_number += 1
        wait = RUN_RETRY.delay(retry_number, error)
        if wait is None:
            raise ModelCallError(message)
        await sleep(wait)


def _reply(response: httpx.Response) -> dict[str, Any]:
    data = response.json()
    usage = data.get("usage") or {}
    headers = response.headers
    return {
        "reply": data["choices"][0]["message"].get("content") or "",
        "finish_reason": data["choices"][0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cost_usd": _float(headers.get("x-gateway-cost-usd")),  # None when unpriced
        "latency_ms": _float(headers.get("x-gateway-latency-ms")),
        "target": headers.get("x-gateway-target"),
        "attempts": int(headers.get("x-gateway-attempts", "1")),
    }


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return response.text[:200]


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None  # e.g. a cost of "unknown"


# --- Running ---


async def run_items(
    items: EvalSet,
    options: RunOptions,
    *,
    client: httpx.AsyncClient,
    judge: Judge | None,
    cache: JsonlCache,
    pacers: Pacers,
    results: Path,
    sleep: Sleep = asyncio.sleep,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Ask, grade and record every item; return the records in the items' order."""
    semaphore = asyncio.Semaphore(options.concurrency)
    pacer = pacers.for_target(options.model)
    records: dict[str, dict[str, Any]] = {}

    async def one(item: EvalItem) -> None:
        async with semaphore:
            record = await _run_one(item, options, client, judge, cache, pacer, sleep)
        records[item.id] = record
        with results.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if progress is not None:
            progress(len(records), len(items), record)

    await asyncio.gather(*(one(item) for item in items.items))
    return [records[item.id] for item in items.items]


async def _run_one(
    item: EvalItem,
    options: RunOptions,
    client: httpx.AsyncClient,
    judge: Judge | None,
    cache: JsonlCache,
    pacer: Pacer,
    sleep: Sleep,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": item.id,
        "task": item.task,
        "split": item.split,
        "grader": item.grader,
        "passed": None,
        "score": None,
        "extracted": None,
        "detail": "",
        "error": None,
    }
    key = request_key(
        kind="reply",
        model=options.model,
        messages=[m.model_dump() for m in item.messages],
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )
    reply = cache.get(key)
    record["cached"] = reply is not None
    if reply is None:
        try:
            reply = await ask_model(client, item, options, pacer, sleep)
        except ModelCallError as exc:  # the model's failure: scored as a fail
            return record | {"passed": False, "score": 0.0, "error": str(exc)}
        cache.put(key, reply)
    record |= reply

    try:
        if item.grader == "judge":
            if judge is None:
                return record | {"detail": "not judged (run without a judge)"}
            result = await judge.grade(item, reply["reply"])
        elif item.grader == "code":
            result = await asyncio.to_thread(grade, item, reply["reply"])
        else:
            result = grade(item, reply["reply"])
    except (JudgeError, ProviderError) as exc:  # the judge's failure: not scored
        return record | {"detail": f"judge failed: {exc}"}
    return record | {
        "passed": result.passed,
        "score": result.value,
        "extracted": result.extracted,
        "detail": result.detail,
    }


# --- Summary ---


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per task and in total.

    `pass_rate` and `mean_score` are over graded items. A model error counts as a
    fail (it is the model's or gateway's fault); an item the judge could not grade
    is left out, and counted in `ungraded`.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["task"], []).append(record)
    ordered: dict[str, list[dict[str, Any]]] = {t: groups[t] for t in TASKS if t in groups}
    ordered["total"] = records
    return {name: _summary(group) for name, group in ordered.items()}


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in records if r["passed"] is not None]
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    interval = mean_interval([r["score"] for r in graded])
    return {
        "items": len(records),
        "graded": len(graded),
        "passed": sum(r["passed"] for r in graded),
        "pass_rate": sum(r["passed"] for r in graded) / len(graded) if graded else None,
        "mean_score": interval.mean if interval else None,
        # 95% confidence interval of mean_score (see evals.stats)
        "score_ci": [interval.low, interval.high] if interval else None,
        "errors": sum(r["error"] is not None for r in records),
        "ungraded": len(records) - len(graded),
        "cached": sum(bool(r.get("cached")) for r in records),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in records),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in records),
        "cost_usd": sum(costs) if costs else None,
        "unpriced": sum(r.get("cost_usd") is None and r["error"] is None for r in records),
        # Replies cut off by max_tokens; often a reasoning model that never got to answer.
        "cut_off": sum(r.get("finish_reason") == "length" for r in records),
        "p50_ms": percentile(latencies, 50),
    }


def format_summary(summary: Mapping[str, Mapping[str, Any]]) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    lines = [
        f"{'task':<10}{'items':>6}{'passed':>8}{'rate':>8}{'score':>7}  {'95% CI':<14}"
        f"{'errors':>7}{'ungraded':>10}{'p50 ms':>8}"
    ]
    for name, s in summary.items():
        score = "-" if s["mean_score"] is None else f"{s['mean_score']:.2f}"
        ci = "-" if s["score_ci"] is None else f"[{s['score_ci'][0]:.2f}, {s['score_ci'][1]:.2f}]"
        p50 = "-" if s["p50_ms"] is None else f"{s['p50_ms']:.0f}"
        lines.append(
            f"{name:<10}{s['items']:>6}{s['passed']:>8}{pct(s['pass_rate']):>8}{score:>7}  "
            f"{ci:<14}{s['errors']:>7}{s['ungraded']:>10}{p50:>8}"
        )
    total = summary["total"]
    cost = "unknown" if total["cost_usd"] is None else f"${total['cost_usd']:.6f}"
    unpriced = f", {total['unpriced']} replies without a price" if total["unpriced"] else ""
    lines.append(
        f"\ntokens {total['prompt_tokens']:,} in, {total['completion_tokens']:,} out; "
        f"cost {cost}{unpriced}; {total['cached']} replies from the cache"
    )
    if total["cut_off"]:
        lines.append(f"{total['cut_off']} replies were cut off at max tokens (--max-tokens)")
    return "\n".join(lines)


# --- The whole run ---


@dataclass(frozen=True)
class RunPaths:
    out: Path = Path("runs")
    cache: Path = Path("runs/cache")


async def run_eval(
    items: EvalSet,
    options: RunOptions,
    *,
    settings: Settings,
    providers: Mapping[str, Provider] | None = None,
    judge_target: str | None = DEFAULT_JUDGE,
    paths: RunPaths | None = None,
    selection: Mapping[str, Any] | None = None,
    rpm: Mapping[str, float] | None = None,
    sleep: Sleep = asyncio.sleep,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> Path:
    """Run `items` against `options.model`; return the run's folder."""
    paths = paths or RunPaths()
    pacers = Pacers(rpm, sleep=sleep)

    async with in_process_gateway(settings, providers) as gateway:
        for target in (options.model, judge_target):
            if target is not None and target.partition("/")[0] not in gateway.providers:
                raise ValueError(f"provider of {target!r} is not enabled (no API key)")
        started = datetime.now(UTC)
        folder = _new_folder(paths.out, f"{started:%Y%m%d-%H%M%S}-{_slug(options.model)}")
        judge = None
        if judge_target is not None and any(i.grader == "judge" for i in items.items):
            via_gateway = OpenAICompatibleProvider(
                "gateway", "http://gateway/v1", gateway.key, gateway.client, timeout_seconds=300
            )
            judge = Judge(
                {"gateway": PacedProvider(via_gateway, pacers.for_target(judge_target))},
                f"gateway/{judge_target}",
                cache=JsonlCache(paths.cache / "judge.jsonl"),
                sleep=sleep,
            )
        meta: dict[str, Any] = {
            "model": options.model,
            "started_at": started.isoformat(timespec="seconds"),
            "gateway_version": __version__,
            "git_commit": _git_commit(),
            "eval_set": {
                "items": len(items),
                "fingerprint": items.fingerprint,
                **(selection or {}),
            },
            "options": asdict(options),
            "judge": judge_target,
            "judge_prompt_version": PROMPT_VERSION,
        }
        _write_json(folder / "run.json", meta)
        records = await run_items(
            items,
            options,
            client=gateway.client,
            judge=judge,
            cache=JsonlCache(paths.cache / "replies.jsonl"),
            pacers=pacers,
            results=folder / "results.jsonl",
            sleep=sleep,
            progress=progress,
        )

    meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["summary"] = summarize(records)
    meta["judge_usage"] = asdict(judge.usage) if judge is not None else None
    _write_json(folder / "run.json", meta)
    return folder


def _new_folder(parent: Path, name: str) -> Path:
    """parent/name, or parent/name-2, -3, ... if a run started in the same second."""
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        folder = parent / (name if n == 1 else f"{name}-{n}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise FileExistsError(parent / name)


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


# --- Command line ---


def default_files() -> list[Path]:
    """The eval set's files, as listed in its manifest."""
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    return [DATA / name for name in manifest["files"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hardened_llm_gateway.evals.run",
        description="Run the eval set against a model through the gateway.",
    )
    parser.add_argument("model", help='"<provider>/<model>", e.g. groq/openai/gpt-oss-20b')
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="dev")
    parser.add_argument("--tasks", help=f"comma-separated, from {', '.join(TASKS)}")
    parser.add_argument("--limit", type=int, help="run a stable sample of this many items")
    parser.add_argument("--seed", default="run", help="picks the --limit sample")
    parser.add_argument("--data", nargs="+", type=Path, help="eval files (default: the set)")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help='"<provider>/<model>"')
    parser.add_argument("--no-judge", action="store_true", help="leave judge items ungraded")
    parser.add_argument("--rpm", type=float, help="requests per minute for the model's provider")
    parser.add_argument("--concurrency", type=int, default=RunOptions.concurrency)
    parser.add_argument("--temperature", type=float, default=RunOptions.temperature)
    parser.add_argument("--max-tokens", type=int, default=RunOptions.max_tokens)
    parser.add_argument("--out", type=Path, default=RunPaths.out)
    args = parser.parse_args(argv)

    tasks = args.tasks.split(",") if args.tasks else None
    if tasks and (unknown := set(tasks) - set(TASKS)):
        parser.error(f"unknown task(s): {', '.join(sorted(unknown))}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        everything = load(args.data or default_files())
    except EvalFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    items = everything.select(None if args.split == "all" else args.split, tasks)
    if args.limit is not None:
        items = items.sample(args.limit, seed=args.seed)
    if not len(items):
        print("error: no items match", file=sys.stderr)
        return 1

    options = RunOptions(args.model, args.temperature, args.max_tokens, args.concurrency)
    selection = {
        "set_fingerprint": everything.fingerprint,
        "split": args.split,
        "tasks": tasks,
        "limit": args.limit,
        "seed": args.seed if args.limit is not None else None,
    }
    rpm = {args.model.partition("/")[0]: args.rpm} if args.rpm else None

    def progress(done: int, total: int, record: dict[str, Any]) -> None:
        if record["error"] or record["detail"].startswith("judge failed"):
            print(f"  {record['id']}: {record['error'] or record['detail']}", file=sys.stderr)
        if done % 10 == 0 or done == total:
            print(f"{done}/{total} done", file=sys.stderr)

    print(f"running {len(items)} items against {args.model}", file=sys.stderr)
    try:
        folder = asyncio.run(
            run_eval(
                items,
                options,
                settings=get_settings(),
                judge_target=None if args.no_judge else args.judge,
                paths=RunPaths(out=args.out, cache=args.out / "cache"),
                selection=selection,
                rpm=rpm,
                progress=progress,
            )
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    print(format_summary(run["summary"]))
    print(f"\nresults: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
