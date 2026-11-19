"""The CI eval gate: replay recorded replies through the whole eval pipeline, offline.

    uv run python -m hardened_llm_gateway.evals.gate                 # check (CI runs this)
    uv run python -m hardened_llm_gateway.evals.gate --update        # accept new grades
    uv run python -m hardened_llm_gateway.evals.gate record runs/a runs/b

CI has no API keys, so it cannot ask models anything. Instead, `record` takes runs
already on disk and saves every model reply and judge verdict they used, keyed by
the exact request, in evals/gate/recording.json. The check then runs those items
through the real gateway, runner, graders and judge, with a replay provider that
answers each request from the recording, and compares every grade with
evals/gate/expected.json.

- **Same grades:** the gate passes.
- **Different grades:** a grader, the judge's verdict parsing or the summary changed
  what an item scores. The gate fails and lists each item. If the change is
  intended, run with --update: the diff of expected.json shows what moved.
- **A request with no recording:** a prompt, the judge's prompt, the eval set or the
  request format changed, so recorded replies no longer apply. Re-record from runs
  made with the new version.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any

from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.evals.cache import JsonlCache, request_key
from hardened_llm_gateway.evals.dataset import EvalFileError, EvalItem, EvalSet, load
from hardened_llm_gateway.evals.judge import Judge
from hardened_llm_gateway.evals.run import RunOptions, RunPaths, default_files, run_eval
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ProviderError,
    Usage,
)

RECORDING = Path("evals/gate/recording.json")
EXPECTED = Path("evals/gate/expected.json")
CACHE = Path("runs/cache")
FORMAT_VERSION = 1
NO_PACING = 1e9  # requests per minute: replays need no spacing


class GateError(Exception):
    """The recording or runs cannot be used."""


def exchange_key(request: ChatRequest) -> str:
    """What identifies a request as a provider receives it."""
    return request_key(
        kind="exchange",
        model=request.model,
        messages=[[m.role, m.content] for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )


class ReplayProvider:
    """Answers each request from a recording; a request it has not seen is a miss."""

    def __init__(self, name: str, exchanges: Mapping[str, Mapping[str, Any]]) -> None:
        self._name = name
        self._exchanges = exchanges
        self.misses: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, request: ChatRequest) -> ChatResponse:
        recorded = self._exchanges.get(exchange_key(request))
        if recorded is None:
            last = request.messages[-1].content.replace("\n", " ")
            self.misses.append(f"{request.model}: {last[:80]!r}")
            raise ProviderError(self._name, "no recorded reply", status_code=400)
        prompt, completion = recorded["prompt_tokens"], recorded["completion_tokens"]
        return ChatResponse(
            id="replay",
            provider=self._name,
            model=request.model,
            content=recorded["content"],
            finish_reason=recorded["finish_reason"],
            usage=Usage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
            ),
            latency_ms=0.0,
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        raise ProviderError(self._name, "the replay provider does not stream")
        yield  # pragma: no cover  (makes this an async generator)

    async def list_models(self) -> list[str]:
        return []


# --- Recording ---


def record(runs: Sequence[Path], cache: Path, items: Mapping[str, EvalItem]) -> dict[str, Any]:
    """The recording of `runs`: their settings, items, replies and judge verdicts.

    An item is left out if its reply failed, or its judge verdict is not in the cache.
    """
    groups: list[dict[str, Any]] = []
    exchanges: dict[str, dict[str, Any]] = {}
    verdicts = JsonlCache(cache / "judge.jsonl")
    for folder in runs:
        info, results = _read_run(folder)
        options = info["options"]
        judge = (
            Judge({"gateway": ReplayProvider("gateway", {})}, f"gateway/{info['judge']}")
            if info.get("judge")
            else None
        )
        kept: list[str] = []
        for result in results:
            item = items.get(result["id"])
            if item is None or result.get("error") is not None or "reply" not in result:
                continue
            judged = item.grader == "judge"
            if judged and judge is None:
                continue
            asked = ChatRequest(
                model=_model_name(info["model"]),
                messages=list(item.messages),
                temperature=options["temperature"],
                max_tokens=options["max_tokens"],
            )
            reply = {
                "content": result["reply"],
                "finish_reason": result.get("finish_reason"),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            }
            if judged:
                assert judge is not None
                judging = judge.request_for(item, result["reply"])
                verdict = verdicts.get(judge.cache_key(judging))
                if verdict is None:
                    continue  # never judged, or judged before the cache existed
                as_sent = judging.model_copy(update={"model": _model_name(judging.model)})
                exchanges[exchange_key(as_sent)] = {
                    "content": verdict["content"],
                    "finish_reason": "stop",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            exchanges[exchange_key(asked)] = reply
            kept.append(item.id)
        groups.append(
            {
                "model": info["model"],
                "judge": info.get("judge"),
                "temperature": options["temperature"],
                "max_tokens": options["max_tokens"],
                "from_run": folder.name,
                "items": kept,
            }
        )
    return {"version": FORMAT_VERSION, "runs": groups, "exchanges": exchanges}


def _read_run(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        lines = (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{folder}: cannot read run: {exc}") from exc
    return info, [json.loads(line) for line in lines if line.strip()]


def _model_name(target: str) -> str:
    """The model name a provider sees: "groq/openai/gpt-oss-20b" -> "openai/gpt-oss-20b"."""
    return target.partition("/")[2]


# --- Replaying ---


async def replay(recording: Mapping[str, Any], items: Mapping[str, EvalItem]) -> dict[str, Any]:
    """Run every recorded item through the pipeline; return grades and misses."""
    if recording.get("version") != FORMAT_VERSION:
        raise GateError(f"recording format {recording.get('version')}, expected {FORMAT_VERSION}")
    grades: dict[str, dict[str, Any]] = {}
    misses: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for group in recording["runs"]:
            missing = [i for i in group["items"] if i not in items]
            if missing:
                raise GateError(f"items not in the eval set: {', '.join(missing[:5])}")
            names = {group["model"].partition("/")[0]}
            if group["judge"]:
                names.add(group["judge"].partition("/")[0])
            providers = {name: ReplayProvider(name, recording["exchanges"]) for name in names}
            folder = await run_eval(
                EvalSet(tuple(items[i] for i in group["items"])),
                RunOptions(group["model"], group["temperature"], group["max_tokens"]),
                settings=offline_settings(),
                providers=providers,
                judge_target=group["judge"],
                paths=RunPaths(out=Path(tmp), cache=Path(tmp) / f"cache-{len(grades)}"),
                rpm=dict.fromkeys(names, NO_PACING),
            )
            grades[group["model"]] = _grades(folder)
            misses += [miss for provider in providers.values() for miss in provider.misses]
    return {"grades": grades, "misses": misses}


def offline_settings() -> Settings:
    """The gateway's defaults, ignoring .env: no request log, no tracing, no keys."""
    # _env_file is pydantic-settings' own init option; mypy's view of Settings lacks it.
    return Settings(_env_file=None, request_log_url="", tracing="off")  # type: ignore[call-arg]


def _grades(folder: Path) -> dict[str, Any]:
    grades = {}
    for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        grades[result["id"]] = {
            "passed": result["passed"],
            "score": result["score"],
            "extracted": result["extracted"],
            "error": result["error"],
        }
    return dict(sorted(grades.items()))


def differences(expected: Mapping[str, Any], got: Mapping[str, Any]) -> list[str]:
    lines = []
    for model in sorted(expected.keys() | got.keys()):
        old, new = expected.get(model, {}), got.get(model, {})
        for item_id in sorted(old.keys() | new.keys()):
            if old.get(item_id) != new.get(item_id):
                lines.append(
                    f"  {model} {item_id}: expected {_brief(old.get(item_id))}, "
                    f"got {_brief(new.get(item_id))}"
                )
    return lines


def _brief(grade: Mapping[str, Any] | None) -> str:
    if grade is None:
        return "nothing"
    text = f"passed={grade['passed']} score={grade['score']} extracted={grade['extracted']!r}"
    return text + (f" error={grade['error']!r}" if grade.get("error") else "")


# --- Command line ---


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hardened_llm_gateway.evals.gate",
        description="Replay recorded replies through the eval pipeline and check the grades.",
    )
    parser.add_argument("--recording", type=Path, default=RECORDING)
    parser.add_argument("--expected", type=Path, default=EXPECTED)
    parser.add_argument(
        "--data", action="append", type=Path, help="an eval file, repeatable (default: the set)"
    )
    parser.add_argument("--update", action="store_true", help="save the grades as expected")
    commands = parser.add_subparsers(dest="command")
    recorder = commands.add_parser("record", help="record runs, then save their grades")
    recorder.add_argument("runs", nargs="+", type=Path)
    recorder.add_argument("--cache", type=Path, default=CACHE, help="where judge.jsonl is")
    args = parser.parse_args(argv)

    try:
        items = {item.id: item for item in load(args.data or default_files()).items}
        if args.command == "record":
            recording = record(args.runs, args.cache, items)
            _write_json(args.recording, recording)
            kept = sum(len(group["items"]) for group in recording["runs"])
            print(f"recorded {kept} items from {len(args.runs)} run(s) in {args.recording}")
        else:
            recording = _read_json(args.recording)
        result = asyncio.run(replay(recording, items))
    except (GateError, EvalFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result["misses"]:
        print(
            f"FAIL: {len(result['misses'])} requests have no recorded reply. A prompt, the "
            "eval set or the request format changed; record new runs "
            "(python -m hardened_llm_gateway.evals.gate record ...).",
            file=sys.stderr,
        )
        for miss in result["misses"][:10]:
            print(f"  {miss}", file=sys.stderr)
        return 1
    total = sum(len(g) for g in result["grades"].values())
    if args.update or args.command == "record":
        _write_json(args.expected, result["grades"])
        print(f"saved the grades of {total} items in {args.expected}")
        return 0
    changed = differences(_read_expected(args.expected), result["grades"])
    if changed:
        print(
            f"FAIL: {len(changed)} of {total} grades changed. If that is intended, run with "
            "--update and review the diff of expected.json.",
            file=sys.stderr,
        )
        print("\n".join(changed), file=sys.stderr)
        return 1
    print(f"ok: {total} items replayed, every grade as expected")
    return 0


def _read_expected(path: Path) -> dict[str, Any]:
    try:
        expected: dict[str, Any] = _read_json(path)
    except GateError:
        return {}
    return expected


if __name__ == "__main__":
    sys.exit(main())
