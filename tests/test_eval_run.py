"""Tests for the eval runner: the real gateway in-process, with scripted providers."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest

from hardened_llm_gateway.config import Settings
from hardened_llm_gateway.evals.cache import JsonlCache
from hardened_llm_gateway.evals.dataset import EvalItem, EvalSet
from hardened_llm_gateway.evals.run import (
    ModelCallError,
    Pacer,
    QuotaExhausted,
    RunOptions,
    RunPaths,
    ask_model,
    format_stop,
    format_summary,
    main,
    run_eval,
    summarize,
)
from hardened_llm_gateway.providers import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ProviderError,
    Usage,
)

pytestmark = pytest.mark.anyio

MODEL = "groq/openai/gpt-oss-20b"  # priced in prices.toml
JUDGE = "gemini/gemini-3.8-flash"
CORRECT = '{"reason": "Right.", "grade": "CORRECT"}'


class Scripted:
    """Answers by the first `needle` found in the last message, or raises it."""

    def __init__(self, name: str, answers: dict[str, str | ProviderError]) -> None:
        self._name = name
        self.answers = answers
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, request: ChatRequest) -> ChatResponse:
        prompt = request.messages[-1].content
        self.prompts.append(prompt)
        for needle, answer in self.answers.items():
            if needle in prompt:
                if isinstance(answer, ProviderError):
                    raise answer
                return ChatResponse(
                    id="r",
                    provider=self._name,
                    model=request.model,
                    content=answer,
                    finish_reason="stop",
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                    latency_ms=12.0,
                )
        raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        chunks: list[ChatChunk] = []
        for chunk in chunks:
            yield chunk

    async def list_models(self) -> list[str]:
        return []


def item(id: str, task: str, grader: str, prompt: str, **fields: Any) -> EvalItem:
    return EvalItem.model_validate(
        {
            "id": id,
            "task": task,
            "split": "dev",
            "grader": grader,
            "messages": [{"role": "user", "content": prompt}],
            "source": "tests",
            **fields,
        }
    )


ITEMS = EvalSet(
    (
        item("france", "facts", "exact", "Capital of France?", answer="Paris"),
        item("spain", "facts", "exact", "Capital of Spain?", answer="Madrid"),
        item("pens", "maths", "numeric", "7 pens at 12 taka?", answer="84"),
        item("add", "code", "code", "Write add(a, b).", tests="assert add(2, 3) == 5"),
        item("hamlet", "facts", "judge", "Who wrote Hamlet?", answer="William Shakespeare"),
        item("haiku", "chat", "judge", "Write a haiku about rain."),
        item("broken", "reasoning", "exact", "A broken question", answer="B"),
    )
)


def providers() -> tuple[Scripted, Scripted]:
    model = Scripted(
        "groq",
        {
            "France": "Paris",
            "Spain": "Barcelona",
            "pens": "7 * 12 = 84",
            "add(a, b)": "```python\ndef add(a, b):\n    return a + b\n```",
            "Hamlet": "Shakespeare wrote it.",
            "haiku": "Soft rain on tin roofs",
            "broken": ProviderError("groq", "HTTP 401", status_code=401),
        },
    )
    judge = Scripted(
        "gemini",
        {"Shakespeare wrote it.": CORRECT, "Soft rain": '{"reason": "Nice.", "rating": 8}'},
    )
    return model, judge


async def no_sleep(seconds: float) -> None:
    return None


async def run(tmp_path: Path, model: Scripted, judge: Scripted, **options: Any) -> Path:
    return await run_eval(
        ITEMS,
        options.pop("run_options", RunOptions(MODEL)),
        settings=Settings(request_log_url=""),
        providers={"groq": model, "gemini": judge},
        judge_target=options.pop("judge_target", JUDGE),
        paths=RunPaths(out=tmp_path / "runs", cache=tmp_path / "runs" / "cache"),
        sleep=no_sleep,
        **options,
    )


def results(folder: Path) -> dict[str, dict[str, Any]]:
    lines = (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in map(json.loads, lines)}


# --- A whole run ---


async def test_a_run_grades_every_kind_of_item(tmp_path: Path) -> None:
    model, judge = providers()
    folder = await run(tmp_path, model, judge, selection={"split": "dev"})
    records = results(folder)

    passed = {id: r["passed"] for id, r in records.items()}
    assert passed == {
        "france": True,
        "spain": False,
        "pens": True,
        "add": True,
        "hamlet": True,
        "haiku": True,
        "broken": False,
    }
    assert records["spain"]["detail"] == "expected 'Madrid', got 'Barcelona'"
    assert (records["haiku"]["score"], records["haiku"]["extracted"]) == (0.8, "8")
    assert records["broken"]["error"] == "HTTP 502: provider 'groq' failed"

    pens = records["pens"]
    assert pens["reply"] == "7 * 12 = 84"
    assert (pens["target"], pens["attempts"], pens["cached"]) == (MODEL, 1, False)
    assert (pens["prompt_tokens"], pens["completion_tokens"], pens["latency_ms"]) == (100, 10, 12)
    assert pens["cost_usd"] > 0  # priced by the gateway

    run_info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run_info["model"] == MODEL
    assert run_info["eval_set"] == {"items": 7, "fingerprint": ITEMS.fingerprint, "split": "dev"}
    assert run_info["judge"] == JUDGE
    assert run_info["judge_usage"]["calls"] == 2
    total = run_info["summary"]["total"]
    assert (total["items"], total["passed"], total["errors"], total["ungraded"]) == (7, 5, 1, 0)
    assert list(run_info["summary"]) == ["facts", "maths", "code", "reasoning", "chat", "total"]


async def test_judge_prompts_go_through_the_gateway(tmp_path: Path) -> None:
    model, judge = providers()
    await run(tmp_path, model, judge)
    assert len(judge.prompts) == 2
    assert all("<reply>" in p or "<predicted_answer>" in p for p in judge.prompts)


async def test_a_rerun_only_pays_for_what_is_new(tmp_path: Path) -> None:
    model, judge = providers()
    await run(tmp_path, model, judge)
    asked = len(model.prompts)

    again = results(await run(tmp_path, model, judge))
    assert len(model.prompts) == asked + 1  # only the failed item is asked again
    assert len(judge.prompts) == 2  # every verdict came from the cache
    assert {id for id, r in again.items() if not r["cached"]} == {"broken"}
    assert again["pens"]["cost_usd"] > 0  # the original cost is kept


async def test_without_a_judge_judge_items_stay_ungraded(tmp_path: Path) -> None:
    model, judge = providers()
    records = results(await run(tmp_path, model, judge, judge_target=None))
    assert records["haiku"]["passed"] is None
    assert records["haiku"]["detail"] == "not judged (run without a judge)"
    assert judge.prompts == []


async def test_a_disabled_provider_stops_the_run_early(tmp_path: Path) -> None:
    model, _ = providers()
    with pytest.raises(ValueError, match="'gemini/gemini-3.8-flash' is not enabled"):
        await run_eval(
            ITEMS,
            RunOptions(MODEL),
            settings=Settings(request_log_url=""),
            providers={"groq": model},
            judge_target=JUDGE,
            paths=RunPaths(out=tmp_path / "runs"),
        )
    assert not (tmp_path / "runs").exists()
    assert model.prompts == []


def quota(provider: str) -> ProviderError:
    """What a provider raises when its daily quota is used up."""
    return ProviderError(
        provider, "HTTP 429", status_code=429, retryable=True, retry_after_seconds=7200
    )


ONE_AT_A_TIME = RunOptions(MODEL, concurrency=1)  # items run in order


async def test_a_daily_quota_stops_the_run_without_scoring(tmp_path: Path) -> None:
    model, judge = providers()
    model.answers["pens"] = quota("groq")  # the 3rd item
    folder = await run(tmp_path, model, judge, run_options=ONE_AT_A_TIME)

    assert list(results(folder)) == ["france", "spain"]  # no fail recorded for "pens"
    assert len(model.prompts) == 3  # nothing after the quota was asked
    run_info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run_info["complete"] is False
    assert run_info["stopped"] == {
        "reason": "HTTP 429: provider 'groq' rate limit reached",
        "retry_after_seconds": 7200,
        "not_run": 5,
    }
    assert run_info["summary"]["total"]["items"] == 2


async def test_a_stopped_run_is_finished_by_running_it_again(tmp_path: Path) -> None:
    model, judge = providers()
    judge.answers["Soft rain"] = quota("gemini")  # the judge runs out on "haiku"
    first = await run(tmp_path, model, judge, run_options=ONE_AT_A_TIME)
    first_info = json.loads((first / "run.json").read_text(encoding="utf-8"))
    assert "haiku" not in results(first)
    assert first_info["stopped"]["reason"].startswith("judge: gateway: HTTP 429")
    assert first_info["stopped"]["not_run"] == 2  # "haiku" and "broken"

    asked = len(model.prompts)
    judge.answers["Soft rain"] = '{"reason": "Nice.", "rating": 8}'  # the next day
    second = await run(tmp_path, model, judge, run_options=ONE_AT_A_TIME)
    second_info = json.loads((second / "run.json").read_text(encoding="utf-8"))
    assert (second_info["complete"], second_info["stopped"]) == (True, None)
    assert results(second)["haiku"]["score"] == 0.8
    # Only "broken" is asked: the 1st run never reached it. The "haiku" reply was
    # cached before its judge ran out, so only the verdict is new.
    assert len(model.prompts) == asked + 1
    assert results(second)["haiku"]["cached"] is True


def test_stop_message() -> None:
    stopped = {"reason": "HTTP 429: slow", "retry_after_seconds": 7200, "not_run": 5}
    assert format_stop(stopped) == (
        "stopped early: HTTP 429: slow\n"
        "5 items were not run, and this run is incomplete. Finished items are cached, "
        "so run the same command again in 2 h or later to finish."
    )
    assert "again in 3 h 1 min or later" in format_stop(stopped | {"retry_after_seconds": 10801})
    assert "again in 1 min or later" in format_stop(stopped | {"retry_after_seconds": 5})
    assert "again later" in format_stop(stopped | {"retry_after_seconds": None})


# --- Asking the model ---


def gateway_client(*responses: httpx.Response) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    queue = list(responses)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gw")
    return client, seen


OK = httpx.Response(
    200,
    json={
        "choices": [{"message": {"content": "84"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1},
    },
    headers={
        "x-gateway-target": "groq/m",
        "x-gateway-attempts": "2",
        "x-gateway-latency-ms": "250",
        "x-gateway-cost-usd": "unknown",
    },
)


async def test_rate_limits_are_waited_out() -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    limited = httpx.Response(
        429, json={"error": {"message": "slow down"}}, headers={"retry-after": "7"}
    )
    client, seen = gateway_client(limited, OK)
    pens = ITEMS.items[2]
    fast = Pacer(60_000)  # the retry also waits for the pacer: keep that short
    reply = await ask_model(client, pens, RunOptions("groq/m", max_tokens=99), fast, sleep)
    assert waits == [7.0]
    assert json.loads(seen[0].content) == {
        "model": "groq/m",
        "messages": [{"role": "user", "content": "7 pens at 12 taka?"}],
        "temperature": 0.0,
        "max_tokens": 99,
    }
    assert reply == {
        "reply": "84",
        "finish_reason": "stop",
        "prompt_tokens": 9,
        "completion_tokens": 1,
        "cost_usd": None,
        "latency_ms": 250.0,
        "target": "groq/m",
        "attempts": 2,
    }


async def test_a_rate_limit_too_long_to_wait_is_a_quota() -> None:
    daily = httpx.Response(
        429, json={"error": {"message": "daily limit"}}, headers={"retry-after": "3600"}
    )
    client, seen = gateway_client(daily)
    with pytest.raises(QuotaExhausted, match="HTTP 429: daily limit") as caught:
        await ask_model(client, ITEMS.items[0], RunOptions("groq/m"), Pacer(60_000), no_sleep)
    assert caught.value.retry_after_seconds == 3600
    assert len(seen) == 1  # not worth retrying


async def test_a_rate_limit_that_never_lifts_is_a_quota() -> None:
    limited = httpx.Response(429, json={"error": {"message": "slow down"}})  # no retry-after
    client, seen = gateway_client(*[limited] * 6)
    with pytest.raises(QuotaExhausted) as caught:
        await ask_model(client, ITEMS.items[0], RunOptions("groq/m"), Pacer(60_000), no_sleep)
    assert caught.value.retry_after_seconds is None
    assert len(seen) == 6  # the first try and 5 retries


@pytest.mark.parametrize("status", [400, 401, 500, 502])
async def test_other_failures_are_not_retried(status: int) -> None:
    client, seen = gateway_client(httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(ModelCallError, match=f"HTTP {status}: nope"):
        await ask_model(client, ITEMS.items[0], RunOptions("groq/m"), Pacer(60), no_sleep)
    assert len(seen) == 1


async def test_pacer_spaces_calls() -> None:
    now = [0.0]
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    pacer = Pacer(60, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        await pacer.wait()
    assert waits == [1.0, 2.0]  # 1 call per second
    now[0] = 10.0
    await pacer.wait()
    assert waits == [1.0, 2.0]  # idle long enough: no wait


# --- Summary ---


def record(task: str, passed: bool | None, **fields: Any) -> dict[str, Any]:
    base = {"task": task, "passed": passed, "score": None if passed is None else float(passed)}
    return base | {"error": None, "cost_usd": 0.001, "latency_ms": 100.0} | fields


def test_summary_counts_errors_as_fails_and_leaves_out_ungraded() -> None:
    records = [
        record("maths", True),
        record("maths", False, error="HTTP 502", cost_usd=None, latency_ms=None),
        record("maths", None),  # the judge could not grade it
        record("chat", True, score=0.8, cached=True),
        record("chat", False, finish_reason="length"),
    ]
    summary = summarize(records)
    maths = summary["maths"]
    assert (maths["items"], maths["graded"], maths["passed"], maths["errors"]) == (3, 2, 1, 1)
    assert (maths["pass_rate"], maths["ungraded"], maths["unpriced"]) == (0.5, 1, 0)
    assert summary["chat"]["mean_score"] == pytest.approx(0.4)  # (0.8 + 0) / 2
    total = summary["total"]
    assert total["cost_usd"] == pytest.approx(0.004)
    assert (total["cached"], total["cut_off"], total["p50_ms"]) == (1, 1, 100.0)

    text = format_summary(summary)
    assert text.splitlines()[0].split() == [
        "task", "items", "passed", "rate", "score", "95%", "CI", "errors", "ungraded", "p50", "ms",
    ]  # fmt: skip
    # 1 of 2 graded: a Wilson interval of [0.09, 0.91]; 2 items say very little.
    assert (
        "maths          3       1   50.0%   0.50  [0.09, 0.91]        1         1     100" in text
    )
    assert maths["score_ci"] == [pytest.approx(0.0945, abs=1e-4), pytest.approx(0.9055, abs=1e-4)]
    assert "cost $0.004000; 1 replies from the cache" in text
    assert text.endswith("1 replies were cut off at max tokens (--max-tokens)")


# --- Command line ---


def test_command_checks_its_arguments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["groq/m", "--tasks", "poetry"])
    assert "unknown task(s): poetry" in capsys.readouterr().err

    data = tmp_path / "set.jsonl"
    data.write_text(ITEMS.items[0].model_dump_json() + "\n", encoding="utf-8")
    assert main(["groq/m", "--data", str(data), "--split", "test"]) == 1
    assert "no items match" in capsys.readouterr().err

    # No API keys in the test environment, so no provider is enabled.
    assert main(["groq/m", "--data", str(data), "--no-judge", "--out", str(tmp_path / "r")]) == 1
    assert "'groq/m' is not enabled" in capsys.readouterr().err


def test_cache_survives_a_restart(tmp_path: Path) -> None:
    cache = JsonlCache(tmp_path / "c.jsonl")
    cache.put("k", {"reply": "84"})
    assert JsonlCache(tmp_path / "c.jsonl").get("k") == {"reply": "84"}
