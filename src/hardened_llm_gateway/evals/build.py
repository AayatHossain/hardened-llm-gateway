"""Build the eval set (1,000 items) from 5 public datasets.

    uv run python -m hardened_llm_gateway.evals.build

| Task      | Source                      | Licence      | Items | Grader          |
|-----------|-----------------------------|--------------|-------|-----------------|
| maths     | GSM8K, test set             | MIT          | 230   | numeric         |
| code      | MBPP sanitized, test split  | CC BY 4.0    | 230   | code            |
| facts     | SimpleQA                    | MIT          | 230   | numeric / judge |
| reasoning | ARC-Challenge, test set     | CC BY-SA 4.0 | 230   | exact (letter)  |
| chat      | MT-Bench, first turns       | Apache 2.0   | 80    | judge           |

Every download is pinned to a commit and checked against a SHA-256, so a rebuild
gives the same files or fails. Items are picked, and given a split (30% dev,
70% test), by a hash of their id, so the choice never depends on file order.

The output is 1 JSONL file per task plus `manifest.json`, which records each
file's fingerprint and where it came from. Downloads are cached in
`.cache/eval-sources` (git-ignored).
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from hardened_llm_gateway.evals.dataset import EvalItem, EvalSet, Task, parse_number
from hardened_llm_gateway.providers import ChatMessage

DEV_SHARE = 0.3
SEED = "eval-set-v1"


def _github(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def get_file(url: str, client: httpx.Client) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


ARC_ROWS = "https://datasets-server.huggingface.co/rows"
ARC_WAITS = (5.0, 15.0, 30.0, 60.0)


def get_arc_rows(client: httpx.Client) -> bytes:
    """ARC is only published as parquet or a 680 MB zip, so read the rows as JSON.

    The rows are written out in a fixed form, and that is what the SHA-256 checks.
    """
    rows: list[dict[str, Any]] = []
    while True:
        params: dict[str, str | int] = {
            "dataset": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "split": "test",
            "offset": len(rows),
            "length": 100,
        }
        for wait in (*ARC_WAITS, None):
            response = client.get(ARC_ROWS, params=params)
            if response.status_code != 429 or wait is None:
                break
            time.sleep(wait)  # the rows API limits quick bursts
        response.raise_for_status()
        page = response.json()
        rows += [entry["row"] for entry in page["rows"]]
        if not page["rows"] or len(rows) >= page["num_rows_total"]:
            break
    return json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()


# --- Converters: source bytes -> every usable item (split is set later) ---


def gsm8k_items(data: bytes, credit: str) -> list[EvalItem]:
    items = []
    for index, line in enumerate(data.decode("utf-8").splitlines()):
        row = json.loads(line)
        answer = row["answer"].rsplit("####", 1)[1].strip()
        question = row["question"].strip()
        items.append(
            EvalItem(
                id=f"gsm8k-{index:04d}",
                task="maths",
                split="test",
                grader="numeric",
                messages=(
                    _user(
                        f"{question}\n\nSolve the problem. "
                        "End your reply with the final answer as a number only."
                    ),
                ),
                answer=answer,
                source=credit,
                meta={"original_index": index},
            )
        )
    return items


def mbpp_items(data: bytes, credit: str) -> list[EvalItem]:
    items = []
    for row in json.loads(data):
        if not 11 <= row["task_id"] <= 510:  # the test split, as in the MBPP paper
            continue
        prompt = (
            f"{row['prompt'].strip()}\n\nYour code should pass this test:\n\n"
            f"{row['test_list'][0]}\n\nReply with the Python code only."
        )
        items.append(
            EvalItem(
                id=f"mbpp-{row['task_id']}",
                task="code",
                split="test",
                grader="code",
                messages=(_user(prompt),),
                tests="\n".join([*row["test_imports"], *row["test_list"]]),
                source=credit,
                meta={"original_id": row["task_id"]},
            )
        )
    return items


def simpleqa_items(data: bytes, credit: str) -> list[EvalItem]:
    """Plain-number answers (mostly years) are graded as numbers, the rest by a judge."""
    items = []
    rows = csv.DictReader(io.StringIO(data.decode("utf-8")))
    for index, row in enumerate(rows):
        meta = ast.literal_eval(row["metadata"])
        question, answer = row["problem"].strip(), row["answer"].strip()
        numeric = parse_number(answer) is not None
        items.append(
            EvalItem(
                id=f"simpleqa-{index:04d}",
                task="facts",
                split="test",
                grader="numeric" if numeric else "judge",
                messages=(
                    _user(f"{question}\n\nAnswer with the number only." if numeric else question),
                ),
                answer=answer,
                source=credit,
                meta={
                    "original_index": index,
                    "topic": meta["topic"],
                    "answer_type": meta["answer_type"],
                },
            )
        )
    return items


def arc_items(data: bytes, credit: str) -> list[EvalItem]:
    """Options are always lettered A, B, C, ... (a few source items use 1, 2, 3)."""
    items = []
    for row in json.loads(data):
        labels = row["choices"]["label"]
        letters = [chr(ord("A") + i) for i in range(len(labels))]
        options = "\n".join(
            f"{letter}. {text}"
            for letter, text in zip(letters, row["choices"]["text"], strict=True)
        )
        items.append(
            EvalItem(
                id="arc-" + re.sub(r"[^a-z0-9._-]", "-", row["id"].lower()),
                task="reasoning",
                split="test",
                grader="exact",
                messages=(
                    _user(
                        f"{row['question'].strip()}\n\n{options}\n\n"
                        "Answer with the letter of the correct option only."
                    ),
                ),
                answer=letters[labels.index(row["answerKey"])],
                source=credit,
                meta={"original_id": row["id"]},
            )
        )
    return items


def mtbench_items(data: bytes, credit: str) -> list[EvalItem]:
    """Only the first turn: the second needs the model's own first reply.

    39 questions come with a human-written reference answer, given to the judge.
    """
    items = []
    for line in data.decode("utf-8").splitlines():
        row = json.loads(line)
        reference = row.get("reference")
        items.append(
            EvalItem(
                id=f"mtbench-{row['question_id']}",
                task="chat",
                split="test",
                grader="judge",
                messages=(_user(row["turns"][0]),),
                answer=reference[0] if reference else None,
                source=credit,
                meta={"original_id": row["question_id"], "category": row["category"]},
            )
        )
    return items


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


# --- Sources ---


@dataclass(frozen=True)
class Source:
    task: Task
    size: int
    name: str
    origin: str
    licence: str
    sha256: str
    download: Callable[[httpx.Client], bytes]
    convert: Callable[[bytes, str], list[EvalItem]]

    @property
    def credit(self) -> str:
        return f"{self.name} ({self.licence})"


GSM8K_URL = _github(
    "openai/grade-school-math",
    "3101c7d5072418e28b9008a6636bde82a006892c",
    "grade_school_math/data/test.jsonl",
)
MBPP_URL = _github(
    "google-research/google-research",
    "e2c3f9b85ff9804147541060bf35d68bf14f2258",
    "mbpp/sanitized-mbpp.json",
)
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
MTBENCH_URL = _github(
    "lm-sys/FastChat",
    "587d5cfa1609a43d192cedb8441cac3c17db105d",
    "fastchat/llm_judge/data/mt_bench/question.jsonl",
)

SOURCES: tuple[Source, ...] = (
    Source(
        task="maths",
        size=230,
        name="GSM8K test set",
        origin=GSM8K_URL,
        licence="MIT",
        sha256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        download=partial(get_file, GSM8K_URL),
        convert=gsm8k_items,
    ),
    Source(
        task="code",
        size=230,
        name="MBPP sanitized, test split",
        origin=MBPP_URL,
        licence="CC BY 4.0",
        sha256="ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
        download=partial(get_file, MBPP_URL),
        convert=mbpp_items,
    ),
    Source(
        task="facts",
        size=230,
        name="SimpleQA",
        origin=SIMPLEQA_URL,
        licence="MIT",
        sha256="feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032",
        download=partial(get_file, SIMPLEQA_URL),
        convert=simpleqa_items,
    ),
    Source(
        task="reasoning",
        size=230,
        name="ARC-Challenge test set",
        origin=f"{ARC_ROWS}?dataset=allenai/ai2_arc&config=ARC-Challenge&split=test",
        licence="CC BY-SA 4.0",
        sha256="91aa41fe25cea5d4366465e0b119959297855241d42e3afa6a902e64c9f07fd3",
        download=get_arc_rows,
        convert=arc_items,
    ),
    Source(
        task="chat",
        size=80,
        name="MT-Bench questions, first turn",
        origin=MTBENCH_URL,
        licence="Apache 2.0",
        sha256="119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
        download=partial(get_file, MTBENCH_URL),
        convert=mtbench_items,
    ),
)


class SourceChangedError(RuntimeError):
    """A download did not match its pinned SHA-256."""


def fetch(source: Source, cache: Path, client: httpx.Client) -> bytes:
    """The source's bytes, from the cache when it holds the right version."""
    path = cache / f"{source.task}.source"
    if path.exists() and _sha256(data := path.read_bytes()) == source.sha256:
        return data
    data = source.download(client)
    if (got := _sha256(data)) != source.sha256:
        raise SourceChangedError(
            f"{source.name}: expected SHA-256 {source.sha256}, got {got}. "
            "The source has changed; check it before updating the pin."
        )
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def choose(items: list[EvalItem], size: int, seed: str = SEED) -> list[EvalItem]:
    """`size` items picked by a hash of their id; the first 30% of a second hash go to dev.

    The result keeps the source's order, so the files read like the original.
    """
    if len(items) < size:
        raise ValueError(f"only {len(items)} items, {size} wanted")
    picked = sorted(items, key=lambda i: _sha256(f"{seed}:pick:{i.id}".encode()))[:size]
    by_split_hash = sorted(picked, key=lambda i: _sha256(f"{seed}:split:{i.id}".encode()))
    dev = {i.id for i in by_split_hash[: round(size * DEV_SHARE)]}
    keep = {i.id for i in picked}
    return [
        item.model_copy(update={"split": "dev" if item.id in dev else "test"})
        for item in items
        if item.id in keep
    ]


def build(cache: Path, out: Path, client: httpx.Client) -> dict[str, Any]:
    """Write 1 JSONL file per task and `manifest.json`; return the manifest."""
    files: dict[str, Any] = {}
    every: list[EvalItem] = []
    for source in SOURCES:
        items = choose(source.convert(fetch(source, cache, client), source.credit), source.size)
        name = f"{source.task}.jsonl"
        (out / name).write_text(
            "".join(i.model_dump_json(exclude_defaults=True) + "\n" for i in items),
            encoding="utf-8",
            newline="\n",
        )
        files[name] = {
            "task": source.task,
            "items": len(items),
            "dev": sum(i.split == "dev" for i in items),
            "test": sum(i.split == "test" for i in items),
            "fingerprint": EvalSet(tuple(items)).fingerprint,
            "source": source.name,
            "licence": source.licence,
            "origin": source.origin,
            "source_sha256": source.sha256,
        }
        every += items
    manifest = {
        "seed": SEED,
        "items": len(every),
        "fingerprint": EvalSet(tuple(every)).fingerprint,
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--cache", type=Path, default=Path(".cache/eval-sources"))
    parser.add_argument("--out", type=Path, default=Path("evals/data"))
    args = parser.parse_args(argv)
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            manifest = build(args.cache, args.out, client)
    except (SourceChangedError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, entry in manifest["files"].items():
        print(f"{name:<16}{entry['items']:>5} items  fingerprint {entry['fingerprint']}")
    print(f"{'total':<16}{manifest['items']:>5} items  fingerprint {manifest['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
