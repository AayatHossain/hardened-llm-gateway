"""Tests for judge calibration: sampling, labelling and scoring."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from hardened_llm_gateway.evals.calibrate import (
    SKIPPED,
    choose,
    cohen_kappa,
    judged_records,
    label,
    main,
    read_labels,
    score,
    write_labels,
)
from hardened_llm_gateway.evals.dataset import EvalItem


def eval_item(id: str, task: str, grader: str, split: str = "dev", **fields: Any) -> str:
    return EvalItem.model_validate(
        {
            "id": id,
            "task": task,
            "split": split,
            "grader": grader,
            "messages": [{"role": "user", "content": f"Question {id}?"}],
            "source": "tests",
            **fields,
        }
    ).model_dump_json()


def write_eval_set(path: Path) -> Path:
    lines = [eval_item(f"f{i}", "facts", "judge", answer=f"answer {i}") for i in range(8)]
    lines += [eval_item(f"c{i}", "chat", "judge") for i in range(4)]
    lines.append(eval_item("t0", "facts", "judge", split="test", answer="x"))
    lines.append(eval_item("m0", "maths", "numeric", answer="4"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def result(
    id: str, task: str, passed: bool | None, extracted: str, **fields: Any
) -> dict[str, Any]:
    return {
        "id": id,
        "task": task,
        "split": "dev",
        "grader": "judge",
        "passed": passed,
        "extracted": extracted,
        "error": None,
        "reply": f"reply to {id}",
    } | fields


def write_run(folder: Path, model: str, records: list[dict[str, Any]]) -> Path:
    folder.mkdir(parents=True)
    info = {"model": model, "judge": "groq/judge", "judge_prompt_version": 1}
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    text = "".join(json.dumps(r) + "\n" for r in records)
    (folder / "results.jsonl").write_text(text, encoding="utf-8")
    return folder


def model_run(folder: Path, model: str) -> Path:
    """Facts f0-f3 pass, f4-f7 fail; chat c0-c1 rated 8, c2-c3 rated 4."""
    records = [
        result(f"f{i}", "facts", i < 4, "CORRECT" if i < 4 else "INCORRECT") for i in range(8)
    ]
    records += [result(f"c{i}", "chat", i < 2, "8" if i < 2 else "4") for i in range(4)]
    records += [
        result("t0", "facts", True, "CORRECT", split="test"),  # test split: left out
        result("m0", "maths", True, "4", grader="numeric"),  # not judged: left out
        result("f9", "facts", None, "", detail="judge failed"),  # ungraded: left out
        result("f8", "facts", False, "", error="HTTP 502"),  # model error: left out
    ]
    return write_run(folder, model, records)


# --- Sampling ---


def test_only_judged_dev_items_are_used(tmp_path: Path) -> None:
    records = judged_records(model_run(tmp_path / "a", "groq/a"))
    assert sorted(r["id"] for r in records) == sorted(
        [f"f{i}" for i in range(8)] + ["c0", "c1", "c2", "c3"]
    )
    assert {(r["model"], r["judge_model"], r["judge_prompt_version"]) for r in records} == {
        ("groq/a", "groq/judge", 1)
    }


def test_choose_spreads_items_over_models_and_verdicts(tmp_path: Path) -> None:
    records = judged_records(model_run(tmp_path / "a", "groq/a"))
    records += judged_records(model_run(tmp_path / "b", "gemini/b"))
    chosen = choose(records, 10, facts_share=0.6, seed="s")
    cells = Counter((r["task"], r["model"], r["passed"]) for r in chosen)
    facts = {k: v for k, v in cells.items() if k[0] == "facts"}
    chat = {k: v for k, v in cells.items() if k[0] == "chat"}
    assert sum(facts.values()) == 6 and sum(chat.values()) == 4
    assert sorted(facts.values()) == [1, 1, 2, 2]  # 6 over 4 cells, as even as it goes
    assert sorted(chat.values()) == [1, 1, 1, 1]
    assert choose(records, 10, 0.6, "s") == chosen  # repeatable
    assert [r["task"] for r in chosen] != sorted(r["task"] for r in chosen)  # mixed order


def test_choose_fills_from_other_cells_when_one_runs_out(tmp_path: Path) -> None:
    records = judged_records(model_run(tmp_path / "a", "groq/a"))
    chosen = choose(records, 14, facts_share=0.6, seed="s")  # wants 8 facts, 6 chat
    tasks = Counter(r["task"] for r in chosen)
    assert tasks == {"facts": 8, "chat": 4}  # only 4 chat items exist


def test_sample_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = write_eval_set(tmp_path / "set.jsonl")
    a = model_run(tmp_path / "runs" / "a", "groq/a")
    labels = tmp_path / "labels.jsonl"
    args = ["--labels", str(labels), "sample", str(a), "--n", "5", "--data", str(data)]
    assert main(args) == 0
    assert f"wrote 5 items to {labels}" in capsys.readouterr().out
    rows = read_labels(labels)
    assert len(rows) == 5
    f0 = next((r for r in rows if r["id"].startswith("f")), None)
    assert f0 is not None
    number = f0["id"][1:]
    assert f0["question"] == f"Question {f0['id']}?"
    assert f0["reference"] == f"answer {number}"
    assert f0["reply"] == f"reply to {f0['id']}"
    assert f0["human"] is None
    chat = [r for r in rows if r["task"] == "chat"]
    assert chat and all(isinstance(r["judge"], int) for r in chat)

    assert main(args) == 1  # labels may already be there: never overwritten by accident
    assert "pass --force to replace it" in capsys.readouterr().err
    assert main([*args, "--force"]) == 0

    test_only = write_run(
        tmp_path / "runs" / "t", "groq/t", [result("t0", "facts", True, "CORRECT", split="test")]
    )
    assert (
        main(["--labels", str(tmp_path / "x.jsonl"), "sample", str(test_only), "--data", str(data)])
        == 1
    )
    assert "no judged dev items" in capsys.readouterr().err


# --- Labelling ---


def label_file(path: Path, *tasks: str) -> Path:
    rows = [
        {"id": f"x{i}", "task": task, "model": "groq/a", "question": "Q?", "reference": "R",
         "reply": "A", "judge": "CORRECT" if task == "facts" else 8,
         "judge_model": "groq/judge", "judge_prompt_version": 1, "human": None}
        for i, task in enumerate(tasks)
    ]  # fmt: skip
    write_labels(path, rows)
    return path


def scripted(*answers: str) -> Any:
    queue = list(answers)

    def ask(prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return ask


def test_labelling_saves_each_answer_and_resumes(tmp_path: Path) -> None:
    path = label_file(tmp_path / "labels.jsonl", "facts", "chat", "facts", "chat")
    shown: list[str] = []
    done = label(path, scripted("x", "c", "11", "7", "q"), shown.append)
    assert done == 2
    assert [r["human"] for r in read_labels(path)] == ["CORRECT", 7, None, None]
    text = "\n".join(shown)
    assert "type c, i or n" in text and "type 1 to 10" in text  # bad answers asked again
    assert "REFERENCE ANSWER\nR" in text

    done = label(path, scripted("s", " 5 "), shown.append)  # resumes at the 3rd item
    assert done == 4
    assert [r["human"] for r in read_labels(path)] == ["CORRECT", 7, SKIPPED, 5]
    assert "[3/4] facts" in "\n".join(shown)
    assert not (tmp_path / "labels.tmp").exists()


def test_the_judges_verdict_is_not_shown(tmp_path: Path) -> None:
    path = label_file(tmp_path / "labels.jsonl", "chat")
    shown: list[str] = []
    label(path, scripted("q"), shown.append)
    card = shown[1]
    assert "REPLY\nA" in card
    assert "groq/judge" not in card and " 8" not in card.replace("1 to 10", "")


# --- Scoring ---


def test_cohen_kappa() -> None:
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohen_kappa([]) is None
    # 20 items: agree on 15 (10 yes, 5 no); 5 disagree. Expected agreement by chance:
    # judge 13 yes / 7 no, person 12 yes / 8 no -> (13*12 + 7*8) / 400 = 0.53.
    pairs = [("y", "y")] * 10 + [("n", "n")] * 5 + [("y", "n")] * 3 + [("n", "y")] * 2
    assert cohen_kappa(pairs) == pytest.approx((0.75 - 0.53) / (1 - 0.53))
    assert cohen_kappa([("y", "y")] * 3) == 1.0  # 1 class only: no chance to disagree


def labelled(task: str, model: str, judge: str | int, human: str | int | None) -> dict[str, Any]:
    return {"id": "x", "task": task, "model": model, "judge": judge, "human": human,
            "judge_model": "groq/judge", "judge_prompt_version": 1}  # fmt: skip


def test_score_report() -> None:
    labels = [
        labelled("facts", "groq/a", "CORRECT", "CORRECT"),
        labelled("facts", "groq/a", "CORRECT", "INCORRECT"),  # the judge was too kind
        labelled("facts", "groq/a", "INCORRECT", "INCORRECT"),
        labelled("facts", "gemini/b", "NOT_ATTEMPTED", "INCORRECT"),  # both fails
        labelled("chat", "groq/a", 9, 8),
        labelled("chat", "gemini/b", 6, 8),  # the judge failed a pass
        labelled("chat", "gemini/b", 3, 2),
        labelled("chat", "gemini/b", 7, None),  # not labelled yet
        labelled("chat", "gemini/b", 7, SKIPPED),
    ]
    report = score(labels)
    assert report.startswith("7 labelled items (2 unlabelled or skipped)\n")
    assert "judge: groq/judge (prompt version 1), 7 items" in report
    assert "same grade:      2/4 = 50%" in report
    assert "same pass/fail:  3/4 = 75%" in report  # NOT_ATTEMPTED vs INCORRECT: both fail
    assert "judge says pass: person agrees 1/2 = 50%" in report
    assert "judge says fail: person agrees 2/2 = 100%" in report
    assert "  CORRECT                      1              1              0" in report
    # chat: 2 of 3 agree on pass/fail; ratings differ by 1, 2, 1
    assert "same pass/fail:  2/3 = 67%" in report
    assert "mean |judge - person| rating: 1.33" in report
    assert "judge - person rating: +0.00 [-2.00, +1.00], p = 1.0000" in report  # +1, -2, +1
    assert "rank correlation (Spearman): 0.87" in report
    # groq/a: person passes 2 of 4, judge 3 of 4; gemini/b: person 1 of 3, judge 0 of 3
    assert "groq/a                             4     50%     75%    +25%" in report
    assert "gemini/b                           3     33%      0%    -33%" in report


def test_score_and_label_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = label_file(tmp_path / "labels.jsonl", "facts")
    assert main(["--labels", str(path), "label"], ask=scripted("i")) == 0
    assert main(["--labels", str(path), "score"]) == 0
    out = capsys.readouterr().out
    assert "1 labelled items" in out and "judge says pass: person agrees 0/1" in out
    assert main(["--labels", str(tmp_path / "missing.jsonl"), "score"]) == 1
    assert "cannot read" in capsys.readouterr().err
