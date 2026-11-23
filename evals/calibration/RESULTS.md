# Judge calibration

How often does the LLM judge agree with a person? This measures it on 100 hand-labelled
answers, so the judged scores in the baseline (facts and chat) come with a known error.

## Method

- **Judge:** `groq/openai/gpt-oss-120b`, prompt version 1 (`evals/judge.py`).
- **Items:** 100 judged answers from dev-split runs (never the test split): 60 facts
  (SimpleQA) and 40 chat (MT-Bench), from gpt-oss-120b (51), qwen3.8-27b (45) and
  gpt-oss-20b (4). They were sampled evenly over model and over the judge's pass and fail
  verdicts (`calibrate sample`, seed `calibration-v1`).
- **Labels:** 1 person, blind to the judge's verdict (`calibrate label`), using the same
  instructions as the judge: 3 grades for facts, a 1 to 10 rating for chat. They are in
  `labels.jsonl`.
- **Scoring:** `uv run python -m hardened_llm_gateway.evals.calibrate score`. Intervals are
  95% Wilson intervals; the rating difference has a paired bootstrap interval and a
  sign-flip test.

## Results

### Facts: the judge agrees on every item

| | Agreement | 95% interval |
|---|---|---|
| Same grade (of 3) | 60/60 | 94% to 100% |
| Same pass/fail | 60/60 | 94% to 100% |
| Judge says pass, person agrees | 4/4 | 51% to 100% |
| Judge says fail, person agrees | 56/56 | 94% to 100% |

Cohen's kappa is 1.00. The judge told INCORRECT (49) from NOT_ATTEMPTED (7) exactly as the
person did.

### Chat: too kind, mostly to unfinished replies

Without any fix, the judge and the person agreed on pass or fail for **33 of 40** replies
(82%, interval 68% to 91%). In all 7 disagreements the judge passed a reply the person
failed.

**5 of those 7 were replies cut off at the token limit.** qwen3.8-27b ran with a
1,000-token limit (its free tier allows 1,000 output tokens a minute), and 6 of its chat
replies stopped mid-text. The judge is not told a reply is unfinished, and it rated them 7
to 9, as if they were whole.

**The fix:** a judged reply cut off at the token limit now fails without asking the judge
(`GRADING_VERSION` 2 in `evals/run.py`). With it:

| | Agreement | 95% interval |
|---|---|---|
| Same pass/fail | 37/40 = 92% | 80% to 97% |
| Judge says pass, person agrees | 31/33 = 94% | 80% to 98% |
| Judge says fail, person agrees | 6/7 = 86% | 49% to 97% |

The 3 remaining disagreements:
- 2 long open-ended pieces (a travel blog post, a character description) that the judge
  passed and the person failed, the length bias LLM judges are known for;
- 1 cut-off reply the person still passed, which the rule now fails.

**Ratings:** the judge rates **+1.25 points higher** than the person on average (interval
+0.90 to +1.60, p = 0.0001), and the rank correlation is 0.52. Only pass/fail is used in the
scores, but chat pass rates should be read as somewhat generous. The prompt was not tuned on
these 40 items: that would fit the judge to this sample.

### By model (pass rates, with the cut-off rule)

| Model | Items | Person | Judge | Judge − person |
|---|---|---|---|---|
| gpt-oss-120b | 51 | 43% | 45% | +2% |
| qwen3.8-27b | 45 | 29% | 29% | +0% |
| gpt-oss-20b | 4 | 25% | 25% | +0% |

The judge is itself gpt-oss-120b, but it did not favour its own replies here. Without the
cut-off rule, its pass rate for qwen3.8-27b was 13 points above the person's.

## Limits

- **1 labeller.** There is no person-to-person agreement to compare with.
- **Few CORRECT facts.** Every model gets most SimpleQA questions wrong, so only 4 items
  were judged CORRECT. The chance that the judge passes a wrong answer is barely measured
  (interval 51% to 100%).
- **Few chat fails.** The judge failed only 1 chat reply before the rule, so agreement on
  bad replies rests on the cut-off items.
- **Dev split, 3 models.** Other models, or longer replies, may be judged differently.
