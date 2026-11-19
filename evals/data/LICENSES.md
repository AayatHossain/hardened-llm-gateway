# Eval data licences

The code in this repository is MIT licensed. The eval files in this folder are
subsets of public datasets and keep their own licences.

| File | Source | Licence | Items |
|---|---|---|---|
| `maths.jsonl` | [GSM8K](https://github.com/openai/grade-school-math), test set. Cobbe et al., 2021 | [MIT](https://github.com/openai/grade-school-math/blob/master/LICENSE) | 230 |
| `code.jsonl` | [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), sanitized test split. Austin et al., 2021 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 230 |
| `facts.jsonl` | [SimpleQA](https://github.com/openai/simple-evals). Wei et al., 2024 | [MIT](https://github.com/openai/simple-evals/blob/main/LICENSE) | 230 |
| `reasoning.jsonl` | [ARC-Challenge](https://allenai.org/data/arc), test set. Clark et al., 2018 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 230 |
| `chat.jsonl` | [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) questions. Zheng et al., 2023 | [Apache 2.0](https://github.com/lm-sys/FastChat/blob/main/LICENSE) | 80 |
| `sample.jsonl` | Hand-written for this project | MIT | 10 |

## Changes from the originals

All 5 sets were changed in the same ways:

- A subset was chosen, and each item was given a dev or test split, by a hash of its id.
- Each item was put in this project's format: a chat message, an answer, a grader.
- An instruction about the reply's form was added to the prompt, e.g. "Answer with the
  letter of the correct option only."

Changes specific to 1 set:

- **GSM8K:** only the final answer is kept; the worked solution is dropped.
- **MBPP:** the prompt shows the first of the item's tests; all tests are kept for grading.
- **ARC:** options are always lettered A, B, C, ...; a few originals use 1, 2, 3.
- **MT-Bench:** only the first turn is kept, with its reference answer where one exists.

`reasoning.jsonl` is an adaptation of ARC, so it is shared under CC BY-SA 4.0.

`manifest.json` records where each file came from: the pinned commit or URL, the
SHA-256 of the downloaded file, and the fingerprint of the result.
`python -m hardened_llm_gateway.evals.build` rebuilds the files exactly.
