# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert zwhe99/DeepMath-103K to verl parquet, difficulty-banded.

DeepMath-103K (arXiv 2504.11456) skews much harder than DeepScaleR
(difficulty 3-10, mass at 5-9), eliciting longer chains of thought — the
long-tail regime DAS targets. The difficulty band keeps GRPO's reward
signal alive: an all-difficulty-9 batch at near-zero pass rate produces
all-fail groups with zero advantage.

data_source is set to "lighteval/MATH" so verl's rule-based boxed-answer
scorer applies. Difficulty is preserved in extra_info (future use: seed
DAS Long/Medium/Short class priors from it).

    python3 integration/verl/examples/prepare_deepmath.py \
        --local-dir /home/ray/data/deepmath \
        [--min-difficulty 4] [--max-difficulty 7] [--test-size 500]
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset  # type: ignore[import-not-found]

INSTRUCTION = " Let's think step by step and output the final answer within \\boxed{}."


def _make_map_fn(split: str):
    def process(example: dict, idx: int) -> dict:
        answer = example.get("final_answer") or example.get("answer") or ""
        return {
            "data_source": "lighteval/MATH",
            "prompt": [{"role": "user", "content": example["question"] + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "extra_info": {
                "split": split,
                "index": idx,
                "difficulty": float(example.get("difficulty", -1)),
            },
        }

    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="/home/ray/data/deepmath")
    parser.add_argument("--min-difficulty", type=float, default=4.0)
    parser.add_argument("--max-difficulty", type=float, default=7.0)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
    total = len(dataset)
    dataset = dataset.filter(
        lambda x: args.min_difficulty <= float(x.get("difficulty", -1)) <= args.max_difficulty
    )
    print(f"difficulty band [{args.min_difficulty}, {args.max_difficulty}]: "
          f"{len(dataset)}/{total} problems kept")

    splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    os.makedirs(args.local_dir, exist_ok=True)  # noqa: PTH103
    for split, data in (("train", splits["train"]), ("test", splits["test"])):
        mapped = data.map(
            _make_map_fn(split), with_indices=True, remove_columns=data.column_names
        )
        out = os.path.join(args.local_dir, f"{split}.parquet")  # noqa: PTH118
        mapped.to_parquet(out)
        print(f"wrote {len(mapped)} rows to {out}")


if __name__ == "__main__":
    main()
