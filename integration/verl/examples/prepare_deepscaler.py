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
"""Convert agentica-org/DeepScaleR-Preview-Dataset to verl parquet.

DeepScaleR (~40k competition-math problems: AIME/AMC/Omni-MATH/Still) elicits
much longer chains of thought than GSM8K/MATH, which is the long-tail rollout
regime DAS targets (the DAS paper benchmarks on a DeepScaleR subset).

data_source is set to "lighteval/MATH" so verl's built-in rule-based math
scorer (boxed-answer extraction) applies without a custom reward function.

    python3 integration/verl/examples/prepare_deepscaler.py \
        --local-dir /home/ray/data/deepscaler [--test-size 500]
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset  # type: ignore[import-not-found]

INSTRUCTION = " Let's think step by step and output the final answer within \\boxed{}."


def _make_map_fn(split: str):
    def process(example: dict, idx: int) -> dict:
        return {
            "data_source": "lighteval/MATH",
            "prompt": [{"role": "user", "content": example["problem"] + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {"split": split, "index": idx},
        }

    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="/home/ray/data/deepscaler")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
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
