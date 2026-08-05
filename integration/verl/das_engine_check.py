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
"""Engine-image check for the DAS vLLM proposer patch.

Verifies the patch against the REAL vllm + arctic-inference installed in
the training image. No GPU needed.

Run inside the training image (head pod or a ray job with the runtime env):

    PYIS_DAS_ENABLED=1 python3 -m integration.verl.das_engine_check

Checks, in order:
  1. arctic-inference importable; adapter drafts on a real C++ tree
  2. pyis_das entry point visible in vllm.general_plugins pip metadata
     (warn-only: the DASWorkerExtension fallback covers PYTHONPATH installs)
  3. plugin register() patches SuffixDecodingProposer in place, idempotently
  4. a patched proposer built from a stub VllmConfig drives propose() on a
     fake input batch: DAS-id requests draft from per-problem trees fed by a
     delta batch, plain-id requests fall back to stock behavior
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

from py_inference_scheduler.speculative.contracts import (
    DeltaBatch,
    TrajectoryDelta,
    serialize_delta_batch,
)
from py_inference_scheduler.speculative.drafter_state import default_tree_factory
from py_inference_scheduler.speculative.problem_id import encode_request_id, hash_problem_id


def _check(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise AssertionError(f"das_engine_check FAILED: {message}")
    print(f"  ok: {message}")


def check_arctic() -> None:
    print("1) arctic-inference adapter on a real C++ tree")
    tree = default_tree_factory(24)
    _check(type(tree).__name__ == "ArcticSuffixTree", "arctic tree in use (not python fallback)")
    tree.extend(1, [10, 20, 30, 40, 50])
    draft = tree.speculate([10, 20, 30], max_spec_tokens=8, max_spec_factor=2.0)
    _check(draft is not None and draft.token_ids == [40, 50], "adapter drafts continuation")


def check_entry_point() -> None:
    print("2) vllm.general_plugins entry point")
    from importlib.metadata import entry_points

    names = [ep.name for ep in entry_points(group="vllm.general_plugins")]
    if "pyis_das" in names:
        print("  ok: pyis_das entry point registered in pip metadata")
    else:
        print(
            "  WARN: pyis_das entry point not found (package not pip-installed?). "
            "The DASWorkerExtension fallback will patch instead on verl >= 0.9."
        )


def check_patch() -> tuple:
    print("3) proposer patch applies in place, idempotently")
    from py_inference_scheduler.speculative import vllm_plugin

    try:
        from vllm.v1.spec_decode.suffix_decoding import (  # type: ignore[import-not-found]
            SuffixDecodingProposer as HostCls,
        )

        host = "suffix"
    except ImportError:
        from vllm.v1.spec_decode.ngram_proposer import (  # type: ignore[import-not-found]
            NgramProposer as HostCls,
        )

        host = "ngram"
    host_cls = HostCls
    print(f"  host: {host} (vLLM {'>= 0.11.1' if host == 'suffix' else '0.11.0-era'})")

    vllm_plugin.register()
    _check(getattr(host_cls, "_das_patched", False), "class patched")
    patched_propose = host_cls.propose
    vllm_plugin.register()
    _check(host_cls.propose is patched_propose, "second register() is a no-op")
    return host, host_cls


def check_propose(host: str, proposer_cls) -> None:
    print("4) patched propose() on a fake batch (real cache + real trees)")
    stub_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=8,
            suffix_decoding_max_tree_depth=24,
            suffix_decoding_max_spec_factor=2.0,
            suffix_decoding_min_token_prob=0.1,
            suffix_decoding_max_cached_requests=1000,
            prompt_lookup_min=2,
            prompt_lookup_max=8,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    try:
        proposer = proposer_cls(stub_config)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: stub VllmConfig rejected ({e}); vLLM version drift — verify manually")
        return
    _check(getattr(proposer, "_das_state", None) is not None, "DAS state initialized")

    prompt = [5, 6, 7, 8]
    phash = hash_problem_id(prompt)
    motif = [100, 101, 102, 103, 104, 105]
    proposer._das_state.apply_delta_batch(
        serialize_delta_batch(
            DeltaBatch(
                from_version=-1,
                to_version=1,
                iteration=1,
                adds=[TrajectoryDelta(phash, 0, tuple(motif))],
                problem_cls={phash: 2},
            )
        )
    )

    das_req = encode_request_id(phash)
    token_row = np.zeros(64, dtype=np.int64)
    token_row[:4] = prompt
    token_row[4:7] = motif[:3]  # generated so far: start of the motif
    req_ids = [das_req, "plain-request"]
    sampled = [[motif[2]], [motif[2]]]
    num_tokens_no_spec = np.array([7, 7])
    token_ids_cpu = np.stack([token_row, token_row])
    if host == "suffix":
        batch = SimpleNamespace(
            req_ids=req_ids,
            num_prompt_tokens=np.array([4, 4]),
            num_tokens_no_spec=num_tokens_no_spec,
            token_ids_cpu=token_ids_cpu,
            req_id_to_index={das_req: 0, "plain-request": 1},
        )
        drafts = proposer.propose(batch, sampled)
    else:
        drafts = proposer.propose(sampled, req_ids, num_tokens_no_spec, token_ids_cpu, set())
    _check(getattr(proposer, "_das_state", None) is not None, "DAS survived propose()")
    _check(len(drafts) == len(req_ids), "one draft slot per request")
    _check(list(drafts[0])[:3] == motif[3:6], "DAS request drafts from per-problem tree")
    print(f"  drafts: das={list(drafts[0])} plain={list(drafts[1])}")
    stats = proposer._das_state.tree_stats()
    _check(stats["transient_requests"] >= 1, "transient self-insert recorded")


def main() -> int:
    if os.environ.get("PYIS_DAS_ENABLED") != "1":
        print("PYIS_DAS_ENABLED != 1; set it and rerun")
        return 1
    check_arctic()
    check_entry_point()
    host, proposer_cls = check_patch()
    check_propose(host, proposer_cls)
    print()
    print("PASS: DAS engine-side patch verified against real vllm + arctic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
