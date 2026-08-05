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
"""GPU-free end-to-end check of the DAS data path on a Ray cluster.

Validates, with a real (detached, named) SuffixDataService actor:

  worker pushes -> central store -> versioned deltas -> engine drafter ->
  correct draft for a repeated pattern -> sliding-window eviction ->
  fresh-replica snapshot resync -> length-class budgets.

Run on the Ray head (or a laptop; it starts a local Ray if none exists):

    python3 -m integration.verl.das_compat_check

Requires no GPUs, no vLLM, and no verl.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import ray

from py_inference_scheduler.speculative.config import (
    CLS_LONG,
    DASClassifierConfig,
    DASConfig,
    DASPushConfig,
)
from py_inference_scheduler.speculative.contracts import TrajectoryPush
from py_inference_scheduler.speculative.drafter_state import DASDrafterState, PySuffixTree
from py_inference_scheduler.speculative.problem_id import (
    encode_request_id,
    hash_problem_id,
    parse_request_id,
)
from py_inference_scheduler.speculative.push_client import DASPushClient
from py_inference_scheduler.speculative.service import get_or_create_suffix_service


def _check(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise AssertionError(f"das_compat_check FAILED: {message}")
    print(f"  ok: {message}")


async def _drain(service, state: DASDrafterState, replica: str) -> None:
    version = state.state_version()
    while True:
        payload = await service.get_deltas.remote(replica, version)
        if payload is None:
            return
        state.apply_delta_batch(payload)
        version = state.state_version()


async def _wait_for_pushes(service, expected: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        metrics = await service.get_metrics.remote()
        if metrics["das_pushes_received"] >= expected:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"service received fewer than {expected} pushes within {timeout_s}s")


async def main() -> None:  # noqa: PLR0914,PLR0915 - linear check script
    cfg = DASConfig(
        window_iterations=2,
        fresh_iterations=1,
        service_name=f"pyis_das_check_{uuid.uuid4().hex[:8]}",
        classifier=DASClassifierConfig(short_max_tokens=4, long_min_tokens=12),
        push=DASPushConfig(batch_size=4, flush_interval_s=0.05),
    )
    service = get_or_create_suffix_service(cfg)
    try:
        print("1) worker pushes reach the central store (batched, fire-and-forget)")
        prompt_a = list(range(100, 120))
        prompt_b = list(range(200, 220))
        phash_a = hash_problem_id(prompt_a)
        phash_b = hash_problem_id(prompt_b)
        _check(phash_a != phash_b, "distinct problems hash distinctly")
        _check(
            parse_request_id(encode_request_id(phash_a)) == phash_a,
            "engine request id round-trips the problem hash",
        )

        await service.begin_iteration.remote(1)
        # Two hook workers, four GRPO siblings per problem, shared motif per
        # problem so cross-sibling drafting has something to find.
        motif_a = [7, 8, 9, 10, 11, 12]
        motif_b = [70, 80, 90, 100]
        clients = [DASPushClient(service, f"hook-{i}", cfg.push) for i in range(2)]
        for i, client in enumerate(clients):
            for j in range(2):
                client.enqueue(
                    TrajectoryPush(phash_a, (*motif_a, 1000 + i, j), 20, f"s{i}")
                )
                client.enqueue(
                    TrajectoryPush(phash_b, (*motif_b, 2000 + i, j), 20, f"s{i}")
                )
        await _wait_for_pushes(service, expected=8)
        print("2) engine replica drains deltas and drafts the shared motif")
        engine1 = DASDrafterState(cfg, tree_factory=PySuffixTree)
        await _drain(service, engine1, "engine-1")
        draft = engine1.speculate(phash_a, motif_a[:3], budget=8, max_spec_factor=2.0)
        _check(draft is not None and draft.token_ids[: 3] == motif_a[3:6],
               "per-problem tree drafts the cross-worker motif")
        _check(
            engine1.speculate(phash_b, motif_a[:3], budget=8, max_spec_factor=2.0) is None,
            "problem trees are isolated from each other",
        )

        print("3) sliding window evicts old iterations")
        await service.begin_iteration.remote(5)
        await _drain(service, engine1, "engine-1")
        _check(
            engine1.speculate(phash_a, motif_a[:3], budget=8, max_spec_factor=2.0) is None,
            "window advance removes evicted sequences from engine trees",
        )

        print("4) fresh replica resyncs (snapshot or full replay) to same state")
        clients[0].enqueue(TrajectoryPush(phash_a, tuple(motif_a * 3), 20, "s0"))
        clients[0].flush()
        await _wait_for_pushes(service, expected=9)
        engine2 = DASDrafterState(cfg, tree_factory=PySuffixTree)
        await _drain(service, engine1, "engine-1")
        await _drain(service, engine2, "engine-2")
        d1 = engine1.speculate(phash_a, motif_a[:3], budget=8, max_spec_factor=2.0)
        d2 = engine2.speculate(phash_a, motif_a[:3], budget=8, max_spec_factor=2.0)
        _check(
            d1 is not None and d2 is not None and d1.token_ids == d2.token_ids,
            "late-joining replica converges to the same drafts",
        )

        print("5) length classes drive budgets")
        # Keep pushing 18-token trajectories until the EMA crosses
        # long_min(12) and problem A reclassifies to Long.
        for _ in range(3):
            clients[0].enqueue(TrajectoryPush(phash_a, tuple(motif_a * 3), 20, "s0"))
        clients[0].flush()
        await _wait_for_pushes(service, expected=12)
        stats = await service.get_length_stats.remote()
        _check(stats[phash_a].cls == CLS_LONG, "long trajectories reclassify the problem")
        await _drain(service, engine1, "engine-1")
        _check(
            engine1.budget_for(phash_a, 0) == cfg.budgets.long,
            "engine budget follows the centrally-computed class",
        )

        metrics = await service.get_metrics.remote()
        print(f"service metrics: {metrics}")
        print("PASS: DAS data path verified end to end")
    finally:
        ray.kill(service)


if __name__ == "__main__":
    ray.init(namespace="pyis", ignore_reinit_error=True)
    asyncio.run(main())
