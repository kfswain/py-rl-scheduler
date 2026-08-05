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
"""Wire contracts between hook workers, the SuffixDataService, and engines.

Delta payloads cross two hops: Ray actor RPC (service -> vLLMHttpServer) and
vLLM ``collective_rpc`` (server -> every TP-rank GPU worker). Both hops are
cluster-internal and same-trust-domain; payloads are pickled with a one-byte
schema version so mixed-version rollouts fail loudly instead of subtly.
"""

from __future__ import annotations

import pickle  # noqa: S403 - payloads only cross cluster-internal trusted channels
from dataclasses import dataclass, field

SCHEMA_VERSION = 1

# seq_id namespaces. Canonical service sequences live in [0, SHADOW_OFFSET);
# their 2x-recency shadow copies at +SHADOW_OFFSET; engine-local transient
# sequences in [TRANSIENT_BASE, 2^31) so the namespaces can never collide.
SHADOW_OFFSET = 2**29
TRANSIENT_BASE = 2**30
SEQ_ID_LIMIT = 2**31


@dataclass(frozen=True)
class TrajectoryPush:
    """One engine call's response tokens, pushed hook -> service."""

    problem_id: str
    token_ids: tuple
    prompt_len: int
    server_id: str


@dataclass(frozen=True)
class TrajectoryDelta:
    """One sequence to insert into a problem tree, service -> engines."""

    problem_id: str
    seq_id: int
    token_ids: tuple


@dataclass(frozen=True)
class LengthStats:
    count: int
    mean: float
    p50: float
    ema_mean: float
    cls: int


@dataclass
class DeltaBatch:
    from_version: int
    to_version: int
    iteration: int
    snapshot: bool = False
    reset: bool = False
    adds: list = field(default_factory=list)  # list[TrajectoryDelta]
    removals: list = field(default_factory=list)  # list[tuple[problem_id, seq_id]]
    dropped_problems: list = field(default_factory=list)  # list[problem_id]
    problem_cls: dict = field(default_factory=dict)  # problem_id -> CLS_*


def serialize_delta_batch(batch: DeltaBatch) -> bytes:
    return bytes([SCHEMA_VERSION]) + pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_delta_batch(payload: bytes) -> DeltaBatch:
    if not payload:
        raise ValueError("empty DAS delta payload")
    version = payload[0]
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"DAS delta schema mismatch: payload v{version}, expected v{SCHEMA_VERSION}"
        )
    batch = pickle.loads(payload[1:])  # noqa: S301 - cluster-internal trusted channel
    if not isinstance(batch, DeltaBatch):
        raise TypeError(f"DAS delta payload decoded to {type(batch).__name__}")
    return batch
