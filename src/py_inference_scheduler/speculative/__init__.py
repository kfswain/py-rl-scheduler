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
"""DAS-style speculative decoding support (arXiv 2511.13841).

Centralized suffix-tree trajectory store + engine-local draft trees:

- :mod:`.service` — ``SuffixDataService`` named Ray actor gathering rollout
  trajectories from every agent-loop worker.
- :mod:`.push_client` — fire-and-forget batched trajectory pushes from the
  verl hook.
- :mod:`.drafter_state` — per-problem suffix trees + length-aware budgets,
  living inside each vLLM GPU worker.
- :mod:`.vllm_plugin` / :mod:`.vllm_proposer` — ``vllm.general_plugins``
  entry point that patches vLLM's stock suffix-decoding proposer in place.

Everything here is inert unless ``PYIS_DAS_ENABLED=1``.
"""

from __future__ import annotations

from py_inference_scheduler.speculative.config import (
    DASConfig,
    das_enabled,
    load_das_config,
)
from py_inference_scheduler.speculative.contracts import (
    DeltaBatch,
    LengthStats,
    TrajectoryDelta,
    TrajectoryPush,
)
from py_inference_scheduler.speculative.problem_id import (
    encode_request_id,
    hash_problem_id,
    parse_request_id,
)

__all__ = [
    "DASConfig",
    "DeltaBatch",
    "LengthStats",
    "TrajectoryDelta",
    "TrajectoryPush",
    "das_enabled",
    "encode_request_id",
    "hash_problem_id",
    "load_das_config",
    "parse_request_id",
]


def __getattr__(name: str) -> object:
    # Ray-dependent and engine-side pieces are imported lazily so that pure
    # consumers (tests, config tooling) never pull in ray or vllm.
    if name in {"SuffixDataService", "get_or_create_suffix_service"}:
        from py_inference_scheduler.speculative import service

        return getattr(service, name)
    if name == "DASPushClient":
        from py_inference_scheduler.speculative.push_client import DASPushClient

        return DASPushClient
    if name == "DASDrafterState":
        from py_inference_scheduler.speculative.drafter_state import DASDrafterState

        return DASDrafterState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
