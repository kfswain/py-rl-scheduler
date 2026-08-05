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
"""verl worker extension exposing DAS RPCs inside every GPU worker.

Wired in by ``VllmEnginePatch`` overriding
``vLLMHttpServer._get_worker_extension_cls``; vLLM mixes this class into the
Worker in each TP-rank process, so its methods are reachable via
``engine.collective_rpc`` — the delivery path for delta batches.

``__new__`` re-applies the proposer patch as a belt-and-braces fallback for
deployments where the ``vllm.general_plugins`` entry point is unavailable
(package not pip-installed, only on PYTHONPATH): verl resolves this class by
qualname inside the worker process before the model runner (and drafter)
exists, which is exactly early enough.
"""

from __future__ import annotations

import logging

from py_inference_scheduler.speculative.config import das_enabled
from py_inference_scheduler.speculative.drafter_state import get_active_state

logger = logging.getLogger(__name__)

try:  # pragma: no cover - verl only exists on engine nodes
    from verl.workers.rollout.vllm_rollout.utils import (  # type: ignore[import-not-found]
        vLLMColocateWorkerExtension as _BaseExtension,
    )
except Exception:  # noqa: BLE001
    _BaseExtension = object  # type: ignore[assignment,misc]


class DASWorkerExtension(_BaseExtension):  # type: ignore[valid-type,misc]
    def __new__(cls, *args, **kwargs) -> DASWorkerExtension:  # noqa: ARG004,PYI034
        if das_enabled():
            try:
                from py_inference_scheduler.speculative.vllm_proposer import (
                    das_patch_proposer,
                )

                das_patch_proposer()
            except Exception as e:  # noqa: BLE001
                logger.warning("DAS: worker-extension patch fallback failed: %s", e)
        return super().__new__(cls)

    # ------------------------------------------------------------ DAS RPCs

    def das_apply_deltas(self, payload: bytes) -> int:
        """Apply a canonical delta batch; returns the new state version."""
        state = self._das_state()
        if state is None:
            return -1
        state.apply_delta_batch(payload)
        return state.state_version()

    def das_get_state_version(self) -> int:
        state = self._das_state()
        return state.state_version() if state is not None else -1

    def das_get_tree_stats(self) -> dict:
        state = self._das_state()
        return state.tree_stats() if state is not None else {}

    def _das_state(self):
        state = get_active_state()
        if state is not None:
            return state
        # Fallback: walk to the drafter (extension methods live on the Worker).
        drafter = getattr(getattr(self, "model_runner", None), "drafter", None)
        return getattr(drafter, "_das_state", None)
