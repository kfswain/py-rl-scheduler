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
"""``vllm.general_plugins`` entry point.

vLLM loads general plugins in every process it owns — process0, the
EngineCore process, and each GPU worker (before the drafter is constructed)
— re-executing entry points from installed package metadata, so this works
under both fork and spawn. Declared in pyproject:

    [project.entry-points."vllm.general_plugins"]
    pyis_das = "py_inference_scheduler.speculative.vllm_plugin:register"

Inert unless ``PYIS_DAS_ENABLED=1`` (env vars inherit into spawned engine
processes). vLLM may invoke plugins more than once; the patch is idempotent.
"""

from __future__ import annotations

import logging

from py_inference_scheduler.speculative.config import das_enabled

logger = logging.getLogger(__name__)


def register() -> None:
    if not das_enabled():
        return
    try:
        from py_inference_scheduler.speculative.vllm_proposer import das_patch_proposer

        das_patch_proposer()
    except Exception as e:  # noqa: BLE001
        # Never break engine startup: DAS silently degrades to stock vLLM.
        logger.warning("DAS: plugin registration failed, running without DAS: %s", e)
