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
"""Disaggregated (fully-async) verl integration: scheduler routing + DAS.

Targets verl's ``verl.experimental.fully_async_policy`` trainer, which splits
training and rollout onto separate resource pools (engines never sleep).
That trainer builds its own ``FullyAsyncAgentLoopManager`` directly, so the
``agent_loop_manager_class`` hook used by the synchronous trainer never fires
— this module patches the fully-async classes instead.

Load via Ray's ``worker_process_setup_hook`` so every worker process in the
job applies the patches before any verl actor is constructed::

    ray job submit --runtime-env-json '{
        "worker_process_setup_hook": "integration.verl.fully_async_das.setup",
        ...}'

Patches (idempotent; each degrades to a no-op where its imports are missing):

1. ``integration.verl.verl_hook`` import — applies ``VllmEnginePatch`` (DAS
   delta pump, worker extension, metrics) in the Rollouter actor process,
   which is where Ray captures the vLLM server actor class by value.
2. ``FullyAsyncAgentLoopManager`` -> subclass spawning scheduler-routing
   agent-loop workers (backpressure routing, DAS problem ids, trajectory
   pushes, logprob stripping — same behavior as the sync hook).
3. Client: ``FullyAsyncLLMServerClient``'s partial-rollout resume loop
   composed over ``InferenceSchedulerServerClient`` via MRO, so every
   (re)generation segment goes through the scheduler.
4. DAS iteration boundary: engines stamp each response with their weight
   version (``extra_fields["global_steps"]``); the first response observed
   at a new version fires ``SuffixDataService.begin_iteration(version + 1)``.
   With service-side per-iteration serving, delta applies therefore align
   with weight-sync boundaries — no Rollouter surgery, no reliance on
   sleep/wake (engines never sleep here).
5. GPU-type pinning: ``RayResourcePool`` gains an ``accelerator_type`` per
   pool from ``PYIS_TRAINER_ACCEL_TYPE`` / ``PYIS_ROLLOUT_ACCEL_TYPE``
   (e.g. ``accelerator_type:H100`` / ``accelerator_type:H200`` — Ray
   auto-publishes these node resources), so the trainer pool can never land
   on rollout GPUs or vice versa in a heterogeneous cluster.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

TRAINER_ACCEL_ENV = "PYIS_TRAINER_ACCEL_TYPE"
ROLLOUT_ACCEL_ENV = "PYIS_ROLLOUT_ACCEL_TYPE"
# PYIS_PLAIN_BASELINE=1: install ONLY the accelerator-type pool pinning
# (infrastructure placement, equivalent to a nodeSelector) and none of the
# data-path integration — no scheduler routing, no DAS, no engine patches.
# Used for true-baseline arms measuring stock verl + vLLM on the same
# disaggregated topology.
PLAIN_BASELINE_ENV = "PYIS_PLAIN_BASELINE"

_APPLIED = False

# The setup hook runs at Ray worker-process start, BEFORE Ray assigns
# CUDA_VISIBLE_DEVICES to the actor the process will host. Importing
# anything that initializes CUDA here (torch.cuda device queries, vllm)
# poisons the process: torch caches all-GPUs-visible and every FSDP worker
# on a node ends up on physical GPU 0 (measured: NCCL "Duplicate GPU
# detected"). setup() therefore imports NOTHING heavy — each patch is
# installed as a post-import trigger and fires only in processes that
# organically import the target verl module (CPU-side actors: TaskRunner,
# Rollouter, FullyAsyncTrainer — never the FSDP GPU workers).
_POST_IMPORT_TARGETS: dict[str, object] = {}


def setup() -> None:
    """``worker_process_setup_hook`` entry point. Never raises, imports light."""
    global _APPLIED  # noqa: PLW0603
    if _APPLIED:
        return
    _APPLIED = True
    _POST_IMPORT_TARGETS["verl.single_controller.ray.base"] = _patch_resource_pools
    if os.environ.get(PLAIN_BASELINE_ENV) == "1":
        logger.info(
            "fully_async_das: plain-baseline mode — pool pinning only, no "
            "scheduler routing / DAS / engine patches"
        )
    else:
        _POST_IMPORT_TARGETS["verl.experimental.fully_async_policy.fully_async_rollouter"] = (
            _install_fully_async_integration
        )
    sys.meta_path.insert(0, _PostImportPatcher())
    # If a target is somehow already imported, patch it now.
    for name, patch in list(_POST_IMPORT_TARGETS.items()):
        if name in sys.modules:
            _run_patch(name, patch)


def _run_patch(name: str, patch) -> None:
    try:
        patch()
    except Exception:  # noqa: BLE001
        logger.exception("fully_async_das: patch for %s failed", name)


class _PostImportPatcher:
    """meta_path finder that runs a callback right after a target module loads."""

    def __init__(self) -> None:
        self._resolving: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname not in _POST_IMPORT_TARGETS or fullname in self._resolving:
            return None
        self._resolving.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._resolving.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, fullname)
        return spec


class _PatchingLoader:
    def __init__(self, loader, fullname: str) -> None:
        self._loader = loader
        self._fullname = fullname

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _run_patch(self._fullname, _POST_IMPORT_TARGETS[self._fullname])

    def __getattr__(self, item):
        return getattr(self._loader, item)


def _patch_resource_pools() -> None:
    """Pin verl resource pools to GPU types via Ray accelerator_type resources.

    Runs after verl.single_controller.ray.base is imported by the process,
    so it adds no imports of its own beyond a sys.modules lookup.
    """
    trainer_accel = os.environ.get(TRAINER_ACCEL_ENV)
    rollout_accel = os.environ.get(ROLLOUT_ACCEL_ENV)
    if not trainer_accel and not rollout_accel:
        return
    from verl.single_controller.ray.base import RayResourcePool

    if getattr(RayResourcePool, "_pyis_accel_patched", False):
        return

    orig_init = RayResourcePool.__init__

    def pinned_init(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        if self.accelerator_type is not None:
            return
        prefix = self.name_prefix or ""
        # Pool names come from verl: the separated trainer creates
        # "trainer_pool"; standalone rollout replicas create
        # "rollout_pool_<rank>..." (see verl.workers.rollout.replica).
        if trainer_accel and prefix.startswith("trainer_pool"):
            self.accelerator_type = trainer_accel
        elif rollout_accel and prefix.startswith("rollout_pool"):
            self.accelerator_type = rollout_accel
        if self.accelerator_type is not None:
            logger.info(
                "fully_async_das: pool %r pinned to %s", prefix, self.accelerator_type
            )

    RayResourcePool.__init__ = pinned_init
    RayResourcePool._pyis_accel_patched = True


def _install_fully_async_integration() -> None:  # noqa: C901
    try:
        import verl.experimental.fully_async_policy.fully_async_rollouter as far
        from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
    except Exception as e:  # noqa: BLE001
        logger.info("fully_async_das: fully-async verl modules unavailable (%s)", e)
        return
    if getattr(far, "_pyis_patched", False):
        return

    # Importing the hook applies VllmEnginePatch/SglangEnginePatch in this
    # process — required in the Rollouter actor, where the engine server
    # actor classes are captured for Ray by value.
    from integration.verl.verl_hook import (
        _VERL_LAYOUT,
        AgentLoopWorker,
        InferenceSchedulerServerClient,
        _with_validate_gating,
    )

    if _VERL_LAYOUT != "modern":
        logger.warning("fully_async_das: requires the modern (>=0.9) verl layout; skipping")
        return

    import ray

    from py_inference_scheduler.speculative.config import load_das_config

    class FullyAsyncInferenceSchedulerServerClient(
        FullyAsyncLLMServerClient, InferenceSchedulerServerClient
    ):
        """Partial-rollout resume loop over scheduler-routed generation.

        MRO does the composition: ``FullyAsyncLLMServerClient.generate`` is
        the abort/resume loop, and its ``super().generate(...)`` resolves to
        ``InferenceSchedulerServerClient.generate`` — so every segment of a
        (possibly interrupted) rollout is routed by py-inference-scheduler,
        carries the DAS problem-hash request id, and pushes its response
        tokens to the trajectory store. Resumed segments reuse the original
        problem hash via the trajectory-sticky request-id mapping.
        """

        _pyis_das_service = None
        _pyis_das_version = -1

        async def generate(  # noqa: PLR0913 - mirrors FullyAsyncLLMServerClient
            self,
            request_id,
            *,
            prompt_ids,
            sampling_params,
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=None,
        ):
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            self._pyis_note_param_version(output)
            return output

        def _pyis_note_param_version(self, output) -> None:
            """Advance the DAS iteration when the engine weight version bumps.

            Engines stamp responses with their current param version; with
            per-iteration delta serving this is the step boundary. Racing
            callers are harmless: begin_iteration is monotone and idempotent.
            """
            if self.core.das is None:
                return
            try:
                extra = getattr(output, "extra_fields", None) or {}
                version = extra.get("max_global_steps")
                if version is None:
                    version = extra.get("global_steps")
                if version is None:
                    return
                version = int(version)
                if version <= self._pyis_das_version:
                    return
                self._pyis_das_version = version
                if self._pyis_das_service is None:
                    from py_inference_scheduler.speculative.service import (
                        get_or_create_suffix_service,
                    )

                    self._pyis_das_service = get_or_create_suffix_service(load_das_config())
                # Engine versions may start at 0; service iterations start at 1.
                self._pyis_das_service.begin_iteration.remote(version + 1)
                logger.info("fully_async_das: DAS iteration advanced to %d", version + 1)
            except Exception as e:  # noqa: BLE001
                logger.warning("fully_async_das: begin_iteration failed: %s", e)

    class PyInferenceFullyAsyncAgentLoopWorker(AgentLoopWorker):  # type: ignore[valid-type,misc]
        """Swap the incoming client for the scheduler-backed resume client."""

        def __init__(
            self,
            config,
            llm_client,
            teacher_client=None,
            reward_loop_worker_handles=None,
        ) -> None:
            scheduler_client = FullyAsyncInferenceSchedulerServerClient(
                config, load_balancer_handle=llm_client._load_balancer
            )
            self._pyis_client = scheduler_client
            super().__init__(config, scheduler_client, teacher_client, reward_loop_worker_handles)

        # Eval batches must not feed the DAS trajectory store.
        generate_sequences = _with_validate_gating(AgentLoopWorker.generate_sequences)

    base_manager = far.FullyAsyncAgentLoopManager

    class PyInferenceFullyAsyncAgentLoopManager(base_manager):  # type: ignore[valid-type,misc]
        """FullyAsyncAgentLoopManager spawning scheduler-routing workers."""

        def __init__(self, *args, **kwargs) -> None:
            # Base AgentLoopManager honors a pre-set workers class.
            self.agent_loop_workers_class = ray.remote(PyInferenceFullyAsyncAgentLoopWorker)
            super().__init__(*args, **kwargs)

    far.FullyAsyncAgentLoopManager = PyInferenceFullyAsyncAgentLoopManager
    far._pyis_patched = True
    logger.info(
        "fully_async_das: installed scheduler-routing agent loop for the "
        "fully-async (disaggregated) trainer"
    )
