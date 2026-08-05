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
"""verl integration hook: delegate rollout routing to py-inference-scheduler.

Supports two verl layouts, auto-detected at import time:

- **legacy** (v0.7.1): ``AsyncLLMServerManager`` lives in
  ``verl.experimental.agent_loop.agent_loop`` and owns the server list.
- **modern** (v0.9.x): ``LLMServerClient`` lives in
  ``verl.workers.rollout.llm_server``; a ``GlobalRequestLoadBalancer`` Ray
  actor owns the server registry and does atomic acquire. The scheduler client
  bootstraps its endpoint set by draining the balancer once at first use
  (acquire every server with unique request ids, record the handles, release).

Both layouts expose the same entrypoint for the trainer flag:
``+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager``
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections import OrderedDict

import ray
from omegaconf import DictConfig  # type: ignore[import-not-found]

try:  # legacy layout (verl v0.7.x)
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopManager,
        AgentLoopWorker,
    )
    from verl.experimental.agent_loop.agent_loop import (
        AsyncLLMServerManager as _LegacyServerManager,
    )

    _VERL_LAYOUT = "legacy"
except ImportError:  # modern layout (verl v0.9.x)
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopManager,
        AgentLoopWorker,
    )
    from verl.workers.rollout.llm_server import (  # type: ignore[import-not-found]
        LLMServerClient as _ModernServerClient,
    )

    _VERL_LAYOUT = "modern"

from backends.verl.sglang import SglangEnginePatch
from backends.verl.vllm import VllmEnginePatch
from py_inference_scheduler import Scheduler
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint, LLMRequest
from py_inference_scheduler.speculative.config import das_enabled, load_das_config
from py_inference_scheduler.speculative.contracts import TrajectoryPush
from py_inference_scheduler.speculative.problem_id import encode_request_id, hash_problem_id

logger = logging.getLogger(__name__)
logger.info("py-inference-scheduler verl hook: %s layout detected", _VERL_LAYOUT)

# Must apply at module level to patch classes before use across distributed
# Ray workers without modifying verl.
VllmEnginePatch.apply()
SglangEnginePatch.apply()


def _rollout_config(config: DictConfig):
    if config.get("actor_rollout_ref"):
        return config.actor_rollout_ref.rollout
    return config.rollout


_TRAJ_PROBLEM_LRU_CAPACITY = 65536


class _DASHookState:
    """Per-worker DAS state: problem-id mapping + fire-and-forget pushes.

    Trajectory-sticky verl request ids map to a problem hash at first turn,
    so multi-turn follow-ups (whose prompts have grown) stay on the problem
    keyed by the first-turn prompt.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.push_enabled = True
        self._service = None
        self._push_client = None
        self._traj_problems: OrderedDict[str, str] = OrderedDict()
        self._errors = 0

    @classmethod
    def create(cls) -> _DASHookState | None:
        if not das_enabled():
            return None
        try:
            return cls(load_das_config())
        except Exception:
            logger.exception("DAS: hook state init failed; DAS disabled for this worker")
            return None

    def problem_for(self, request_id: str, prompt_ids: list[int] | None) -> str | None:
        phash = self._traj_problems.get(request_id)
        if phash is not None:
            self._traj_problems.move_to_end(request_id)
            return phash
        if not prompt_ids:
            return None
        phash = hash_problem_id(prompt_ids)
        self._traj_problems[request_id] = phash
        while len(self._traj_problems) > _TRAJ_PROBLEM_LRU_CAPACITY:
            self._traj_problems.popitem(last=False)
        return phash

    def record_completion(
        self, phash: str, output: object, prompt_len: int, server_id: str
    ) -> None:
        """Push one completed engine call's response tokens. Never raises."""
        if not self.push_enabled:
            return
        try:
            tokens = getattr(output, "token_ids", output)
            if not isinstance(tokens, (list, tuple)) or not tokens:
                return
            self._ensure_push_client().enqueue(
                TrajectoryPush(
                    problem_id=phash,
                    token_ids=tuple(int(t) for t in tokens),
                    prompt_len=prompt_len,
                    server_id=server_id,
                )
            )
        except Exception as e:  # noqa: BLE001
            self._errors += 1
            if self._errors == 1 or self._errors % 1000 == 0:
                logger.warning("DAS: trajectory push failed (%d so far): %s", self._errors, e)

    def _ensure_push_client(self):
        if self._push_client is None:
            from py_inference_scheduler.speculative.push_client import DASPushClient
            from py_inference_scheduler.speculative.service import get_or_create_suffix_service

            self._service = get_or_create_suffix_service(self.cfg)
            self._push_client = DASPushClient(
                self._service,
                worker_id=f"hook-{uuid.uuid4().hex[:8]}",
                config=self.cfg.push,
            )
        return self._push_client


def _worker_das(worker) -> _DASHookState | None:
    client = getattr(worker, "server_manager", None) or getattr(worker, "_pyis_client", None)
    core = getattr(client, "core", None)
    return getattr(core, "das", None)


def _with_validate_gating(base_fn):
    """Wrap AgentLoopWorker.generate_sequences: no pushes for eval batches."""

    def _apply(self, batch) -> None:
        das = _worker_das(self)
        if das is None:
            return
        try:
            validate = bool(getattr(batch, "meta_info", {}).get("validate", False))
        except Exception:  # noqa: BLE001
            validate = False
        das.push_enabled = not validate

    if inspect.iscoroutinefunction(base_fn):
        async def async_wrapper(self, batch, *args, **kwargs):
            _apply(self, batch)
            return await base_fn(self, batch, *args, **kwargs)

        return async_wrapper

    def sync_wrapper(self, batch, *args, **kwargs):
        _apply(self, batch)
        return base_fn(self, batch, *args, **kwargs)

    return sync_wrapper


class _SchedulerCore:
    """Layout-independent scheduling state: engine, inflight tracking, metrics."""

    def __init__(self) -> None:
        self.scheduler = Scheduler()
        self.inflight_store = InflightStore()
        self.endpoints: list[Endpoint] = []
        self.lb_acquired_requests: set[str] = set()
        self.lock = asyncio.Lock()
        self.das = _DASHookState.create()

    def das_request_id(
        self, request_id: str, prompt_ids: list[int] | None
    ) -> tuple[str | None, str]:
        """(problem_hash, engine_request_id) for one engine call.

        A fresh engine-side id avoids vLLM KV-cache collisions with verl's
        sticky multi-turn ids; with DAS on it carries the problem hash so
        the patched proposer routes to the request's per-problem tree.
        """
        if self.das is not None:
            try:
                phash = self.das.problem_for(request_id, prompt_ids)
            except Exception as e:  # noqa: BLE001
                logger.debug("DAS: problem hash failed for %s: %s", request_id, e)
                phash = None
            if phash is not None:
                return phash, encode_request_id(phash)
        return None, uuid.uuid4().hex

    def das_record(
        self, phash: str | None, output: object, prompt_len: int, server_id: str
    ) -> None:
        if self.das is not None and phash is not None:
            self.das.record_completion(phash, output, prompt_len, server_id)

    async def schedule(self, request_id: str, prompt_ids: list[int] | None) -> Endpoint | None:
        """Refresh metrics and pick an endpoint; None means fall back to verl's LB.

        The lock makes metric refresh part of the scheduling task itself:
        verl composes the whole batch before any task runs, so an independent
        poller task would never be interleaved by the FIFO event loop.
        """
        async with self.lock:
            await asyncio.gather(
                *(fetch_worker_metrics(ep, self.inflight_store) for ep in self.endpoints)
            )
            for ep in self.endpoints:
                ep.attributes["queue_len"] = self.inflight_store.get(ep.name)

            request = LLMRequest(request_id=request_id, body=prompt_ids)
            selected = self.scheduler.run(request, candidates=self.endpoints)
            if not selected:
                return None
            winner: Endpoint = selected[0].endpoint
            self.inflight_store.increment(winner.name)
            return winner


if _VERL_LAYOUT == "legacy":

    class InferenceSchedulerServerManager(_LegacyServerManager):  # type: ignore[misc]
        """Delegate routing to py-inference-scheduler. Compatible with verl v0.7.1."""

        def __init__(
            self,
            config: DictConfig,
            servers: list[tuple[str, ray.actor.ActorHandle]],
            load_balancer_handle: ray.actor.ActorHandle,
            *args: object,
            **kwargs: object,
        ) -> None:
            super().__init__(config, servers, load_balancer_handle, *args, **kwargs)
            self.rollout_config = _rollout_config(config)
            self.core = _SchedulerCore()
            self.core.endpoints = [
                Endpoint(name=server_id, attributes={"replica_obj": handle, "routing_stats": {}})
                for server_id, handle in servers
            ]

        async def _acquire_server(
            self,
            request_id: str,
            prompt_ids: list[int] | None = None,
        ) -> tuple[str, ray.actor.ActorHandle]:
            winner = await self.core.schedule(request_id, prompt_ids)
            if winner is None:
                logger.warning(
                    "py-inference-scheduler returned no endpoints, falling back to verl global LB."
                )
                self.core.lb_acquired_requests.add(request_id)
                server_id, handle = await super()._acquire_server(request_id)  # type: ignore[no-any-return]
                self.core.inflight_store.increment(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.inflight_store.decrement(server_id)
            if request_id and request_id in self.core.lb_acquired_requests:
                super()._release_server(server_id)
                self.core.lb_acquired_requests.remove(request_id)

        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, object],
            image_data: list[object] | None = None,
            video_data: list[object] | None = None,
        ) -> object:
            # Yield CPU so queued metric/scheduling tasks can interleave.
            await asyncio.sleep(0)
            server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

            # vLLMAsyncServer ignores ignore_eos from config, so pass it explicitly.
            # A fresh request_id per generation avoids vLLM KV-cache collisions
            # with verl's sticky multi-turn request ids.
            ignore_eos = self.rollout_config.get("ignore_eos", False)
            if isinstance(sampling_params, dict):
                sampling_params["ignore_eos"] = ignore_eos
            elif hasattr(sampling_params, "ignore_eos"):
                sampling_params.ignore_eos = ignore_eos

            phash, engine_request_id = self.core.das_request_id(request_id, prompt_ids)
            try:
                output = await server.generate.remote(
                    request_id=engine_request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                )
                self.core.das_record(phash, output, len(prompt_ids), server_id)
                return output
            finally:
                self._release_server(server_id, request_id)

    class PyInferenceAgentLoopWorker(AgentLoopWorker):  # type: ignore[misc]
        """Inject the custom ServerManager before calling super().__init__."""

        def __init__(
            self,
            config: DictConfig,
            servers: list[tuple[str, ray.actor.ActorHandle]],
            load_balancer_handle: ray.actor.ActorHandle,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
        ) -> None:
            self.server_manager = InferenceSchedulerServerManager(
                config, servers, load_balancer_handle
            )
            super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)

        # Eval batches must not feed the DAS trajectory store (speculation
        # itself stays on: it is lossless either way).
        generate_sequences = _with_validate_gating(AgentLoopWorker.generate_sequences)

else:  # modern layout

    class InferenceSchedulerServerClient(_ModernServerClient):  # type: ignore[misc]
        """Delegate routing to py-inference-scheduler. Compatible with verl v0.9.x.

        The GlobalRequestLoadBalancer actor owns the (server_id -> handle)
        registry but exposes no enumeration API, so the endpoint set is
        bootstrapped once by draining it: with all inflight counters equal,
        consecutive acquires with unique request ids visit every server.
        """

        def __init__(
            self,
            config: DictConfig,
            load_balancer_handle: ray.actor.ActorHandle = None,
            **kwargs: object,
        ) -> None:
            super().__init__(config, load_balancer_handle, **kwargs)
            self.rollout_config = _rollout_config(config)
            self.core = _SchedulerCore()

        async def _ensure_endpoints(self) -> None:
            if self.core.endpoints:
                return
            server_ids = await self._load_balancer.get_all_servers.remote()
            handles: dict[str, ray.actor.ActorHandle] = {}
            acquired: list[str] = []
            for _ in range(max(1, len(server_ids)) * 3):
                server_id, handle = await self._load_balancer.acquire_server.remote(
                    request_id=f"pyis-bootstrap-{uuid.uuid4().hex}"
                )
                acquired.append(server_id)
                handles[server_id] = handle
                if len(handles) >= len(server_ids):
                    break
            for server_id in acquired:
                self._load_balancer.release_server.remote(server_id=server_id)
            self.core.endpoints = [
                Endpoint(name=server_id, attributes={"replica_obj": handle, "routing_stats": {}})
                for server_id, handle in handles.items()
            ]
            logger.info(
                "py-inference-scheduler bootstrapped %d endpoints from global LB", len(handles)
            )

        async def _acquire_server(
            self,
            request_id: str,
            prompt_ids: list[int] | None = None,
        ) -> tuple[str, ray.actor.ActorHandle]:
            await self._ensure_endpoints()
            winner = await self.core.schedule(request_id, prompt_ids)
            if winner is None:
                logger.warning(
                    "py-inference-scheduler returned no endpoints, falling back to verl global LB."
                )
                self.core.lb_acquired_requests.add(request_id)
                server_id, handle = await super()._acquire_server(request_id)
                self.core.inflight_store.increment(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.inflight_store.decrement(server_id)
            if request_id and request_id in self.core.lb_acquired_requests:
                super()._release_server(server_id)
                self.core.lb_acquired_requests.remove(request_id)

        async def generate(  # noqa: PLR0913 - mirrors verl's LLMServerClient.generate
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, object],
            image_data: list[object] | None = None,
            video_data: list[object] | None = None,
            audio_data: list[object] | None = None,
            mm_processor_kwargs: dict[str, object] | None = None,
            **kwargs: object,
        ) -> object:
            await asyncio.sleep(0)
            server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

            ignore_eos = self.rollout_config.get("ignore_eos", False)
            if isinstance(sampling_params, dict):
                sampling_params["ignore_eos"] = ignore_eos

            multimodal_kwargs: dict[str, object] = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            phash, engine_request_id = self.core.das_request_id(request_id, prompt_ids)
            try:
                output = await server.generate.remote(
                    request_id=engine_request_id,  # fresh id per turn, mirrors upstream
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    **multimodal_kwargs,
                    **kwargs,
                )
                self.core.das_record(phash, output, len(prompt_ids), server_id)
                return output
            finally:
                self._release_server(server_id, request_id)

    class PyInferenceAgentLoopWorker(AgentLoopWorker):  # type: ignore[misc,no-redef]
        """Swap the incoming LLMServerClient for the scheduler-backed client."""

        def __init__(
            self,
            config: DictConfig,
            llm_client: object,
            teacher_client: dict | None = None,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
        ) -> None:
            scheduler_client = InferenceSchedulerServerClient(
                config, load_balancer_handle=llm_client._load_balancer
            )
            self._pyis_client = scheduler_client
            super().__init__(config, scheduler_client, teacher_client, reward_loop_worker_handles)

        # Eval batches must not feed the DAS trajectory store (speculation
        # itself stays on: it is lossless either way).
        generate_sequences = _with_validate_gating(AgentLoopWorker.generate_sequences)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """Main hook entrypoint loaded by ray_trainer.py.

    Overrides the worker actor class that verl spawns across the cluster.
    Works on both supported verl layouts (the worker class above is selected
    at import time).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        self._das_service = None
        self._das_step = 0
        if das_enabled():
            try:
                from py_inference_scheduler.speculative.service import (
                    get_or_create_suffix_service,
                )

                self._das_service = get_or_create_suffix_service(load_das_config())
                logger.info("DAS: SuffixDataService attached (named detached actor)")
            except Exception:
                logger.exception("DAS: SuffixDataService unavailable; continuing without DAS")
        super().__init__(*args, **kwargs)

    def _das_begin_iteration(self, prompts) -> None:
        """Advance the central sliding window once per training step."""
        if self._das_service is None:
            return
        try:
            meta = getattr(prompts, "meta_info", None) or {}
            if meta.get("validate", False):
                return
            step = int(meta.get("global_steps", self._das_step + 1))
            step = max(step, self._das_step + 1)
            self._das_step = step
            ray.get(self._das_service.begin_iteration.remote(step), timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("DAS: begin_iteration failed (window not advanced): %s", e)

    def generate_sequences(self, prompts, *args: object, **kwargs: object) -> object:
        self._das_begin_iteration(prompts)
        return super().generate_sequences(prompts, *args, **kwargs)
