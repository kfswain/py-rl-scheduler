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


import asyncio
import logging
import ray
import os
import uuid
import yaml
from typing import Dict, List, Any
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AsyncLLMServerManager,
    AgentLoopWorker,
    AgentLoopManager
)

from scheduling.scheduler import Scheduler
from scheduling.scheduler_config import SchedulerConfig
from scheduling.types import LLMRequest, Endpoint
from scheduling.inflight_store import InflightStore
from scheduling.plugins import (
    SimpleFilter,
    WaitingQueueScorer,
    MaxScorePicker,
    SingleProfileHandler
)
from scheduling.framework import SchedulerProfile
from scheduling.ray_request_scheduler import RayRequestScheduler

logger = logging.getLogger(__name__)

class InferenceSchedulerServerManager(AsyncLLMServerManager):
    """
    Subclass of verl's AsyncLLMServerManager that delegates routing
    to the native py-inference-scheduler engine.
    """
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], *args, **kwargs):
        super().__init__(config, server_handles, *args, **kwargs)
        self.ray_request_scheduler = RayRequestScheduler()
        self.inflight_store = InflightStore()
        self.endpoints = []

        for i, rep_handle in enumerate(self.server_handles):
            ep = Endpoint(
                name=f"verl-worker-{i}",
                attributes={
                    "replica_obj": rep_handle,
                    "routing_stats": {}
                }
            )
            self.endpoints.append(ep)

        self._metrics_task = None

    async def poll_worker_metrics_loop(self):
        """Asynchronously scrapes metrics to update endpoint queue sizes (50ms interval)"""
        import os
        import aiohttp

        self._worker_urls = None

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    if self._worker_urls is None:
                        discovered_urls = []
                        for ep in self.endpoints:
                            actor = ep.attributes["replica_obj"]
                            try:
                                ip, port = await actor.get_server_address.remote()
                                discovered_urls.append(f"http://{ip}:{port}/metrics")
                            except Exception as e:
                                logger.error(f"Failed to fetch IP for {ep.name}: {e}")
                                discovered_urls.append(None)
                        self._worker_urls = discovered_urls
                        logger.info(f"Natively mapped Worker Metrics IPs: {self._worker_urls}")

                    # Map exactly to self._worker_urls index to pair with endpoints
                    async def fetch_worker_metrics(session, idx, url):
                        if url is None:
                            return
                        try:
                            async with session.get(url, timeout=0.200) as response:
                                text = await response.text()
                                stats = {
                                    "num_waiting_reqs": 0,
                                    "num_running_reqs": 0,
                                    "kv": 0.0,
                                    "queue_len": 0
                                }

                                for line in text.split('\n'):
                                    try:
                                        if line.startswith("vllm:num_requests_waiting"):
                                            stats["num_waiting_reqs"] = int(float(line.split(" ")[-1]))
                                        elif line.startswith("vllm:num_requests_running"):
                                            stats["num_running_reqs"] = int(float(line.split(" ")[-1]))
                                        elif line.startswith("vllm:kv_cache_usage_perc"):
                                            stats["kv"] = float(line.split(" ")[-1])
                                    except IndexError:
                                        continue

                                endpoint = self.endpoints[idx]
                                local_inflight = self.inflight_store.get(endpoint.name)
                                stats["num_running_reqs"] = max(stats["num_running_reqs"], local_inflight)

                                stats["queue_len"] = stats["num_waiting_reqs"] + stats["num_running_reqs"]
                                endpoint.attributes["routing_stats"] = stats

                        except asyncio.TimeoutError:
                            logger.debug(f"Timeout connecting to {url}")
                        except Exception as e:
                            logger.error(f"Failed to scrape {url}: {e}")

                    tasks = [fetch_worker_metrics(session, i, url) for i, url in enumerate(self._worker_urls)]
                    await asyncio.gather(*tasks)

                except Exception as e:
                    logger.error(f"Metrics poll error: {e}")

                # Poll at 50ms - same as gateway
                await asyncio.sleep(0.05)

    async def generate(self, request_id: str, **kwargs):
        """Overrides Verl's native generate to forcefully yield to the metrics poller"""

        # Yield CPU to check for metrics poller
        await asyncio.sleep(0)

        prompt_ids = kwargs.get("prompt_ids", None)
        winning_endpoint = self._choose_server(request_id, prompt_ids=prompt_ids)

        if isinstance(winning_endpoint, Endpoint):
            server = winning_endpoint.attributes["replica_obj"]
            endpoint_name = winning_endpoint.name
            self.inflight_store.increment(endpoint_name)
            stats = winning_endpoint.attributes.setdefault("routing_stats", {})
            stats["num_running_reqs"] = max(stats.get("num_running_reqs", 0), self.inflight_store.get(endpoint_name))
            stats["queue_len"] = stats.get("num_waiting_reqs", 0) + stats["num_running_reqs"]
        else:
            server = winning_endpoint
            endpoint_name = None

        # vLLM requires a fresh request_id per generation to prevent KV cache collisions.
        kwargs["request_id"] = uuid.uuid4().hex

        try:
            return await server.generate.remote(**kwargs)
        finally:
            if endpoint_name:
                self.inflight_store.decrement(endpoint_name)

    def _choose_server(self, request_id: str, prompt_ids: List[int] = None) -> ray.actor.ActorHandle:
        """Overrides Verl's Native LRU Router"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self.poll_worker_metrics_loop())

        self.ray_request_scheduler._maybe_reload_config()
        req = LLMRequest(request_id=request_id, body=prompt_ids)
        selected_endpoints = self.ray_request_scheduler.run(req, candidates=self.endpoints)

        if not selected_endpoints:
            logger.warning("py-inference-scheduler returned no endpoints, falling back to basic LRU.")
            return super()._choose_server(request_id)

        winning_endpoint: Endpoint = selected_endpoints[0].endpoint
        stats = winning_endpoint.attributes.get('routing_stats', {})
        logger.info(f"[{request_id[:6]}] Routed to {winning_endpoint.name} (stats: {stats})")
        return winning_endpoint


class PyInferenceAgentLoopWorker(AgentLoopWorker):
    """
    Overrides the Ray worker actor to inject the custom ServerManager
    before calling super().__init__
    """
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], *args, **kwargs):
        self.server_manager = InferenceSchedulerServerManager(config, server_handles)
        super().__init__(config, server_handles, *args, **kwargs)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """
    The main hook entrypoint loaded by ray_trainer.py
    Overrides the worker actor class that verl spawns across the cluster.
    """
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
