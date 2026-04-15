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
from typing import Sequence
from scheduling.framework import Endpoint
from datalayer.verl.datastore import InflightStore

logger = logging.getLogger(__name__)


async def fetch_worker_metrics(ep: Endpoint, inflight_store: InflightStore):
    """Request stats from a single worker via RPC and update the endpoint attributes."""
    actor = ep.attributes.get("replica_obj")
    if not actor:
        return
    try:
        stats = await actor.get_routing_stats.remote()
        if stats.get("error"):
            logger.error(f"RPC stats error for {ep.name}: {stats['error']}")
        local_inflight = inflight_store.get(ep.name)
        ep.attributes["queue_len"] = local_inflight
        ep.attributes["routing_stats"] = {
            "num_waiting_reqs": stats.get("num_waiting_reqs", 0),
            "num_running_reqs": stats.get("num_running_reqs", 0),
            "kv": stats.get("kv", 0.0)
        }
    except Exception as e:
        logger.error(f"Failed to scrape RPC metrics for {ep.name}: {e}")


async def verl_metrics_polling_loop(endpoints: Sequence[Endpoint], inflight_store: InflightStore):
    """Asynchronously pulls metrics directly from the engine via the injected RPC hook."""
    while True:
        try:
            tasks = [fetch_worker_metrics(ep, inflight_store) for ep in endpoints]
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Metrics poll error: {e}")
        
        await asyncio.sleep(0.05)
