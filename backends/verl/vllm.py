from __future__ import annotations

import asyncio
import inspect
import logging
import socket
import uuid

from py_inference_scheduler.datalayer.metrics.verl.vllm import get_vllm_routing_stats
from py_inference_scheduler.speculative.config import DASConfig, das_enabled, load_das_config

logger = logging.getLogger(__name__)

_DAS_WORKER_EXTENSION = "py_inference_scheduler.speculative.worker_extension.DASWorkerExtension"


class VllmEnginePatch:
    """Monkey-patching vLLM V1 (0.11.0+) to allow metrics extraction."""

    @classmethod
    def apply(cls) -> None:
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import (  # type: ignore[import-not-found]
                vLLMHttpServer,
            )
            from vllm.ray import ray_env  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001 - CPU-only nodes raise beyond ImportError (triton)
            logger.info("Skipping vLLM patch (normal on head node if vLLM is not installed): %s", e)
            return

        try:
            # Patch get_env_vars_to_copy to include PROMETHEUS_MULTIPROC_DIR
            original_get_env_vars = ray_env.get_env_vars_to_copy

            def patched_get_env_vars(destination="DPEngineCoreActor"):
                vars_list = original_get_env_vars(destination)
                if "PROMETHEUS_MULTIPROC_DIR" not in vars_list:
                    vars_list.append("PROMETHEUS_MULTIPROC_DIR")
                return vars_list

            ray_env.get_env_vars_to_copy = patched_get_env_vars
            vLLMHttpServer.get_routing_stats = get_vllm_routing_stats

            # Patch launch_server to create metrics directory on worker
            original_launch = vLLMHttpServer.launch_server

            async def patched_launch(self, *args, **kwargs):
                import os
                metrics_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR', '/tmp/metrics')  # noqa: S108
                os.makedirs(metrics_dir, exist_ok=True)  # noqa: PTH103
                os.environ['PROMETHEUS_MULTIPROC_DIR'] = metrics_dir
                return await original_launch(self, *args, **kwargs)

            vLLMHttpServer.launch_server = patched_launch

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to apply vLLM patch: %s", e)

        if das_enabled():
            try:
                cls._apply_das(vLLMHttpServer)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to apply DAS vLLM patch: %s", e)

    @classmethod
    def _apply_das(cls, http_server_cls) -> None:
        """DAS wiring on the vLLMHttpServer actor (process0 of each replica).

        - route verl's worker_extension_cls to DASWorkerExtension so the DAS
          RPCs exist inside every GPU worker (and, as a fallback injection
          path, so the proposer patch applies even without pip metadata);
        - start the delta pump after the engine server launches;
        - gate the pump while the engine is slept for training.
        """
        if getattr(http_server_cls, "_das_patched", False):
            return
        cfg = load_das_config()

        if hasattr(http_server_cls, "_get_worker_extension_cls"):
            def das_worker_extension_cls(self):
                return _DAS_WORKER_EXTENSION

            http_server_cls._get_worker_extension_cls = das_worker_extension_cls
        else:
            logger.warning(
                "DAS: vLLMHttpServer has no _get_worker_extension_cls; delta delivery "
                "RPCs unavailable, engines will run stock suffix decoding only"
            )

        das_original_launch = http_server_cls.launch_server

        async def das_launch(self, *args, **kwargs):
            result = await das_original_launch(self, *args, **kwargs)
            try:
                _start_das_pump(self, cfg)
            except Exception:
                logger.exception("DAS: delta pump failed to start; engine runs stock suffix")
            return result

        http_server_cls.launch_server = das_launch

        original_sleep = getattr(http_server_cls, "sleep", None)
        if original_sleep is not None:
            async def das_sleep(self, *args, **kwargs):
                self._das_engine_sleeping = True
                return await original_sleep(self, *args, **kwargs)

            http_server_cls.sleep = das_sleep

        original_wake = getattr(http_server_cls, "wake_up", None)
        if original_wake is not None:
            async def das_wake_up(self, *args, **kwargs):
                result = await original_wake(self, *args, **kwargs)
                self._das_engine_sleeping = False
                return result

            http_server_cls.wake_up = das_wake_up

        http_server_cls._das_patched = True
        logger.info("DAS: vLLMHttpServer patched (worker extension + delta pump)")


def _detect_dp_size(server) -> int:
    """Best-effort data_parallel_size lookup; DAS supports 1 per engine."""
    for holder in (server, getattr(server, "config", None)):
        if holder is None:
            continue
        for attr in ("data_parallel_size", "dp_size"):
            try:
                value = getattr(holder, attr, None)
                if value is None and hasattr(holder, "get"):
                    value = holder.get(attr)
                if value:
                    return int(value)
            except Exception:  # noqa: BLE001,S112,PERF203
                continue
    return 1


def _start_das_pump(server, cfg: DASConfig) -> None:
    if getattr(server, "_das_pump_task", None) is not None:
        return
    if not hasattr(server, "collective_rpc"):
        logger.warning(
            "DAS: this verl vLLMHttpServer has no collective_rpc (needs verl >= 0.9); "
            "delta delivery disabled, engines run stock suffix decoding"
        )
        return
    if _detect_dp_size(server) > 1:
        logger.warning(
            "DAS: data_parallel_size > 1 within one engine is unsupported; "
            "collective_rpc reaches a single DP shard. DAS disabled for this "
            "engine (stock suffix decoding stays on)."
        )
        return
    server._das_engine_sleeping = getattr(server, "_das_engine_sleeping", False)
    loop = asyncio.get_running_loop()
    server._das_pump_task = loop.create_task(_das_delta_pump(server, cfg))
    logger.info("DAS: delta pump started (poll every %.1fs)", cfg.poll_interval_s)


async def _das_collective_rpc(server, method: str, payload: bytes):
    """Call verl's collective_rpc tolerating minor signature drift."""
    try:
        result = server.collective_rpc(method, args=(payload,))
    except TypeError:
        result = server.collective_rpc(method=method, args=(payload,))
    if inspect.isawaitable(result):
        result = await result
    return result


async def _das_delta_pump(server, cfg: DASConfig) -> None:
    """Pull versioned deltas from the central store into all TP workers.

    Runs forever on the vLLMHttpServer actor's event loop. Never lets an
    error escape: DAS data flow degrades, the engine itself is untouched.
    """
    import ray

    from py_inference_scheduler.speculative.contracts import deserialize_delta_batch

    replica_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    service = None
    version = -1
    errors = 0
    while True:
        try:
            if getattr(server, "_das_engine_sleeping", False):
                await asyncio.sleep(1.0)
                continue
            if service is None:
                try:
                    service = ray.get_actor(cfg.service_name, namespace=cfg.service_namespace)
                except Exception:  # noqa: BLE001
                    # Service not up yet (e.g. engines launch before the
                    # trainer's manager); keep waiting quietly.
                    await asyncio.sleep(5.0)
                    continue
            payload = await service.get_deltas.remote(replica_id, version)
            if payload is None:
                await asyncio.sleep(cfg.poll_interval_s)
                continue
            batch = deserialize_delta_batch(payload)
            await _das_collective_rpc(server, "das_apply_deltas", payload)
            version = batch.to_version
            errors = 0
            # Loop immediately: the batch may have been size-bounded.
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors == 1 or errors % 30 == 0:
                logger.warning("DAS: delta pump error (%d consecutive): %s", errors, e)
            service = None
            await asyncio.sleep(min(30.0, cfg.poll_interval_s * (2 ** min(errors, 4))))
