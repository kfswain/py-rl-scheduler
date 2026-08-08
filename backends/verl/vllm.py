from __future__ import annotations

import asyncio
import inspect
import logging
import pathlib
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
                _clean_dead_multiproc_files(metrics_dir)
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
                # Boundary drain, fire-and-forget: applies overlap the
                # rollout's prefill phase instead of blocking its start.
                prior = getattr(self, "_das_drain_task", None)
                if (
                    getattr(self, "_das_replica_id", None)
                    and (prior is None or prior.done())
                    and _das_ensure_service(self, cfg)
                ):
                    self._das_drain_task = asyncio.get_running_loop().create_task(
                        _das_boundary_drain_bg(self, cfg)
                    )
                return result

            http_server_cls.wake_up = das_wake_up

        http_server_cls._das_patched = True
        logger.info("DAS: vLLMHttpServer patched (worker extension + delta pump)")


def _clean_dead_multiproc_files(metrics_dir: str) -> None:
    """Remove prometheus multiproc files left by dead processes.

    The multiproc dir is shared pod state that survives jobs; every scrape
    aggregates ALL files, so dead-job residue makes /metrics slower for
    every subsequent run (and the scheduler scrapes per request). Files of
    live pids (concurrently launching sibling engines) are untouched.
    """
    import re

    removed = 0
    try:
        for path in pathlib.Path(metrics_dir).iterdir():
            m = re.match(r".+_(\d+)\.db$", path.name)
            if m and not pathlib.Path(f"/proc/{m.group(1)}").exists():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    except OSError as e:
        logger.warning("multiproc cleanup skipped: %s", e)
        return
    if removed:
        logger.info("cleaned %d dead prometheus multiproc files from %s", removed, metrics_dir)


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
    server._das_service = None
    server._das_version = -1
    server._das_replica_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    server._das_buffer = []  # prefetched (to_version, payload) pairs, in order
    server._das_buffer_bytes = 0
    loop = asyncio.get_running_loop()
    server._das_pump_task = loop.create_task(_das_delta_pump(server, cfg))
    logger.info(
        "DAS: delta pump started (service-side prefetch, poll %.1fs; ALL engine "
        "applies happen in the wake_up boundary drain — zero mid-decode RPCs)",
        cfg.poll_interval_s,
    )


async def _das_collective_rpc(server, method: str, payload: bytes | None = None):
    """Call verl's collective_rpc tolerating minor signature drift."""
    rpc_args = (payload,) if payload is not None else ()
    try:
        result = server.collective_rpc(method, args=rpc_args)
    except TypeError:
        result = server.collective_rpc(method=method, args=rpc_args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _das_ensure_service(server, cfg: DASConfig) -> bool:
    if getattr(server, "_das_service", None) is not None:
        return True
    import ray

    try:
        server._das_service = ray.get_actor(cfg.service_name, namespace=cfg.service_namespace)
    except Exception:  # noqa: BLE001
        # Service not up yet (engines launch before the trainer's manager).
        return False
    return True


# Prefetch buffer cap: beyond this, leave data at the service until the
# next boundary rather than growing process0 memory.
_DAS_BUFFER_CAP_BYTES = 256 * 1024 * 1024


async def _das_prefetch_available(server, cfg: DASConfig) -> int:
    """Pull delta payloads from the service into the local buffer.

    Touches only the service actor — never the engine — so it is safe at
    any time, including mid-decode.
    """
    from py_inference_scheduler.speculative.contracts import deserialize_delta_batch

    fetched = 0
    while server._das_buffer_bytes < _DAS_BUFFER_CAP_BYTES:
        next_version = (
            server._das_buffer[-1][0] if server._das_buffer else server._das_version
        )
        payload = await server._das_service.get_deltas.remote(
            server._das_replica_id, next_version
        )
        if payload is None:
            return fetched
        batch = deserialize_delta_batch(payload)
        server._das_buffer.append((batch.to_version, payload))
        server._das_buffer_bytes += len(payload)
        fetched += 1
    return fetched


async def _das_boundary_drain_bg(server, cfg: DASConfig) -> None:
    try:
        applied = await _das_apply_boundary(server, cfg)
        if applied:
            logger.info("DAS: boundary drain applied %d batches", applied)
    except Exception:
        logger.exception("DAS: boundary drain failed; next boundary catches up")


async def _das_apply_boundary(server, cfg: DASConfig) -> int:
    """Apply buffered payloads + any remainder. Engine idle: RPCs are cheap.

    The only place engine-side applies ever happen.
    """
    await _das_prefetch_available(server, cfg)
    applied = 0
    while server._das_buffer:
        to_version, payload = server._das_buffer.pop(0)
        server._das_buffer_bytes -= len(payload)
        await _das_collective_rpc(server, "das_apply_deltas", payload)
        server._das_version = to_version
        applied += 1
    return applied


async def _das_delta_pump(server, cfg: DASConfig) -> None:
    """Service-side prefetch loop. Never touches the engine.

    Payload delivery to the GPU workers happens exclusively in the wake_up
    boundary drain; this loop just keeps the local buffer warm so the
    boundary drain is a handful of local applies instead of a fetch storm.
    """
    errors = 0
    while True:
        try:
            if not _das_ensure_service(server, cfg):
                await asyncio.sleep(5.0)
                continue
            await _das_prefetch_available(server, cfg)
            errors = 0
            await asyncio.sleep(cfg.poll_interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors == 1 or errors % 30 == 0:
                logger.warning("DAS: delta prefetch error (%d consecutive): %s", errors, e)
            server._das_service = None
            await asyncio.sleep(min(30.0, cfg.poll_interval_s * (2 ** min(errors, 4))))
