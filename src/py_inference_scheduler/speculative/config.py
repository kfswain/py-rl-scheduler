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
"""DAS configuration, loaded from ``DAS_CONFIG_PATH`` yaml with env overrides.

The DAS config is deliberately separate from the router yaml: it is consumed
by different processes (driver, agent-loop workers, vLLM server actors, and
GPU workers), none of which participate in the router's mtime hot-reload.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENABLE_ENV_VAR = "PYIS_DAS_ENABLED"
CONFIG_PATH_ENV_VAR = "DAS_CONFIG_PATH"

# Length classes (ordering matters: budgets may only be upgraded S -> M -> L).
CLS_SHORT = 0
CLS_MEDIUM = 1
CLS_LONG = 2


def das_enabled() -> bool:
    return os.environ.get(ENABLE_ENV_VAR, "") == "1"


@dataclass(frozen=True)
class DASBudgetConfig:
    """Length-aware speculation budgets (max draft tokens per class)."""

    long: int = 24
    medium: int = 8
    short: int = 0  # 0 == skip speculation entirely

    def for_class(self, cls: int) -> int:
        if cls >= CLS_LONG:
            return self.long
        if cls == CLS_MEDIUM:
            return self.medium
        return self.short


@dataclass(frozen=True)
class DASClassifierConfig:
    """Generation-length thresholds separating Short/Medium/Long problems."""

    short_max_tokens: int = 256
    long_min_tokens: int = 1024

    def class_for_length(self, length: float) -> int:
        if length >= self.long_min_tokens:
            return CLS_LONG
        if length >= self.short_max_tokens:
            return CLS_MEDIUM
        return CLS_SHORT


@dataclass(frozen=True)
class DASServiceLimits:
    max_total_tokens: int = 64_000_000
    max_problems: int = 8192
    max_seqs_per_problem: int = 512


@dataclass(frozen=True)
class DASPushConfig:
    batch_size: int = 64
    flush_interval_s: float = 0.5
    queue_capacity: int = 4096


@dataclass(frozen=True)
class DASConfig:
    enabled: bool = True
    # Sliding window over training iterations retained in the central store.
    window_iterations: int = 16
    # Iterations whose trajectories also carry a 2x-count shadow copy
    # (approximates DAS's recency down-weighting with raw-count trees).
    fresh_iterations: int = 2
    max_tree_depth: int = 24
    # Draft-length scaling vs match length and confidence floor, used by the
    # ngram host (the suffix host takes these from vLLM's speculative_config).
    max_spec_factor: float = 2.0
    min_token_prob: float = 0.1
    # Engine delta-pump poll cadence; also ships intra-step cross-engine data.
    poll_interval_s: float = 2.0
    max_delta_batch_tokens: int = 500_000
    reset_on_weight_update: bool = False
    # vLLM excludes any request with logprobs from speculative decoding
    # (v0.11.0 is_spec_decode_unsupported), and verl's agent loop requests
    # them unconditionally. Safe to strip when rollout logprobs are unused
    # (verl's default: the trainer recomputes old log-probs with the actor).
    # Set false if you train with actor_rollout_ref.rollout.calculate_log_probs=True.
    strip_rollout_logprobs: bool = True
    # Occupancy gate: 0 (default) drafts every round — decode is memory-
    # bound at these batch sizes, so verification has slack all step.
    # Positive values confine drafting to the collapsed-batch tail, for
    # compute-bound regimes only. Drafting-only: delta applies are governed
    # by apply_on_poll below, never by this knob.
    tail_max_active: int = 0
    # Apply gate, decoupled from the drafting gate above. True: the delta
    # pump applies servable batches as soon as it fetches them — with
    # per-iteration serving they only become available at the step boundary,
    # so applies land during the rollout's prefill window regardless of
    # verl's engine sleep internals. False: applies happen exclusively in
    # the wake_up boundary drain (requires verl to drive vLLMHttpServer.wake_up).
    apply_on_poll: bool = True
    # Cap on the engine-local (request-scoped) arctic SuffixDecodingCache.
    # The stock 10k-request cap holds ~70M tokens of resident suffix
    # structures per worker at 16k generations; ~2k bounds it without
    # hurting the own-context/self-repetition hit rate (active requests are
    # never evicted, only completed ones).
    engine_cache_max_requests: int = 2000
    service_name: str = "pyis_das_suffix_service"
    service_namespace: str = "pyis"
    budgets: DASBudgetConfig = field(default_factory=DASBudgetConfig)
    classifier: DASClassifierConfig = field(default_factory=DASClassifierConfig)
    limits: DASServiceLimits = field(default_factory=DASServiceLimits)
    push: DASPushConfig = field(default_factory=DASPushConfig)


def _build(section: dict) -> DASConfig:
    budgets = section.get("budgets", {})
    classifier = section.get("classifier", {})
    limits = section.get("service", {})
    push = section.get("push", {})
    return DASConfig(
        enabled=bool(section.get("enabled", True)),
        window_iterations=int(section.get("window_iterations", 16)),
        fresh_iterations=int(section.get("fresh_iterations", 2)),
        max_tree_depth=int(section.get("max_tree_depth", 24)),
        max_spec_factor=float(section.get("max_spec_factor", 2.0)),
        min_token_prob=float(section.get("min_token_prob", 0.1)),
        poll_interval_s=float(section.get("poll_interval_s", 2.0)),
        max_delta_batch_tokens=int(section.get("max_delta_batch_tokens", 500_000)),
        reset_on_weight_update=bool(section.get("reset_on_weight_update")),
        strip_rollout_logprobs=bool(section.get("strip_rollout_logprobs", True)),
        tail_max_active=int(section.get("tail_max_active", 0)),
        apply_on_poll=bool(section.get("apply_on_poll", True)),
        engine_cache_max_requests=int(section.get("engine_cache_max_requests", 2000)),
        service_name=str(section.get("service_name", "pyis_das_suffix_service")),
        service_namespace=str(section.get("service_namespace", "pyis")),
        budgets=DASBudgetConfig(
            long=int(budgets.get("long", 24)),
            medium=int(budgets.get("medium", 8)),
            short=int(budgets.get("short", 0)),
        ),
        classifier=DASClassifierConfig(
            short_max_tokens=int(classifier.get("short_max_tokens", 256)),
            long_min_tokens=int(classifier.get("long_min_tokens", 1024)),
        ),
        limits=DASServiceLimits(
            max_total_tokens=int(limits.get("max_total_tokens", 64_000_000)),
            max_problems=int(limits.get("max_problems", 8192)),
            max_seqs_per_problem=int(limits.get("max_seqs_per_problem", 512)),
        ),
        push=DASPushConfig(
            batch_size=int(push.get("batch_size", 64)),
            flush_interval_s=float(push.get("flush_interval_s", 0.5)),
            queue_capacity=int(push.get("queue_capacity", 4096)),
        ),
    )


def load_das_config(path: str | None = None) -> DASConfig:
    """Load config from *path*, ``DAS_CONFIG_PATH``, or defaults.

    Never raises: any load problem logs a warning and returns defaults so an
    engine process can always come up (DAS degrades, the engine does not).
    """
    path = path or os.environ.get(CONFIG_PATH_ENV_VAR)
    if not path:
        return DASConfig()
    try:
        import yaml

        with open(path, encoding="utf-8") as f:  # noqa: PTH123
            raw = yaml.safe_load(f) or {}
        section = raw.get("das", raw)
        if not isinstance(section, dict):
            logger.warning(
                "DAS config %s: expected a mapping under 'das', got %s; using defaults",
                path,
                type(section).__name__,
            )
            return DASConfig()
        return _build(section)
    except Exception as e:  # noqa: BLE001
        logger.warning("DAS config load failed for %s (%s); using defaults", path, e)
        return DASConfig()
