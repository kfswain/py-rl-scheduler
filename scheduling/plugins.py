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

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Mapping
from .framework import FilterPlugin, ScorerPlugin, PickerPlugin, ProfileHandler, WeightedScorer, SchedulerProfile
from .types import Endpoint, ScoredEndpoint, CycleState, LLMRequest, ProfileRunResult
from .helpers import score_by_metric
from .registry import register_scorer, register_filter, register_picker, register_profile_handler
import random

@register_filter("simple")
class SimpleFilter:
    """A filter that keeps endpoints whose attribute `key` equals `value` (if provided)."""

    def __init__(self, key: str, value: Optional[Any] = None) -> None:
        self.key = key
        self.value = value

    def filter(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Sequence[Endpoint]:
        # endpoints may be a mapping name->Endpoint
        if isinstance(endpoints, Mapping):
            result: Dict[str, Endpoint] = {}
            for name, p in endpoints.items():
                v = p.attributes.get(self.key)
                if self.value is None or v == self.value:
                    result[name] = p
            return result

        out: List[Endpoint] = []
        for p in endpoints:
            v = p.attributes.get(self.key)
            if self.value is None or v == self.value:
                out.append(p)
        # convert sequence result into a name->Endpoint mapping for the framework
        return {p.name: p for p in out}


@register_scorer("constant")
class ConstantScorer(ScorerPlugin):
    """Scorer that returns a constant score (useful for tests)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[Endpoint, float]:
        # endpoints may be provided as a mapping name->Endpoint; normalize
        if isinstance(endpoints, Mapping):
            return {name: float(self.value) for name in endpoints.keys()}
        return {p.name: float(self.value) for p in endpoints}


@register_scorer("queue_length")
class QueueLengthScorer(ScorerPlugin):
    """Scores endpoints based on a 'waiting_queue_size' attribute.

    Lower queue size is better. The scorer returns a numeric score per endpoint
    keyed by endpoint name. By default the plugin expects integer sizes under
    the attribute key `waiting_queue_size` in the endpoint `attributes` map.

    Scoring function: score = - waiting_queue_size (so smaller queues get higher
    scores when sorted descending by score).
    """

    def __init__(self, attribute_key: str = "waiting_queue_size") -> None:
        self.attribute_key = attribute_key

    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        # endpoints may be mapping name->Endpoint
        if isinstance(endpoints, Mapping):
            result: Dict[str, float] = {}
            for name, ep in endpoints.items():
                raw = ep.attributes.get(self.attribute_key, 0)
                try:
                    size = int(raw)
                except Exception:
                    size = 0
                result[name] = float(-size)
            return result

        # sequence fallback
        return {ep.name: float(-int(ep.attributes.get(self.attribute_key, 0))) for ep in endpoints}


@register_scorer("least_queue")
class LeastQueueScorer(ScorerPlugin):
    """Scores endpoints based on their real-time Ray Serve actor queue length.

    The router injects 'queue_len' from its internal ReplicaQueueLengthCache 
    into the Endpoint attributes. This metric tracks the total number of 
    ongoing requests at the replica (both executing and in the mailbox queue).
    """

    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("queue_len", 0)),
            lower_is_better=True
        )

@register_scorer("waiting_queue")
class WaitingQueueScorer(ScorerPlugin):
    """Scores candidate endpoints based on the number of waiting requests inside the vLLM engine.
    Mirrors the EPP Gateway QueueScorer.
    Score = (max_waiting - current_waiting) / (max_waiting - min_waiting).
    Fewer waiting requests yields a higher score.
    """
    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("routing_stats", {}).get("num_waiting_reqs", 0)),
            lower_is_better=True
        )


@register_scorer("running_queue")
class RunningQueueScorer(ScorerPlugin):
    """Scores candidate endpoints based on the number of running requests inside the vLLM engine.
    Fewer running requests yields a higher score.
    """
    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("routing_stats", {}).get("num_running_reqs", 0)),
            lower_is_better=True
        )


@register_scorer("kv_cache")
class KVCacheScorer(ScorerPlugin):
    """Scores candidate endpoints based on KV cache utilization.
    Mirrors the EPP Gateway KVCacheUtilizationScorer but uses relative scoring.
    Score = (max_usage - current_usage) / (max_usage - min_usage).
    Lower KV cache usage yields a higher score.
    """
    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("routing_stats", {}).get("kv", 0.0)),
            lower_is_better=True
        )


@register_picker("random")
class RandomPicker:
    def __init__(self, max_num: int = 1) -> None:
        self.max_num = max_num

    def pick(self, cycle_state: CycleState, request: LLMRequest, scored_endpoints: Sequence[ScoredEndpoint]) -> Optional[ScoredEndpoint]:
        if not scored_endpoints:
            return None
        # sample from the top `max_num` if specified
        top = scored_endpoints[: self.max_num]
        return random.choice(top)


@register_picker("max_score")
class MaxScorePicker:
    def __init__(self, max_num: int = 1) -> None:
        self.max_num = max_num

    def pick(self, cycle_state: CycleState, request: LLMRequest, scored_endpoints: Sequence[ScoredEndpoint]) -> Optional[ScoredEndpoint]:
        if not scored_endpoints:
            return None
        # sample from the top `max_num` if specified, the
        top = scored_endpoints[: self.max_num]
        return top[0]


@register_profile_handler("single_profile")
class SingleProfileHandler:
    """Simple profile handler that runs all profiles and returns the first as primary."""

    def pick(self, cycle_state: CycleState, request: LLMRequest, profiles: Dict[str, SchedulerProfile], profile_results: Dict[str, Optional[ProfileRunResult]]) -> Dict[str, SchedulerProfile]:
        # run all profiles once
        return profiles.copy()

    def process_results(self, cycle_state: CycleState, request: LLMRequest, profile_results: Dict[str, Optional[ProfileRunResult]]) -> Optional[str]:
        # pick the first profile that has a non-empty result, otherwise None
        for name, res in profile_results.items():
            if res is not None and res.endpoint_list:
                return name
        # fallback: return first profile name if any
        if profile_results:
            return next(iter(profile_results))
        return None
