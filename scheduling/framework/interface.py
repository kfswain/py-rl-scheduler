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
from typing import Dict, List, Optional, Protocol, Sequence, Mapping
from .types import Endpoint, ScoredEndpoint, CycleState, LLMRequest, ProfileRunResult
from .._scheduling import (
    WeightedScorer as WeightedScorer,
    SchedulerProfile as SchedulerProfile,
)


class FilterPlugin(Protocol):
    def filter(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> Mapping[str, Endpoint]: ...


class ScorerPlugin(Protocol):
    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> Dict[str, float]: ...


class PickerPlugin(Protocol):
    def pick(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        scored_pods: Sequence[ScoredEndpoint],
    ) -> Optional[ScoredEndpoint]: ...


class ProfileHandler(Protocol):
    def pick(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        profiles: Dict[str, "SchedulerProfile"],
        profile_results: Dict[str, Optional[ProfileRunResult]],
    ) -> Dict[str, "SchedulerProfile"]: ...

    def process_results(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        profile_results: Dict[str, Optional[ProfileRunResult]],
    ) -> Optional[str]: ...
