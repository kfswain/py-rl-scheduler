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

from typing import Dict, Mapping
from .types import Endpoint, CycleState, LLMRequest
from .registry import register_scorer
import threading

@register_scorer("round_robin")
class RoundRobinScorer:
    """A scorer that cycles through endpoints in a round-robin fashion (NeMO equivalent).
    
    Internal counter that increments with each score call. The endpoint corresponding to 
    (counter % num_endpoints) receives a score of 1.0 to comply with weighted scoring.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def score(self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]) -> Dict[str, float]:
        if not pods:
            return {}

        # sorted keys ensure the idx of each pod is deterministic
        names = sorted(pods.keys(), key=str)
        with self._lock:
            idx = self._counter % len(names)
            self._counter += 1
        selected_name = names[idx]

        print(f"RoundRobinScorer: selected {selected_name} (index {idx})")
        return {selected_name: 1.0}
