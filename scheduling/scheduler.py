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

from typing import Dict, Optional, Sequence
from .scheduler_config import SchedulerConfig
from .types import LLMRequest, Endpoint, SchedulingResult, CycleState, ProfileRunResult
from .framework import ProfileHandler


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.profile_handler = config.profile_handler
        self.profiles = config.profiles

    @classmethod
    def new_with_config(cls, config: SchedulerConfig) -> "Scheduler":
        return cls(config)

    def schedule(self, request: LLMRequest, candidates: Sequence[Endpoint]) -> SchedulingResult:
        if not candidates:
            raise ValueError("no scheduling candidates provided")

        cycle_state = CycleState()
        profile_results: Dict[str, Optional[ProfileRunResult]] = {}

        # ask profile handler which profiles to run
        selected = self.profile_handler.pick(cycle_state, request, self.profiles, profile_results)
        assert selected is not None

        
        for name, profile in selected.items():
            try:
                res = profile.run(request, cycle_state, candidates)
                profile_results[name] = res
            except Exception as e:
                print(f"Error running profile {name}: ")
                print(repr(e))
                profile_results[name] = None

        
        primary = self.profile_handler.process_results(cycle_state, request, profile_results)

        # Build SchedulingResult
        result = SchedulingResult(profile_results={k: v or ProfileRunResult() for k, v in profile_results.items()}, primary_profile_name=primary)
        # todo: this isnt exactly true with multiple profiles
        # we should update this to be more naunced logging in the future
        return result
