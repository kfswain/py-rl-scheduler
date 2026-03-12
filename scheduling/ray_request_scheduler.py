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

import os
import yaml
from typing import Sequence, Optional
from scheduling.scheduler_config import SchedulerConfig
from scheduling.scheduler import Scheduler
from scheduling.types import LLMRequest, Endpoint, ScoredEndpoint, CycleState
from scheduling.framework import PickerPlugin
from scheduling import PrefixCacheScorer
from scheduling.prefix_plugin import _hash_prompt_bytes, _get_user_input_bytes

class RayRequestScheduler:
    def __init__(self):
        self.config_path = os.environ.get("ROUTER_CONFIG_PATH")
        if not self.config_path:
            raise ValueError("ROUTER_CONFIG_PATH environment variable is missing. Ensure the ConfigMap is mounted.")

        self.last_mtime = 0
        self._maybe_reload_config()

    def _maybe_reload_config(self):
        mtime = os.path.getmtime(self.config_path)
        if mtime > self.last_mtime:
            with open(self.config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise ValueError("Parsed configuration is not a valid dictionary.")
            self.config = SchedulerConfig.from_dict(config_dict)
            self.scheduler = Scheduler.new_with_config(self.config)
            self.last_mtime = mtime

    def run(self, request: LLMRequest, candidates: Sequence[Endpoint]) -> Sequence[ScoredEndpoint]:
        scheduler_output = self.scheduler.schedule(request, candidates)
        profile_name = scheduler_output.primary_profile_name
        profile_results = scheduler_output.profile_results.get(profile_name)

        print(f"Profile {profile_name} results: {profile_results}")
        selected_endpoint = profile_results.endpoint_list[:1]

        if len(selected_endpoint) > 0:
            self.pre_request(request, selected_endpoint[0].endpoint, profile_name)
            return selected_endpoint  # pick top 1
        print("No endpoint selected, defaulting to first candidate")
        return []

    def pre_request(self, request: LLMRequest, selected_endpoint: Endpoint, profile_name: str):
        print(selected_endpoint)
        for scorer_config in self.scheduler.profiles[profile_name].scorers:
            scorer = scorer_config.scorer
            if isinstance(scorer, PrefixCacheScorer) and request.body is not None:
                scorer.add_prefixes_for_server(selected_endpoint.name, _hash_prompt_bytes(request.target_model, _get_user_input_bytes(request.body), 64, 256))

class MaxScorePicker(PickerPlugin):
    def pick(self, cycle_state: CycleState, request: LLMRequest, scored_endpoints: Sequence[ScoredEndpoint]) -> Optional[ScoredEndpoint]:
        if not scored_endpoints:
            return None
        # pick the endpoint with the highest score
        return max(scored_endpoints, key=lambda se: se.score)
