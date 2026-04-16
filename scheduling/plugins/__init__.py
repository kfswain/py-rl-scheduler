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

# Import sub-modules to trigger registration side-effects
from .scorers import prefix_plugin as prefix_plugin, backpressure as backpressure, generic as generic_scorers  # noqa: F401
from .pickers import generic as generic_pickers  # noqa: F401
from .handlers import generic as generic_handlers  # noqa: F401

# Re-export key plugins if needed, but registry handles dynamic lookup
from .scorers.prefix_plugin import PrefixCacheScorer as PrefixCacheScorer
from .scorers.backpressure import (
    WaitingQueueScorer as WaitingQueueScorer,
    RunningQueueScorer as RunningQueueScorer,
    KVCacheScorer as KVCacheScorer,
    LeastQueueScorer as LeastQueueScorer,
    QueueLengthScorer as QueueLengthScorer,
)
from .scorers.generic import RoundRobinScorer as RoundRobinScorer, ConstantScorer as ConstantScorer
from .pickers.generic import RandomPicker as RandomPicker, MaxScorePicker as MaxScorePicker
from .handlers.generic import SingleProfileHandler as SingleProfileHandler, SimpleFilter as SimpleFilter
