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
"""In-place patch of vLLM's stock SuffixDecodingProposer.

vLLM's ``SpeculativeMethod`` is a closed Literal, so DAS rides
``method="suffix"``: ``__init__`` and ``propose`` are rebound ON the stock
class object (surviving vLLM's ``isinstance`` assert and any import order).

The patched propose keeps stock bookkeeping/behavior byte-for-byte for
requests without a DAS problem id, and for known problems additionally:

- lets the per-problem tree (canonical cross-worker data) compete with the
  stock cache's draft by score,
- applies length-aware Long/Medium/Short budgets (Short skips drafting),
- transiently self-inserts sampled tokens so same-batch GRPO siblings reuse
  each other immediately, before the central copy arrives.

Any structural error (vLLM version drift) permanently reverts that worker to
the stock path — DAS degrades, decoding never breaks.
"""

from __future__ import annotations

import logging

from py_inference_scheduler.speculative.config import das_enabled, load_das_config
from py_inference_scheduler.speculative.drafter_state import (
    DASDrafterState,
    register_active_state,
)
from py_inference_scheduler.speculative.problem_id import parse_request_id

logger = logging.getLogger(__name__)


def das_patch_suffix_proposer() -> None:
    """Rebind SuffixDecodingProposer.__init__/.propose in place. Idempotent."""
    from vllm.v1.spec_decode.suffix_decoding import (  # type: ignore[import-not-found]
        SuffixDecodingProposer,
    )

    if getattr(SuffixDecodingProposer, "_das_patched", False):
        return

    orig_init = SuffixDecodingProposer.__init__
    orig_propose = SuffixDecodingProposer.propose

    def das_init(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        self._das_state = None
        self._das_failed = False
        if not das_enabled():
            return
        try:
            self._das_state = DASDrafterState(load_das_config())
            register_active_state(self._das_state)
            logger.info("DAS: suffix proposer initialized with per-problem drafter state")
        except Exception:
            logger.exception("DAS: drafter state init failed; running stock suffix decoding")
            self._das_state = None

    def das_propose(self, input_batch, sampled_token_ids, *args, **kwargs):
        state = getattr(self, "_das_state", None)
        if state is None:
            return orig_propose(self, input_batch, sampled_token_ids, *args, **kwargs)
        try:
            return _das_propose_impl(self, state, input_batch, sampled_token_ids)
        except Exception:
            if not getattr(self, "_das_failed", False):
                logger.exception(
                    "DAS: propose failed (likely vLLM drift); permanently reverting "
                    "this worker to stock suffix decoding"
                )
            self._das_failed = True
            self._das_state = None
            register_active_state(None)
            return orig_propose(self, input_batch, sampled_token_ids, *args, **kwargs)

    SuffixDecodingProposer.__init__ = das_init
    SuffixDecodingProposer.propose = das_propose
    SuffixDecodingProposer._das_patched = True
    logger.info("DAS: SuffixDecodingProposer patched in place")


def _das_propose_impl(self, state: DASDrafterState, input_batch, sampled_token_ids):  # noqa: PLR0912
    """Mirrors the stock propose loop with per-problem trees and budgets."""
    draft_token_ids = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        if not sampled_ids:
            # Partial prefill: no sampled tokens yet.
            draft_token_ids.append([])
            continue

        req_id = input_batch.req_ids[i]
        if req_id not in self.suffix_cache.active_requests:
            if req_id in self.suffix_cache.cached_requests:
                self.suffix_cache.evict_cached_response(req_id)
            num_prompt_tokens = input_batch.num_prompt_tokens[i]
            prompt_token_ids = input_batch.token_ids_cpu[i, :num_prompt_tokens]
            self.suffix_cache.start_request(req_id, prompt_token_ids)
        self.suffix_cache.add_active_response(req_id, sampled_ids)

        phash = parse_request_id(req_id)
        if phash is not None:
            # Transient self-insert first, so same-batch siblings see it.
            state.on_request_tokens(phash, req_id, sampled_ids)

        num_tokens = int(input_batch.num_tokens_no_spec[i])
        if num_tokens >= self.max_model_len:
            draft_token_ids.append([])
            continue

        if phash is None:
            budget = self.num_speculative_tokens
        else:
            observed = num_tokens - int(input_batch.num_prompt_tokens[i])
            budget = state.budget_for(phash, observed)
        budget = min(budget, self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
        if budget <= 0:
            # Short class: skip drafting entirely (the DAS "skip speculation"
            # rule); data collection above still happened.
            draft_token_ids.append([])
            continue

        start = max(0, num_tokens - self.max_tree_depth)
        pattern = input_batch.token_ids_cpu[i, start:num_tokens]
        best = self.suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=budget,
            max_spec_factor=self.max_spec_factor,
            min_token_prob=self.min_token_prob,
        )
        if phash is not None:
            problem_draft = state.speculate(
                phash,
                [int(t) for t in pattern],
                budget,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            if problem_draft is not None and (
                best is None or problem_draft.score >= float(getattr(best, "score", 0.0))
            ):
                best = problem_draft
        draft_token_ids.append(list(best.token_ids) if best is not None else [])

    # Requests that left the batch: stock cache cleanup + transient drop.
    active_ids = set(getattr(input_batch, "req_id_to_index", {}) or {})
    if not active_ids:
        active_ids = {rid for rid in input_batch.req_ids if rid is not None}
    for req_id in self.suffix_cache.active_requests - active_ids:
        self.suffix_cache.stop_request(req_id)
        state.drop_request(req_id)
    return draft_token_ids
