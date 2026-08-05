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
"""In-place patch of vLLM's stock speculative proposer, hosting DAS.

vLLM's ``SpeculativeMethod`` is a closed Literal, so DAS rides a stock
method by rebinding ``__init__``/``propose`` ON the stock class object
(surviving vLLM's ``isinstance`` asserts and any import order). Two hosts,
auto-selected by what the installed vLLM ships:

- **suffix host** (vLLM >= 0.11.1): ``method="suffix"``'s
  SuffixDecodingProposer. Stock bookkeeping kept byte-for-byte for
  non-DAS requests; per-problem trees compete with the stock cache.
- **ngram host** (vLLM 0.11.0, no suffix module): ``method="ngram"``'s
  NgramProposer. DAS replaces prompt-lookup drafting wholesale with
  suffix-decoding behavior (arctic SuffixDecodingCache when available +
  per-problem trees), since 0.11.0 has no suffix cache of its own.

Both hosts add the DAS layer: per-problem trees fed by central deltas,
length-aware Long/Medium/Short budgets (Short skips drafting), and
transient self-inserts so same-batch GRPO siblings reuse each other
immediately. Any structural error (vLLM version drift) permanently reverts
that worker to the stock path — DAS degrades, decoding never breaks.
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


def das_patch_proposer() -> None:
    """Patch whichever host proposer the installed vLLM provides."""
    try:
        das_patch_suffix_proposer()
    except ImportError:
        logger.info("DAS: vLLM has no suffix decoding; falling back to the ngram host")
        das_patch_ngram_proposer()


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


# --------------------------------------------------------------- ngram host


def das_patch_ngram_proposer() -> None:
    """Rebind NgramProposer.__init__/.propose in place. Idempotent.

    Host for vLLM builds without suffix decoding (0.11.0): verified call
    contract there is positional —
    propose(sampled_token_ids, req_ids, num_tokens_no_spec, token_ids_cpu,
    spec_decode_unsupported_reqs) — with an isinstance(NgramProposer)
    assert at the call site.
    """
    from vllm.v1.spec_decode.ngram_proposer import (  # type: ignore[import-not-found]
        NgramProposer,
    )

    if getattr(NgramProposer, "_das_patched", False):
        return

    orig_init = NgramProposer.__init__
    orig_propose = NgramProposer.propose

    def das_init(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        self._das_state = None
        self._das_cache = None
        self._das_failed = False
        if not das_enabled():
            return
        try:
            cfg = load_das_config()
            self._das_state = DASDrafterState(cfg)
            self._das_cache = _build_suffix_cache(cfg.max_tree_depth)
            register_active_state(self._das_state)
            logger.info(
                "DAS: ngram proposer hosting DAS (suffix cache: %s)",
                "arctic" if self._das_cache is not None else "unavailable",
            )
        except Exception:
            logger.exception("DAS: ngram-host init failed; running stock ngram")
            self._das_state = None

    def das_propose(  # noqa: PLR0913,PLR0917 - mirrors 0.11.0's positional contract
        self,
        sampled_token_ids,
        req_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        spec_decode_unsupported_reqs,
        *args,
        **kwargs,
    ):
        state = getattr(self, "_das_state", None)
        if state is None:
            return orig_propose(
                self,
                sampled_token_ids,
                req_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                spec_decode_unsupported_reqs,
                *args,
                **kwargs,
            )
        try:
            return _das_propose_ngram_impl(
                self,
                state,
                sampled_token_ids,
                req_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                spec_decode_unsupported_reqs,
            )
        except Exception:
            if not getattr(self, "_das_failed", False):
                logger.exception(
                    "DAS: ngram-host propose failed (likely vLLM drift); permanently "
                    "reverting this worker to stock ngram"
                )
            self._das_failed = True
            self._das_state = None
            register_active_state(None)
            return orig_propose(
                self,
                sampled_token_ids,
                req_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                spec_decode_unsupported_reqs,
                *args,
                **kwargs,
            )

    NgramProposer.__init__ = das_init
    NgramProposer.propose = das_propose
    NgramProposer._das_patched = True
    logger.info("DAS: NgramProposer patched in place (ngram host)")


def _build_suffix_cache(max_tree_depth: int):
    """Arctic SuffixDecodingCache for the ngram host, or None without arctic."""
    try:
        from arctic_inference.suffix_decoding import (  # type: ignore[import-not-found]
            SuffixDecodingCache,
        )

        return SuffixDecodingCache(
            max_tree_depth=max_tree_depth, max_cached_requests=10_000
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("DAS: arctic SuffixDecodingCache unavailable (%s)", e)
        return None


def _das_propose_ngram_impl(  # noqa: C901, PLR0912, PLR0913, PLR0917
    self,
    state: DASDrafterState,
    sampled_token_ids,
    req_ids,
    num_tokens_no_spec,
    token_ids_cpu,
    spec_decode_unsupported_reqs,
):
    """Suffix-decoding drafting hosted in the ngram slot (vLLM 0.11.0).

    The ngram contract exposes no prompt boundary, so a request's first
    sighting fixes its baseline (num_tokens minus this step's sampled
    tokens ~= prompt length) for budgets and cache prompt trees.
    """
    cfg = state.cfg
    cache = getattr(self, "_das_cache", None)
    drafts = []
    active_ids = set()
    for i, sampled_ids in enumerate(sampled_token_ids):
        req_id = req_ids[i]
        if req_id is not None:
            active_ids.add(req_id)
        if not sampled_ids or req_id in spec_decode_unsupported_reqs:
            drafts.append([])
            continue

        num_tokens = int(num_tokens_no_spec[i])
        baseline = state.note_request_start(req_id, num_tokens, len(sampled_ids))
        phash = parse_request_id(req_id)

        if cache is not None:
            if req_id not in cache.active_requests:
                if req_id in cache.cached_requests:
                    cache.evict_cached_response(req_id)
                prompt_ids = [int(t) for t in token_ids_cpu[i, :baseline]]
                cache.start_request(req_id, prompt_ids)
            cache.add_active_response(req_id, [int(t) for t in sampled_ids])
        if phash is not None:
            state.on_request_tokens(phash, req_id, sampled_ids)

        if num_tokens >= self.max_model_len:
            drafts.append([])
            continue
        observed = max(0, num_tokens - baseline)
        budget = state.budget_for(phash, observed) if phash is not None else self.k
        budget = min(budget, self.k, self.max_model_len - num_tokens - 1)
        if budget <= 0:
            drafts.append([])
            continue

        start = max(0, num_tokens - cfg.max_tree_depth)
        pattern = [int(t) for t in token_ids_cpu[i, start:num_tokens]]
        best = None
        if cache is not None:
            candidate = cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=budget,
                max_spec_factor=cfg.max_spec_factor,
                min_token_prob=cfg.min_token_prob,
            )
            if candidate is not None and getattr(candidate, "token_ids", None):
                best = candidate
        if phash is not None:
            problem_draft = state.speculate(
                phash,
                pattern,
                budget,
                max_spec_factor=cfg.max_spec_factor,
                min_token_prob=cfg.min_token_prob,
            )
            if problem_draft is not None and (
                best is None or problem_draft.score >= float(getattr(best, "score", 0.0))
            ):
                best = problem_draft
        drafts.append(list(best.token_ids) if best is not None else [])

    state.drop_departed(active_ids)
    if cache is not None:
        for req_id in set(cache.active_requests) - active_ids:
            cache.stop_request(req_id)
    return drafts
