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
"""Engine-local DAS drafter state: per-problem suffix trees + budgets.

Lives inside every vLLM GPU worker (one per TP rank). Mutated only by
``apply_delta_batch`` (canonical data from the central service, delivered
via collective_rpc so all ranks stay byte-identical) and by transient
self-inserts of the engine's own sampled tokens, which are dropped wholesale
whenever a delta batch advances the iteration.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Callable

from py_inference_scheduler.speculative.config import CLS_MEDIUM, DASConfig
from py_inference_scheduler.speculative.contracts import (
    SEQ_ID_LIMIT,
    TRANSIENT_BASE,
    deserialize_delta_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class DraftResult:
    token_ids: list[int]
    score: float
    match_len: int


class PySuffixTree:
    """Pure-Python frequency-greedy suffix drafter.

    Semantically mirrors arctic's SuffixTree (longest-suffix match, greedy
    most-frequent continuation, frequency-ratio probabilities, the
    ``max_spec_factor * match_len`` cap) with O(corpus) scans. Used for unit
    tests and as an emergency fallback when arctic-inference is missing.
    """

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self._seqs: dict[int, list[int]] = {}

    def extend(self, seq_id: int, token_ids) -> None:
        self._seqs.setdefault(seq_id, []).extend(int(t) for t in token_ids)

    def remove(self, seq_id: int) -> None:
        self._seqs.pop(seq_id, None)

    def speculate(
        self,
        pattern,
        max_spec_tokens: int,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
    ) -> DraftResult | None:
        pattern = [int(t) for t in pattern][-self.max_depth :]
        for match_len in range(len(pattern), 0, -1):
            suffix = pattern[-match_len:]
            cursors = self._match_positions(suffix)
            if not cursors:
                continue
            cap = min(max_spec_tokens, int(match_len * max_spec_factor + max_spec_offset + 1e-6))
            draft = self._greedy_draft(cursors, cap, min_token_prob)
            if draft.token_ids:
                draft.match_len = match_len
                return draft
        return None

    def _match_positions(self, suffix: list[int]) -> list[tuple[int, int]]:
        n = len(suffix)
        out: list[tuple[int, int]] = []
        for seq_id, seq in self._seqs.items():
            out.extend(
                (seq_id, pos + n)
                for pos in range(len(seq) - n + 1)
                if seq[pos : pos + n] == suffix
            )
        return out

    def _greedy_draft(
        self, cursors: list[tuple[int, int]], cap: int, min_token_prob: float
    ) -> DraftResult:
        tokens: list[int] = []
        prob = 1.0
        score = 0.0
        while len(tokens) < cap:
            counts: dict[int, int] = {}
            total = 0
            for seq_id, pos in cursors:
                seq = self._seqs[seq_id]
                if pos < len(seq):
                    counts[seq[pos]] = counts.get(seq[pos], 0) + 1
                    total += 1
            if not counts:
                break
            best_tok = max(counts, key=lambda t: counts[t])
            prob *= counts[best_tok] / total
            if prob < min_token_prob:
                break
            tokens.append(best_tok)
            score += prob
            cursors = [(s, p + 1) for s, p in cursors if self._seqs[s][p : p + 1] == [best_tok]]
        return DraftResult(tokens, score, 0)


class ArcticSuffixTree:
    """Thin adapter over arctic-inference's C++ SuffixTree."""

    def __init__(self, max_depth: int) -> None:
        try:
            from arctic_inference.suffix_decoding._C import (  # type: ignore[import-not-found]
                SuffixTree,
            )
        except ImportError:
            from arctic_inference.suffix_decoding.cache import (  # type: ignore[import-not-found]
                SuffixTree,
            )

        self.max_depth = max_depth
        self._tree = SuffixTree(max_depth)

    def extend(self, seq_id: int, token_ids) -> None:
        self._tree.extend(seq_id, [int(t) for t in token_ids])

    def remove(self, seq_id: int) -> None:
        self._tree.remove(seq_id)

    def speculate(
        self,
        pattern,
        max_spec_tokens: int,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
    ) -> DraftResult | None:
        pattern = [int(t) for t in pattern][-self.max_depth :]
        if not pattern:
            return None
        # The nanobind binding is positional-only:
        # speculate(seq, max_spec_tokens, max_spec_factor, max_spec_offset,
        #           min_token_prob, use_tree_spec)
        draft = self._tree.speculate(
            pattern,
            int(max_spec_tokens),
            float(max_spec_factor),
            float(max_spec_offset),
            float(min_token_prob),
            False,  # noqa: FBT003 - use_tree_spec; binding is positional-only
        )
        token_ids = list(getattr(draft, "token_ids", []) or [])
        if not token_ids:
            return None
        return DraftResult(
            token_ids,
            float(getattr(draft, "score", 0.0)),
            int(getattr(draft, "match_len", 0)),
        )


def default_tree_factory(max_depth: int) -> ArcticSuffixTree | PySuffixTree:
    try:
        return ArcticSuffixTree(max_depth)
    except Exception as e:  # noqa: BLE001
        logger.warning("arctic-inference unavailable (%s); using PySuffixTree fallback", e)
        return PySuffixTree(max_depth)


class BudgetPolicy:
    """Length-aware Long/Medium/Short budgets with upgrade-only runtime reclass."""

    def __init__(self, cfg: DASConfig) -> None:
        self._budgets = cfg.budgets
        self._classifier = cfg.classifier

    def budget_for(self, problem_cls: int | None, observed_gen_len: int) -> int:
        # Unknown problems default to Medium so cold problems still speculate.
        base = CLS_MEDIUM if problem_cls is None else problem_cls
        runtime = self._classifier.class_for_length(float(observed_gen_len))
        return self._budgets.for_class(max(base, runtime))


class DASDrafterState:
    def __init__(
        self,
        cfg: DASConfig | None = None,
        tree_factory: Callable[[int], object] | None = None,
    ) -> None:
        self.cfg = cfg or DASConfig()
        self._tree_factory = tree_factory or default_tree_factory
        self.budget_policy = BudgetPolicy(self.cfg)
        self._trees: dict[str, object] = {}
        self._cls: dict[str, int] = {}
        self._transients: dict[str, tuple[str, int]] = {}  # req_id -> (phash, seq_id)
        # req_id -> first-seen token count minus that step's sampled tokens
        # (~= prompt length); the ngram host has no prompt boundary of its own.
        self._req_baselines: dict[str, int] = {}
        self._next_transient = TRANSIENT_BASE
        self._iteration = -1
        self._version = -1
        self.applied_batches = 0
        self.snapshots_applied = 0
        # Occupancy + attribution counters (set by the patched propose).
        self.last_active_count = -1
        self.rounds_seen = 0
        self.rounds_gated = 0
        self.drafts_emitted = 0

    # -------------------------------------------------------------- deltas

    def apply_delta_batch(self, payload: bytes) -> None:
        batch = deserialize_delta_batch(payload)
        if batch.snapshot or batch.reset:
            self._clear_all()
            self.snapshots_applied += 1
        if batch.iteration > self._iteration:
            self._drop_all_transients()
            self._iteration = batch.iteration
        for problem_id in batch.dropped_problems:
            self._trees.pop(problem_id, None)
            self._cls.pop(problem_id, None)
        for problem_id, seq_id in batch.removals:
            tree = self._trees.get(problem_id)
            if tree is not None:
                self._safe_remove(tree, seq_id)
        for delta in batch.adds:
            self._tree(delta.problem_id).extend(delta.seq_id, delta.token_ids)
        self._cls.update(batch.problem_cls)
        self._version = batch.to_version
        self.applied_batches += 1

    def _clear_all(self) -> None:
        self._trees.clear()
        self._cls.clear()
        self._transients.clear()
        self._req_baselines.clear()

    def _drop_all_transients(self) -> None:
        for phash, seq_id in self._transients.values():
            tree = self._trees.get(phash)
            if tree is not None:
                self._safe_remove(tree, seq_id)
        self._transients.clear()

    @staticmethod
    def _safe_remove(tree, seq_id: int) -> None:
        # Removal of an unknown seq is benign.
        with contextlib.suppress(Exception):
            tree.remove(seq_id)

    def _tree(self, problem_id: str):
        tree = self._trees.get(problem_id)
        if tree is None:
            tree = self._tree_factory(self.cfg.max_tree_depth)
            self._trees[problem_id] = tree
        return tree

    # ---------------------------------------------------------- transients

    def on_request_tokens(self, problem_id: str, req_id: str, token_ids) -> None:
        entry = self._transients.get(req_id)
        if entry is None:
            seq_id = self._next_transient
            self._next_transient = TRANSIENT_BASE + (
                (self._next_transient + 1 - TRANSIENT_BASE) % (SEQ_ID_LIMIT - TRANSIENT_BASE)
            )
            entry = (problem_id, seq_id)
            self._transients[req_id] = entry
        self._tree(entry[0]).extend(entry[1], token_ids)

    def drop_request(self, req_id: str) -> None:
        self._req_baselines.pop(req_id, None)
        entry = self._transients.pop(req_id, None)
        if entry is None:
            return
        tree = self._trees.get(entry[0])
        if tree is not None:
            self._safe_remove(tree, entry[1])

    def note_request_start(self, req_id: str, num_tokens: int, num_sampled: int) -> int:
        """Record (once) and return the request's prompt-length baseline."""
        baseline = self._req_baselines.get(req_id)
        if baseline is None:
            baseline = max(0, num_tokens - num_sampled)
            self._req_baselines[req_id] = baseline
        return baseline

    def drop_departed(self, active_req_ids) -> None:
        """Drop transients/baselines for requests no longer in the batch."""
        departed = (set(self._req_baselines) | set(self._transients)) - set(active_req_ids)
        for req_id in departed:
            self.drop_request(req_id)

    # -------------------------------------------------------------- drafts

    def speculate(
        self,
        problem_id: str,
        pattern,
        budget: int,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ) -> DraftResult | None:
        tree = self._trees.get(problem_id)
        if tree is None or budget <= 0:
            return None
        return tree.speculate(
            pattern,
            max_spec_tokens=budget,
            max_spec_factor=max_spec_factor,
            max_spec_offset=0.0,
            min_token_prob=min_token_prob,
        )

    def budget_for(self, problem_id: str | None, observed_gen_len: int) -> int:
        cls = self._cls.get(problem_id) if problem_id else None
        return self.budget_policy.budget_for(cls, observed_gen_len)

    # --------------------------------------------------------------- intro

    def state_version(self) -> int:
        return self._version

    def tree_stats(self) -> dict:
        return {
            "iteration": self._iteration,
            "version": self._version,
            "problem_trees": len(self._trees),
            "transient_requests": len(self._transients),
            "applied_batches": self.applied_batches,
            "snapshots_applied": self.snapshots_applied,
            "last_active_count": self.last_active_count,
            "rounds_seen": self.rounds_seen,
            "rounds_gated": self.rounds_gated,
            "drafts_emitted": self.drafts_emitted,
        }


# Module-level handle so the verl worker extension (which only has the vLLM
# Worker object) can reach the proposer's state without traversing private
# model-runner internals.
_ACTIVE_STATE: DASDrafterState | None = None


def register_active_state(state: DASDrafterState | None) -> None:
    global _ACTIVE_STATE  # noqa: PLW0603
    _ACTIVE_STATE = state


def get_active_state() -> DASDrafterState | None:
    return _ACTIVE_STATE
