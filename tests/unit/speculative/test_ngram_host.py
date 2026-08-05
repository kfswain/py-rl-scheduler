"""The ngram-host propose impl (vLLM 0.11.0 shim), driven duck-typed."""

from types import SimpleNamespace

import numpy as np

from py_inference_scheduler.speculative.config import DASBudgetConfig, DASConfig
from py_inference_scheduler.speculative.contracts import (
    DeltaBatch,
    TrajectoryDelta,
    serialize_delta_batch,
)
from py_inference_scheduler.speculative.drafter_state import DASDrafterState, PySuffixTree
from py_inference_scheduler.speculative.problem_id import encode_request_id, hash_problem_id
from py_inference_scheduler.speculative.vllm_proposer import _das_propose_ngram_impl

MOTIF = [100, 101, 102, 103, 104, 105]
PROMPT = [5, 6, 7, 8]


def make_state(**cfg_overrides):
    cfg = DASConfig(**cfg_overrides)
    return DASDrafterState(cfg, tree_factory=PySuffixTree)


def make_host(state, k=8, max_model_len=4096, cache=None):
    return SimpleNamespace(k=k, max_model_len=max_model_len, _das_cache=cache, _das_state=state)


def seed_problem(state, phash, cls=None):
    batch = DeltaBatch(
        from_version=-1,
        to_version=1,
        iteration=1,
        adds=[TrajectoryDelta(phash, 0, tuple(MOTIF))],
        problem_cls={} if cls is None else {phash: cls},
    )
    state.apply_delta_batch(serialize_delta_batch(batch))


def make_batch_args(req_ids, generated=3):
    n = len(req_ids)
    row = np.zeros(64, dtype=np.int64)
    row[: len(PROMPT)] = PROMPT
    row[len(PROMPT) : len(PROMPT) + generated] = MOTIF[:generated]
    num_tokens = np.array([len(PROMPT) + generated] * n)
    return num_tokens, np.stack([row] * n)


def test_das_request_drafts_from_problem_tree():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    host = make_host(state)
    req_ids = [encode_request_id(phash), "plain-request"]
    num_tokens, token_ids = make_batch_args(req_ids)

    drafts = _das_propose_ngram_impl(
        host, state, [[MOTIF[2]], [MOTIF[2]]], req_ids, num_tokens, token_ids, set()
    )
    assert len(drafts) == 2
    assert drafts[0][:3] == MOTIF[3:6]
    # No arctic cache and no problem id: plain request gets no draft.
    assert drafts[1] == []


def test_unsupported_and_empty_requests_skip():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    host = make_host(state)
    das_req = encode_request_id(phash)
    req_ids = [das_req, "other"]
    num_tokens, token_ids = make_batch_args(req_ids)

    drafts = _das_propose_ngram_impl(
        host, state, [[MOTIF[2]], []], req_ids, num_tokens, token_ids, {das_req}
    )
    assert drafts == [[], []]


def test_short_class_skips_drafting_but_collects():
    state = make_state(budgets=DASBudgetConfig(long=24, medium=8, short=0))
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash, cls=0)  # Short
    host = make_host(state)
    req_ids = [encode_request_id(phash)]
    num_tokens, token_ids = make_batch_args(req_ids)

    drafts = _das_propose_ngram_impl(
        host, state, [[MOTIF[2]]], req_ids, num_tokens, token_ids, set()
    )
    assert drafts == [[]]
    assert state.tree_stats()["transient_requests"] == 1  # data still collected


def test_baseline_fixed_at_first_sight():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    host = make_host(state)
    req = encode_request_id(phash)
    # First decode step after prefill: one generated token, one sampled id.
    num_tokens, token_ids = make_batch_args([req], generated=1)

    _das_propose_ngram_impl(host, state, [[MOTIF[0]]], [req], num_tokens, token_ids, set())
    baseline = state.note_request_start(req, 999, 1)  # second call: unchanged
    assert baseline == len(PROMPT)


def test_departed_requests_are_dropped():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    host = make_host(state)
    req = encode_request_id(phash)
    num_tokens, token_ids = make_batch_args([req])
    _das_propose_ngram_impl(host, state, [[MOTIF[2]]], [req], num_tokens, token_ids, set())
    assert state.tree_stats()["transient_requests"] == 1

    # Next step: the request is gone from the batch.
    other = encode_request_id(hash_problem_id([1, 2, 3]))
    num_tokens, token_ids = make_batch_args([other])
    _das_propose_ngram_impl(host, state, [[MOTIF[2]]], [other], num_tokens, token_ids, set())
    assert state.tree_stats()["transient_requests"] == 1  # only the new one


def test_budget_capped_by_host_k_and_model_len():
    state = make_state(budgets=DASBudgetConfig(long=24, medium=8, short=0))
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash, cls=2)  # Long: budget 24
    host = make_host(state, k=2)  # host cap wins
    req_ids = [encode_request_id(phash)]
    num_tokens, token_ids = make_batch_args(req_ids)
    drafts = _das_propose_ngram_impl(
        host, state, [[MOTIF[2]]], req_ids, num_tokens, token_ids, set()
    )
    assert len(drafts[0]) <= 2

    # At max_model_len: no draft at all.
    host = make_host(state, max_model_len=int(num_tokens[0]))
    drafts = _das_propose_ngram_impl(
        host, state, [[MOTIF[2]]], req_ids, num_tokens, token_ids, set()
    )
    assert drafts == [[]]
