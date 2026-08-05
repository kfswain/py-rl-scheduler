"""DASDrafterState + PySuffixTree, exercised end-to-end against the service."""

from py_inference_scheduler.speculative.config import (
    DASBudgetConfig,
    DASClassifierConfig,
    DASConfig,
)
from py_inference_scheduler.speculative.contracts import (
    TRANSIENT_BASE,
    DeltaBatch,
    TrajectoryDelta,
    TrajectoryPush,
    serialize_delta_batch,
)
from py_inference_scheduler.speculative.drafter_state import DASDrafterState, PySuffixTree
from py_inference_scheduler.speculative.service import SuffixDataService


def make_state(**cfg_overrides):
    cfg = DASConfig(
        max_tree_depth=cfg_overrides.pop("max_tree_depth", 24),
        budgets=cfg_overrides.pop("budgets", DASBudgetConfig(long=24, medium=8, short=0)),
        classifier=cfg_overrides.pop(
            "classifier", DASClassifierConfig(short_max_tokens=256, long_min_tokens=1024)
        ),
    )
    assert not cfg_overrides
    return DASDrafterState(cfg, tree_factory=PySuffixTree)


def batch_payload(**kwargs):
    defaults = {"from_version": -1, "to_version": 1, "iteration": 0}
    defaults.update(kwargs)
    return serialize_delta_batch(DeltaBatch(**defaults))


# ------------------------------------------------------------- PySuffixTree


def test_py_suffix_tree_drafts_continuation():
    tree = PySuffixTree(max_depth=16)
    tree.extend(1, [10, 20, 30, 40, 50, 60])
    draft = tree.speculate([10, 20, 30], max_spec_tokens=8, max_spec_factor=2.0)
    assert draft is not None
    assert draft.token_ids == [40, 50, 60]
    assert draft.match_len == 3


def test_py_suffix_tree_longest_match_wins():
    tree = PySuffixTree(max_depth=16)
    tree.extend(1, [1, 2, 3, 4])
    tree.extend(2, [9, 2, 3, 7])
    # Suffix [1,2,3] matches only seq 1 -> continuation 4, not 7.
    draft = tree.speculate([1, 2, 3], max_spec_tokens=4, max_spec_factor=2.0)
    assert draft.token_ids == [4]


def test_py_suffix_tree_spec_factor_caps_draft():
    tree = PySuffixTree(max_depth=16)
    tree.extend(1, [5, 1, 2, 3, 4, 5, 6, 7, 8])
    draft = tree.speculate([9, 9, 5], max_spec_tokens=8, max_spec_factor=1.0)
    assert draft is not None
    assert len(draft.token_ids) == 1  # match_len 1 * factor 1.0


def test_py_suffix_tree_min_prob_truncates():
    tree = PySuffixTree(max_depth=16)
    # After [1], continuations split 50/50 -> prob 0.5 each step.
    tree.extend(1, [1, 2, 3])
    tree.extend(2, [1, 4, 5])
    draft = tree.speculate([1], max_spec_tokens=8, max_spec_factor=8.0, min_token_prob=0.6)
    assert draft is None or draft.token_ids == []


def test_py_suffix_tree_remove():
    tree = PySuffixTree(max_depth=16)
    tree.extend(1, [1, 2, 3, 4])
    tree.remove(1)
    assert tree.speculate([1, 2, 3], max_spec_tokens=4) is None


# ------------------------------------------------------------ delta apply


def test_apply_adds_enables_drafting():
    state = make_state()
    payload = batch_payload(adds=[TrajectoryDelta("p1", 0, (1, 2, 3, 4, 5))])
    state.apply_delta_batch(payload)
    draft = state.speculate("p1", [1, 2, 3], budget=8, max_spec_factor=2.0)
    assert draft.token_ids == [4, 5]
    assert state.state_version() == 1


def test_apply_removals_and_drops():
    state = make_state()
    state.apply_delta_batch(
        batch_payload(
            adds=[TrajectoryDelta("p1", 0, (1, 2, 3, 4)), TrajectoryDelta("p2", 1, (5, 6, 7, 8))]
        )
    )
    state.apply_delta_batch(
        batch_payload(from_version=1, to_version=2, removals=[("p1", 0)], dropped_problems=["p2"])
    )
    assert state.speculate("p1", [1, 2, 3], budget=8) is None
    assert state.speculate("p2", [5, 6, 7], budget=8) is None
    assert state.tree_stats()["problem_trees"] == 1  # p1 tree survives empty


def test_removal_of_unknown_seq_is_benign():
    state = make_state()
    state.apply_delta_batch(batch_payload(removals=[("p1", 12345)]))
    assert state.state_version() == 1


def test_snapshot_clears_prior_state():
    state = make_state()
    state.apply_delta_batch(batch_payload(adds=[TrajectoryDelta("p1", 0, (1, 2, 3, 4))]))
    state.apply_delta_batch(
        batch_payload(
            from_version=-1,
            to_version=5,
            snapshot=True,
            adds=[TrajectoryDelta("p2", 2, (7, 8, 9, 10))],
        )
    )
    assert state.speculate("p1", [1, 2, 3], budget=8) is None
    assert state.speculate("p2", [7, 8, 9], budget=8).token_ids == [10]
    assert state.snapshots_applied == 1


def test_iteration_bump_drops_transients():
    state = make_state()
    state.apply_delta_batch(batch_payload(iteration=1))
    state.on_request_tokens("p1", "req-a", [1, 2, 3, 4, 5])
    assert state.speculate("p1", [1, 2, 3], budget=8, max_spec_factor=2.0).token_ids == [4, 5]
    assert state.tree_stats()["transient_requests"] == 1

    state.apply_delta_batch(batch_payload(from_version=1, to_version=2, iteration=2))
    assert state.tree_stats()["transient_requests"] == 0
    assert state.speculate("p1", [1, 2, 3], budget=8) is None


def test_same_iteration_keeps_transients():
    state = make_state()
    state.apply_delta_batch(batch_payload(iteration=1))
    state.on_request_tokens("p1", "req-a", [1, 2, 3, 4])
    state.apply_delta_batch(batch_payload(from_version=1, to_version=2, iteration=1))
    assert state.tree_stats()["transient_requests"] == 1


def test_transient_ids_use_reserved_namespace():
    state = make_state()
    state.on_request_tokens("p1", "req-a", [1, 2])
    state.on_request_tokens("p1", "req-b", [3, 4])
    tree = state._trees["p1"]
    assert all(seq_id >= TRANSIENT_BASE for seq_id in tree._seqs)


def test_drop_request_removes_transient():
    state = make_state()
    state.on_request_tokens("p1", "req-a", [1, 2, 3, 4])
    state.drop_request("req-a")
    assert state.speculate("p1", [1, 2, 3], budget=8) is None
    state.drop_request("req-a")  # idempotent


def test_multi_turn_transient_accumulates():
    state = make_state()
    state.on_request_tokens("p1", "req-a", [1, 2, 3])
    state.on_request_tokens("p1", "req-a", [4, 5])
    draft = state.speculate("p1", [1, 2, 3], budget=8, max_spec_factor=2.0)
    assert draft.token_ids == [4, 5]
    assert state.tree_stats()["transient_requests"] == 1


# ---------------------------------------------------------------- budgets


def test_budget_matrix():
    state = make_state(
        budgets=DASBudgetConfig(long=24, medium=8, short=0),
        classifier=DASClassifierConfig(short_max_tokens=256, long_min_tokens=1024),
    )
    state.apply_delta_batch(batch_payload(problem_cls={"pS": 0, "pM": 1, "pL": 2}))

    # Unknown problems default to Medium.
    assert state.budget_for(None, 0) == 8
    assert state.budget_for("unknown", 0) == 8
    # Class prior applies from token zero.
    assert state.budget_for("pS", 0) == 0
    assert state.budget_for("pM", 0) == 8
    assert state.budget_for("pL", 0) == 24
    # Runtime upgrade-only reclassification.
    assert state.budget_for("pS", 256) == 8
    assert state.budget_for("pS", 1024) == 24
    assert state.budget_for("pM", 1024) == 24
    # Long never downgrades.
    assert state.budget_for("pL", 10) == 24


# ------------------------------------------------- service -> state e2e


def test_service_to_state_end_to_end():
    svc = SuffixDataService(DASConfig(window_iterations=4, fresh_iterations=1))
    state = make_state()

    svc.begin_iteration(1)
    svc.add_trajectories(
        "w", [TrajectoryPush("p1", (10, 20, 30, 40, 50), prompt_len=2, server_id="s")]
    )
    version = -1
    while True:
        payload = svc.get_deltas("r1", version)
        if payload is None:
            break
        state.apply_delta_batch(payload)
        version = state.state_version()

    draft = state.speculate("p1", [10, 20, 30], budget=8, max_spec_factor=2.0)
    assert draft.token_ids == [40, 50]

    # Window rolls far enough that the sequence (and shadow) is evicted.
    svc.begin_iteration(10)
    while True:
        payload = svc.get_deltas("r1", version)
        if payload is None:
            break
        state.apply_delta_batch(payload)
        version = state.state_version()
    assert state.speculate("p1", [10, 20, 30], budget=8) is None
