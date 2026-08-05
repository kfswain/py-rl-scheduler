import pytest

from py_inference_scheduler.speculative.contracts import (
    SEQ_ID_LIMIT,
    SHADOW_OFFSET,
    TRANSIENT_BASE,
    DeltaBatch,
    TrajectoryDelta,
    deserialize_delta_batch,
    serialize_delta_batch,
)


def test_round_trip():
    batch = DeltaBatch(
        from_version=3,
        to_version=9,
        iteration=4,
        snapshot=False,
        adds=[TrajectoryDelta("p1", 7, (1, 2, 3))],
        removals=[("p2", 5)],
        dropped_problems=["p3"],
        problem_cls={"p1": 2},
    )
    out = deserialize_delta_batch(serialize_delta_batch(batch))
    assert out.from_version == 3
    assert out.to_version == 9
    assert out.iteration == 4
    assert out.adds == [TrajectoryDelta("p1", 7, (1, 2, 3))]
    assert out.removals == [("p2", 5)]
    assert out.dropped_problems == ["p3"]
    assert out.problem_cls == {"p1": 2}


def test_schema_version_mismatch_raises():
    payload = serialize_delta_batch(DeltaBatch(0, 1, 0))
    bad = bytes([payload[0] + 1]) + payload[1:]
    with pytest.raises(ValueError, match="schema mismatch"):
        deserialize_delta_batch(bad)


def test_empty_payload_raises():
    with pytest.raises(ValueError, match="empty"):
        deserialize_delta_batch(b"")


def test_non_batch_payload_raises():
    import pickle  # noqa: S403

    payload = bytes([1]) + pickle.dumps({"not": "a batch"})
    with pytest.raises(TypeError):
        deserialize_delta_batch(payload)


def test_seq_id_namespaces_are_disjoint():
    # Canonical [0, SHADOW_OFFSET), shadows [SHADOW_OFFSET, 2*SHADOW_OFFSET),
    # transients [TRANSIENT_BASE, SEQ_ID_LIMIT).
    assert 2 * SHADOW_OFFSET <= TRANSIENT_BASE
    assert TRANSIENT_BASE < SEQ_ID_LIMIT
