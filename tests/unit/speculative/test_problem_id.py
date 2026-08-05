from py_inference_scheduler.speculative.problem_id import (
    PROBLEM_HASH_LEN,
    encode_request_id,
    hash_problem_id,
    parse_request_id,
)


def test_hash_is_stable_across_calls():
    prompt = [101, 42, 8, 99, 12345]
    assert hash_problem_id(prompt) == hash_problem_id(prompt)


def test_hash_matches_for_grpo_siblings():
    # Siblings share an identical first-turn prompt; iterables of any kind.
    prompt = list(range(500))
    assert hash_problem_id(prompt) == hash_problem_id(tuple(prompt))


def test_hash_differs_for_different_prompts():
    assert hash_problem_id([1, 2, 3]) != hash_problem_id([1, 2, 4])
    # Order matters.
    assert hash_problem_id([1, 2, 3]) != hash_problem_id([3, 2, 1])
    # Boundary shifts matter (no concatenation ambiguity).
    assert hash_problem_id([12, 3]) != hash_problem_id([1, 23])


def test_hash_model_salt():
    prompt = [5, 6, 7]
    assert hash_problem_id(prompt, model="a") != hash_problem_id(prompt, model="b")


def test_hash_length_and_charset():
    h = hash_problem_id([9, 8, 7])
    assert len(h) == PROBLEM_HASH_LEN
    assert all(c in "0123456789abcdef" for c in h)


def test_encode_parse_round_trip():
    phash = hash_problem_id([1, 2, 3])
    req_id = encode_request_id(phash)
    assert parse_request_id(req_id) == phash


def test_encode_produces_unique_ids():
    phash = hash_problem_id([1, 2, 3])
    assert encode_request_id(phash) != encode_request_id(phash)


def test_parse_tolerates_child_request_suffixes():
    phash = hash_problem_id([1, 2, 3])
    req_id = encode_request_id(phash)
    assert parse_request_id(req_id + "-0") == phash
    assert parse_request_id(req_id + "_child") == phash


def test_parse_rejects_hostile_ids():
    assert parse_request_id("plain-uuid-here") is None
    assert parse_request_id("dasp-nothex") is None
    assert parse_request_id("daspZZZZZZZZZZZZZZZZ-abc") is None
    assert parse_request_id("dasp0123456789abcde-abc") is None  # 15 hex, not 16
    assert parse_request_id("") is None
    assert parse_request_id(None) is None
    assert parse_request_id(12345) is None
