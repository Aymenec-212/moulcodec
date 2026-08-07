"""Tests for src/sampling.py. Fully offline.

    uv run pytest tests/test_sampling.py -v
"""

from __future__ import annotations

import random

import pytest

from src.sampling import (
    duration_bucket,
    select_audit_sample,
    stratified_order,
    stratum_counts,
)


def corpus(n: int = 500, seed: int = 0):
    rng = random.Random(seed)
    channels = [f"ch{rng.randrange(8)}" for _ in range(n)]
    durations = [rng.uniform(1.2, 12.0) for _ in range(n)]
    ids = [f"clip_{i:05d}" for i in range(n)]
    return ids, durations, channels


# ------------------------------------------------------------------ buckets


def test_duration_bucket_boundaries():
    assert duration_bucket(1.9) == 0
    assert duration_bucket(2.0) == 1      # edges are exclusive lower bounds
    assert duration_bucket(11.9) == 5


# ------------------------------------------------------------- the invariant


@pytest.mark.parametrize("n1,n2", [(10, 50), (50, 200), (200, 500)])
def test_sample_is_nested(n1, n2):
    """THE load-bearing property: expanding must reuse everything already done."""
    ids, dur, ch = corpus()
    small = select_audit_sample(ids, dur, ch, n1)
    large = select_audit_sample(ids, dur, ch, n2)
    assert small == large[:n1]
    assert set(small) <= set(large)


def test_full_ladder_is_nested():
    ids, dur, ch = corpus()
    prev: list[int] = []
    for n in (10, 100, 300, 500):
        cur = select_audit_sample(ids, dur, ch, n)
        assert cur[: len(prev)] == prev
        prev = cur


def test_order_is_deterministic_across_calls():
    ids, dur, ch = corpus()
    assert stratified_order(ids, dur, ch) == stratified_order(ids, dur, ch)


def test_different_seed_changes_order():
    ids, dur, ch = corpus()
    assert stratified_order(ids, dur, ch, seed=1) != stratified_order(ids, dur, ch, seed=2)


def test_order_is_a_permutation():
    ids, dur, ch = corpus(300)
    order = stratified_order(ids, dur, ch)
    assert sorted(order) == list(range(300))


def test_order_independent_of_input_row_order():
    """Shuffling the corpus must not change WHICH clips get picked."""
    ids, dur, ch = corpus(200)
    first = {ids[i] for i in select_audit_sample(ids, dur, ch, 60)}

    perm = list(range(200))
    random.Random(7).shuffle(perm)
    ids2 = [ids[i] for i in perm]
    dur2 = [dur[i] for i in perm]
    ch2 = [ch[i] for i in perm]
    second = {ids2[i] for i in select_audit_sample(ids2, dur2, ch2, 60)}

    assert first == second


# ------------------------------------------------------------------ balance


def test_sample_is_roughly_proportional_across_strata():
    ids, dur, ch = corpus(1000)
    n = 200
    idxs = select_audit_sample(ids, dur, ch, n)
    pop = stratum_counts(dur, ch)
    smp = stratum_counts(dur, ch, idxs)

    for stratum, pop_n in pop.items():
        if pop_n < 20:
            continue  # tiny strata round noisily; not worth asserting
        want = n * pop_n / len(ids)
        got = smp.get(stratum, 0)
        assert abs(got - want) <= max(2.0, 0.5 * want), f"{stratum}: {got} vs {want:.1f}"


def test_every_large_stratum_is_represented():
    ids, dur, ch = corpus(1000)
    idxs = select_audit_sample(ids, dur, ch, 200)
    smp = stratum_counts(dur, ch, idxs)
    for stratum, pop_n in stratum_counts(dur, ch).items():
        if pop_n >= 50:
            assert smp.get(stratum, 0) > 0, f"stratum {stratum} missing from sample"


# ------------------------------------------------------------------- edges


def test_n_larger_than_corpus_returns_everything():
    ids, dur, ch = corpus(30)
    assert len(select_audit_sample(ids, dur, ch, 999)) == 30


def test_zero_and_negative_return_empty():
    ids, dur, ch = corpus(30)
    assert select_audit_sample(ids, dur, ch, 0) == []
    assert select_audit_sample(ids, dur, ch, -5) == []


def test_empty_corpus():
    assert stratified_order([], [], []) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        stratified_order(["a", "b"], [1.0], ["c", "d"])


def test_single_item():
    assert select_audit_sample(["a"], [3.0], ["ch"], 1) == [0]