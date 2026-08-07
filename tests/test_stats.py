"""Tests for src/stats.py. Offline (jiwer only).

    uv run pytest tests/test_stats.py -v
"""

from __future__ import annotations

import pytest

from src.stats import (
    EXPAND,
    NO_TRIGGER,
    TRIGGER,
    ClipCounts,
    bootstrap_ci,
    clip_counts,
    corpus_rates,
    decide,
    delta_wer_points,
    mean,
)


def cc(aid="a", we_o=0, we_r=0, wref=10, ce_o=0, ce_r=0, cref=50) -> ClipCounts:
    return ClipCounts(aid, we_o, we_r, wref, ce_o, ce_r, cref)


def test_clip_counts_identical_is_zero_error():
    c = clip_counts("x", "واخا نبداو العملية", "واخا نبداو العملية", "واخا نبداو العملية")
    assert c.w_err_orig == 0 and c.w_err_recon == 0
    assert c.w_ref == 3


def test_clip_counts_detects_recon_only_damage():
    c = clip_counts("x", "واخا نبداو العملية", "واخا نبداو العملية", "واخا نبداو")
    assert c.w_err_orig == 0 and c.w_err_recon == 1


def test_clip_counts_normalisation_applied_to_all_sides():
    ref = "وَاخَا، نبداو!"
    c = clip_counts("x", ref, "واخا نبداو", "واخا نبداو")
    assert c.w_err_orig == 0, "normalisation must make punctuation/diacritics free"


def test_corpus_wer_weights_by_words_not_clips():
    """A long clip must not count the same as a short one."""
    short = cc("s", we_o=1, we_r=1, wref=1)      # 100% WER, 1 word
    long = cc("l", we_o=0, we_r=0, wref=99)      # 0% WER, 99 words
    r = corpus_rates([short, long])
    assert r["wer_orig"] == pytest.approx(1.0)   # 1/100, not the 50% a clip-mean gives


def test_delta_is_difference_of_corpus_rates():
    r = corpus_rates([cc(we_o=1, we_r=3, wref=10)])
    assert r["delta_wer"] == pytest.approx(20.0)


def test_delta_zero_when_recon_matches_orig():
    assert delta_wer_points([cc(we_o=4, we_r=4, wref=20)]) == 0.0


def test_corpus_rates_rejects_empty_reference():
    with pytest.raises(ValueError):
        corpus_rates([cc(wref=0, cref=0)])


def test_rates_are_percentage_points():
    r = corpus_rates([cc(we_o=5, we_r=5, wref=10)])
    assert r["wer_orig"] == pytest.approx(50.0)  # points, not fraction


def test_bootstrap_ci_brackets_the_point_estimate():
    counts = [cc(f"c{i}", we_o=1, we_r=2, wref=10) for i in range(50)]
    point = delta_wer_points(counts)
    lo, hi = bootstrap_ci(counts, delta_wer_points, iters=300, seed=1)
    assert lo <= point <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    counts = [cc(f"c{i}", we_o=i % 3, we_r=i % 4, wref=10) for i in range(40)]
    a = bootstrap_ci(counts, delta_wer_points, iters=200, seed=7)
    b = bootstrap_ci(counts, delta_wer_points, iters=200, seed=7)
    assert a == b


def test_bootstrap_ci_narrows_with_more_clips():
    def width(n):
        counts = [cc(f"c{i}", we_o=i % 3, we_r=(i % 3) + 1, wref=10 + (i % 5)) for i in range(n)]
        lo, hi = bootstrap_ci(counts, delta_wer_points, iters=400, seed=3)
        return hi - lo

    assert width(400) < width(40)


def test_bootstrap_rejects_single_item():
    with pytest.raises(ValueError):
        bootstrap_ci([cc()], delta_wer_points, iters=10)


def test_mean_ignores_none():
    assert mean([1.0, None, 3.0]) == pytest.approx(2.0)


def test_verdict_no_trigger_when_both_clearly_pass():
    v, _ = decide((0.5, 2.0), (0.90, 0.95))
    assert v == NO_TRIGGER


def test_verdict_trigger_when_wer_ci_above_threshold():
    v, _ = decide((6.0, 9.0), (0.90, 0.95))
    assert v == TRIGGER


def test_verdict_trigger_when_speaker_ci_below_threshold():
    v, _ = decide((0.5, 2.0), (0.60, 0.75))
    assert v == TRIGGER


def test_verdict_expand_when_wer_straddles():
    v, reasons = decide((3.0, 7.0), (0.90, 0.95))
    assert v == EXPAND and any("straddles" in r for r in reasons)


def test_verdict_expand_when_speaker_straddles():
    v, _ = decide((0.5, 2.0), (0.78, 0.83))
    assert v == EXPAND


def test_failure_beats_ambiguity():
    """A clear failure on one axis triggers even if the other is ambiguous."""
    v, _ = decide((6.0, 9.0), (0.78, 0.83))
    assert v == TRIGGER


def test_missing_speaker_ci_falls_back_to_wer_only():
    assert decide((0.5, 2.0), None)[0] == NO_TRIGGER
    assert decide((3.0, 7.0), None)[0] == EXPAND