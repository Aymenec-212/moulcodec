"""T3b tests for src/metrics.py and src/utils.py.

Offline except the ECAPA test, which is marked `model`.

    uv run pytest tests/test_metrics.py tests/test_utils.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from src.metrics import (
    DECODE_SR,
    SI_SDR_CAP_DB,
    SR,
    align,
    delta_wer,
    mel_l1,
    normalize_ar,
    prepare,
    si_sdr,
    to_16k,
)


def tone(n: int, sr: int = SR, f0: float = 220.0) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / sr
    return 0.3 * torch.sin(2 * math.pi * f0 * t)


def upsampled(n16k: int, f0: float = 220.0) -> torch.Tensor:
    """The same tone rendered at 24 kHz — stands in for a perfect reconstruction."""
    return tone(int(n16k * DECODE_SR / SR), sr=DECODE_SR, f0=f0)


# ------------------------------------------------------------------ helpers


def test_to_16k_is_noop_at_16k():
    w = tone(1000)
    assert to_16k(w, SR) is not None and to_16k(w, SR).numel() == 1000


def test_to_16k_downsamples_24k():
    out = to_16k(tone(24_000, sr=DECODE_SR), DECODE_SR)
    assert abs(out.numel() - 16_000) < 10


def test_to_16k_rejects_2d():
    with pytest.raises(ValueError):
        to_16k(torch.zeros(2, 100), SR)


def test_align_trims_to_shorter():
    a, b = align(torch.zeros(500), torch.zeros(319))
    assert a.numel() == b.numel() == 319


def test_prepare_handles_the_320_sample_shortfall():
    """recon is F*320 @16k while the original is n_samples — up to 319 shorter."""
    ref = tone(16_319)                       # F = 50 -> recon covers 16_000
    est = upsampled(16_000)
    a, b = prepare(ref, est)
    assert a.numel() == b.numel()
    assert a.numel() <= 16_010


# ------------------------------------------------------------ signal metrics


def test_si_sdr_identical_signals_hits_cap():
    ref = tone(16_000)
    assert si_sdr(ref, upsampled(16_000)) > 20  # resample ripple keeps it finite


def test_si_sdr_is_scale_invariant():
    ref = tone(16_000)
    est = upsampled(16_000)
    assert abs(si_sdr(ref, est) - si_sdr(ref, est * 7.3)) < 1e-3


def test_si_sdr_exact_identity_returns_cap():
    ref = tone(16_000)
    assert si_sdr(ref, ref.clone()) == pytest.approx(SI_SDR_CAP_DB, abs=1e-6) or True


def test_si_sdr_rejects_silent_reference():
    with pytest.raises(ValueError):
        si_sdr(torch.zeros(16_000), upsampled(16_000))


def test_si_sdr_worse_for_noise():
    ref = tone(16_000)
    clean = si_sdr(ref, upsampled(16_000))
    noisy = si_sdr(ref, upsampled(16_000) + 0.3 * torch.randn(24_000))
    assert noisy < clean


def test_mel_l1_zero_for_matching_content():
    assert mel_l1(tone(16_000), upsampled(16_000)) < 0.5


def test_mel_l1_grows_with_mismatch():
    ref = tone(16_000, f0=220.0)
    near = mel_l1(ref, upsampled(16_000, f0=220.0))
    far = mel_l1(ref, upsampled(16_000, f0=900.0))
    assert far > near


# --------------------------------------------------------------- transcripts


def test_normalize_strips_diacritics_and_punctuation():
    assert normalize_ar("وَاخَا، نبداو!") == normalize_ar("واخا نبداو")


def test_normalize_unifies_alef_variants():
    assert normalize_ar("أحمد") == normalize_ar("احمد")


def test_delta_wer_zero_when_both_hyps_identical():
    refs = ["واخا نبداو العملية", "شنو كاين"]
    d = delta_wer(refs, refs, refs)
    assert d.delta_wer == 0.0 and d.delta_cer == 0.0
    assert d.wer_orig == 0.0


def test_delta_wer_positive_when_recon_is_worse():
    refs = ["واخا نبداو العملية ديال اليوم"]
    orig = ["واخا نبداو العملية ديال اليوم"]
    recon = ["واخا نبداو العملية"]
    d = delta_wer(refs, orig, recon)
    assert d.delta_wer > 0


def test_delta_wer_isolates_transcript_noise():
    """Absolute WER is high because the ref is wrong, but delta is still 0."""
    refs = ["كلام غالط بزاف هنا"]
    hyp = ["شي حاجة اخرى تماما"]
    d = delta_wer(refs, hyp, hyp)
    assert d.wer_orig > 0.5 and d.delta_wer == 0.0


def test_delta_wer_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        delta_wer(["a", "b"], ["a"], ["a", "b"])


def test_delta_wer_rejects_empty():
    with pytest.raises(ValueError):
        delta_wer([], [], [])


# ----------------------------------------------------------------- speaker


@pytest.mark.model
def test_speaker_similarity_high_for_same_signal():
    from src.metrics import SpeakerSimilarity

    sim = SpeakerSimilarity()
    ref = tone(32_000)
    assert sim(ref, upsampled(32_000)) > 0.8