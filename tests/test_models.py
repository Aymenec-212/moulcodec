"""T3 tests for src/models.py.

These download real neucodec weights, so they are marked `model` and skipped
by default in CI. Run locally with:

    uv run pytest tests/test_models.py -m model -s
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models import (
    DECODE_SR,
    HOP,
    SAMPLES_PER_FRAME_24K,
    NeuCodecWrapper,
    frames_for_samples,
    pad_to_hop,
)

SR = 16_000


def synth(duration_s: float, f0: float = 220.0, seed: int = 0) -> torch.Tensor:
    """Synthetic speech-ish signal: harmonic stack + light noise, 1-D @16 kHz.

    Silence is a bad fixture — it produces degenerate tokens that hide bugs.
    """
    g = torch.Generator().manual_seed(seed)
    n = int(duration_s * SR)
    t = torch.arange(n, dtype=torch.float32) / SR
    x = sum(torch.sin(2 * math.pi * f0 * k * t) / k for k in (1, 2, 3))
    x = x * torch.hann_window(n) if n > 1 else x
    return (0.3 * x + 0.01 * torch.randn(n, generator=g)).float()


# ----------------------------------------------------------------- pure units


def test_pad_to_hop_is_idempotent_on_multiples():
    w = torch.zeros(HOP * 3)
    assert pad_to_hop(w).numel() == HOP * 3


@pytest.mark.parametrize("n", [1, 319, 321, 16_001])
def test_pad_to_hop_reaches_multiple(n):
    assert pad_to_hop(torch.zeros(n)).numel() % HOP == 0


def test_pad_to_hop_rejects_2d():
    with pytest.raises(ValueError):
        pad_to_hop(torch.zeros(2, 1000))


def test_frame_formula_matches_50hz():
    assert frames_for_samples(16_000) == 50          # T0 smoke test ground truth
    assert frames_for_samples(8_000) == 25
    assert frames_for_samples(100) == 1              # floor guard, never 0


# ------------------------------------------------------------- model-backed


@pytest.fixture(scope="module")
def codec():
    return NeuCodecWrapper(device="cpu")


@pytest.mark.model
@pytest.mark.parametrize("dur", [0.5, 1.0, 1.37, 3.42])
def test_frame_count_matches_formula(codec, dur):
    """The FRAME FORMULA must hold across durations, not just at 1.0 s."""
    wav = pad_to_hop(synth(dur))
    tokens = codec.encode_one(wav)
    expected = frames_for_samples(wav.numel())
    assert abs(len(tokens) - expected) <= 1, f"{dur}s: got {len(tokens)}, want {expected}"


@pytest.mark.model
def test_roundtrip_shapes_and_rate(codec):
    wav = pad_to_hop(synth(2.0))
    tokens = codec.encode_one(wav)
    recon = codec.decode_one(tokens)
    assert recon.numel() == len(tokens) * SAMPLES_PER_FRAME_24K
    # decoded duration must match input duration (24 kHz out, 16 kHz in)
    assert abs(recon.numel() / DECODE_SR - wav.numel() / SR) < 0.05


@pytest.mark.model
def test_tokens_in_fsq_vocab(codec):
    tokens = codec.encode_one(pad_to_hop(synth(1.0)))
    assert tokens.min() >= 0 and tokens.max() < 65_536
    assert tokens.dtype == torch.long


@pytest.mark.model
def test_encode_path_is_frozen(codec):
    assert not any(p.requires_grad for p in codec.model.parameters())


@pytest.mark.model
def test_mixed_length_batch_rejected(codec):
    with pytest.raises(ValueError, match="mixed-length"):
        codec.encode_batch([synth(1.0), synth(2.0)])


@pytest.mark.model
def test_equal_length_batch_is_token_identical(codec):
    """THE load-bearing invariant: same-length batching must not change tokens.

    If this fails, batched encoding cannot be used at all and T4 must run
    batch_size=1.
    """
    clips = [synth(1.0, f0=f, seed=i) for i, f in enumerate((180.0, 220.0, 300.0))]
    clips = [pad_to_hop(c) for c in clips]
    solo = [codec.encode_one(c) for c in clips]
    batched = codec.encode_batch(clips)
    for i, (a, b) in enumerate(zip(solo, batched)):
        assert torch.equal(a, b), f"clip {i}: batching changed tokens"


@pytest.mark.model
def test_characterize_mixed_length_divergence(codec):
    """CHARACTERIZATION, not a gate. Decides T4's batching strategy.

    Measures how much padding a short clip to a longer batch-mate perturbs its
    tokens. Prints the divergence; only fails if it is total (which would mean
    strict=False is unusable).
    """
    short = pad_to_hop(synth(1.0, f0=220.0))
    long = pad_to_hop(synth(3.0, f0=180.0))
    solo = codec.encode_one(short)
    mixed = codec.encode_batch([short, long], strict=False)[0]

    n = min(len(solo), len(mixed))
    diff = (solo[:n] != mixed[:n]).float()
    frac = diff.mean().item()
    first_bad = int(diff.nonzero()[0].item()) if diff.any() else -1
    print(
        f"\n[mixed-length divergence] {frac:.1%} of {n} frames differ; "
        f"first differing frame = {first_bad} of {n}"
    )
    assert frac < 1.0, "every frame changed — mixed-length batching is unusable"


@pytest.mark.model
def test_decode_batch_trims_per_item(codec):
    seqs = [codec.encode_one(pad_to_hop(synth(d))) for d in (1.0, 1.0)]
    outs = codec.decode_batch(seqs)
    assert len(outs) == 2
    for t, o in zip(seqs, outs):
        assert o.numel() == len(t) * SAMPLES_PER_FRAME_24K

@pytest.mark.model
def test_prepadding_changes_tokens(codec):
    """Does our own pad_to_hop perturb tokens vs. the vanilla upstream path?"""
    raw = synth(1.37)
    a = codec.encode_one(raw)
    b = codec.encode_one(pad_to_hop(raw))
    n = min(len(a), len(b))
    frac = (a[:n] != b[:n]).float().mean().item()
    print(f"\n[pre-pad effect] frames {len(a)} vs {len(b)}; "
          f"{frac:.1%} of {n} shared frames differ")  

@pytest.mark.model
def test_equal_length_decode_batch_identical(codec):
    clips = [pad_to_hop(synth(1.0, f0=f, seed=i)) for i, f in enumerate((180.0, 300.0))]
    toks = codec.encode_batch(clips)
    solo = torch.stack([codec.decode_one(t) for t in toks])
    batched = torch.stack(codec.decode_batch(toks))
    assert torch.allclose(solo, batched, atol=1e-4)          