"""T2 tests for src/data.py.

All offline: fixtures are synthetic wavs written to tmp_path, so these run in CI
without Hub access (spec ground rule 4). No `model` marker needed.

    uv run pytest tests/test_data.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, Dataset

from src.data import (
    HOP,
    TARGET_SR,
    as_float32,
    exact_length_groups,
    expected_frames,
    extract_samples,
    preprocess,
)


def _tone(n: int, sr: int, f0: float = 220.0, channels: int = 1) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    x = (0.3 * np.sin(2 * math.pi * f0 * t)).astype(np.float32)
    if channels == 1:
        return x
    return np.stack([x, np.roll(x, 7)], axis=-1)  # soundfile wants [T, C]


def _write(tmp_path, name, n, sr, channels=1):
    p = tmp_path / name
    sf.write(p, _tone(n, sr, channels=channels), sr)
    return str(p)


def _ds(paths):
    return Dataset.from_dict({"audio": paths}).cast_column(
        "audio", Audio(sampling_rate=TARGET_SR)
    )


# ------------------------------------------------------------- pure helpers


def test_expected_frames_matches_50hz():
    assert expected_frames(16_000) == 50      # T0 ground truth
    assert expected_frames(21_920) == 68      # T3 ground truth (1.37 s)
    assert expected_frames(100) == 1          # floor guard


def test_as_float32_roundtrip_is_lossless_enough():
    x = np.array([-1.0, -0.5, 0.0, 0.25, 0.999], dtype=np.float32)
    i16 = (np.clip(x, -1, 1) * 32767.0).astype(np.int16)
    assert np.allclose(as_float32(i16), x, atol=1e-4)
    assert as_float32(x).dtype == np.float32


def test_extract_rejects_unknown_type():
    with pytest.raises(TypeError):
        extract_samples(object())


def test_extract_handles_legacy_dict_and_downmixes():
    stereo = np.stack([np.ones(100), np.zeros(100)])  # [C, T]
    arr, sr = extract_samples({"array": stereo, "sampling_rate": TARGET_SR})
    assert arr.ndim == 1 and arr.shape == (100,)
    assert np.allclose(arr, 0.5)                       # mean of the two channels
    assert arr.dtype == np.float32 and sr == TARGET_SR


# ------------------------------------------------------- decode / normalise


def test_stereo_is_downmixed_to_mono(tmp_path):
    ds = _ds([_write(tmp_path, "st.wav", 16_000, TARGET_SR, channels=2)])
    out = preprocess(ds, {"preprocess": {"num_proc": 1}})
    assert np.asarray(out[0]["wav"]).ndim == 1


def test_non_16k_input_is_resampled(tmp_path):
    ds = _ds([_write(tmp_path, "hi.wav", 44_100, 44_100)])  # exactly 1.0 s @44.1k
    out = preprocess(ds, {"preprocess": {"num_proc": 1}})
    assert abs(out[0]["n_samples"] - TARGET_SR) < 200      # ~1.0 s @16k


def test_odd_length_is_preserved_not_padded(tmp_path):
    """The critical T3 finding: preprocessing must NOT pad to a hop multiple."""
    odd = 16_001
    ds = _ds([_write(tmp_path, "odd.wav", odd, TARGET_SR)])
    out = preprocess(ds, {"preprocess": {"num_proc": 1}})
    assert out[0]["n_samples"] == odd
    assert out[0]["n_samples"] % HOP != 0


def test_expected_frames_column_is_populated(tmp_path):
    ds = _ds([_write(tmp_path, "a.wav", 16_000, TARGET_SR)])
    out = preprocess(ds, {"preprocess": {"num_proc": 1}})
    assert out[0]["expected_frames"] == 50


def test_audio_column_is_dropped(tmp_path):
    ds = _ds([_write(tmp_path, "a.wav", 8_000, TARGET_SR)])
    out = preprocess(ds, {"preprocess": {"num_proc": 1}})
    assert "audio" not in out.column_names
    assert {"wav", "n_samples", "expected_frames"} <= set(out.column_names)


def test_int16_storage_halves_and_restores(tmp_path):
    ds = _ds([_write(tmp_path, "a.wav", 16_000, TARGET_SR)])
    f32 = preprocess(ds, {"preprocess": {"num_proc": 1}})
    i16 = preprocess(ds, {"preprocess": {"num_proc": 1, "store_int16": True}})
    a = np.asarray(f32[0]["wav"], dtype=np.float32)
    b = as_float32(np.asarray(i16[0]["wav"], dtype=np.int16))
    assert np.allclose(a, b, atol=1e-3)


# ---------------------------------------------------------------- grouping


def test_exact_length_groups_never_mix_lengths():
    lengths = [100, 200, 100, 300, 200, 100]
    for g in exact_length_groups(lengths, max_batch=4):
        assert len({lengths[i] for i in g}) == 1


def test_exact_length_groups_respect_max_batch():
    lengths = [100] * 7
    sizes = [len(g) for g in exact_length_groups(lengths, max_batch=3)]
    assert sizes == [3, 3, 1]


def test_exact_length_groups_cover_every_index_once():
    lengths = [5, 5, 9, 1, 9, 9]
    seen = [i for g in exact_length_groups(lengths, max_batch=2) for i in g]
    assert sorted(seen) == list(range(len(lengths)))


def test_exact_length_groups_are_deterministic():
    lengths = [7, 3, 7, 3, 11]
    a = list(exact_length_groups(lengths, max_batch=2))
    b = list(exact_length_groups(lengths, max_batch=2))
    assert a == b, "grouping must be stable or a resumed run changes tokens"


def test_default_max_batch_is_one():
    assert [len(g) for g in exact_length_groups([4, 4, 4])] == [1, 1, 1]