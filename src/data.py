"""MoulSot-Full loader + CPU preprocessing (HLD §2.1, spec T2).


NO PADDING HAPPENS HERE. Pre-padding to a hop multiple perturbs ~85% of tokens
(measured in T3: +160 samples changed 85.3% of frames) because the semantic
encoder normalises features per utterance over whatever it is handed. Raw
lengths go to encode_code; NeuCodec owns its own internal padding.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from datasets import Audio, Dataset, load_dataset

TARGET_SR = 16_000
HOP = 320  # frames = n_samples // HOP  (you can check src/models.py FRAME FORMULA)

REQUIRED_COLUMNS = {"id", "audio", "text", "duration", "channel", "sample_rate", "n_channels"}
SOURCE_QUALITY_COLUMNS = ("pesq_hyp", "stoi_hyp", "si_sdr_hyp")


# --------------------------------------------------------------------- load


def load_subset(cfg: dict, pilot: int | None = None, streaming: bool = False) -> Dataset:
    """Load the transcribed subset and force 16 kHz decode.

    `pilot` slices the first N rows (spec T5 gate). Casting the Audio feature is
    what makes torchcodec resample on decode, so downstream code can assume 16 kHz.
    """
    d = cfg["dataset"]
    ds = load_dataset(
        d["id"],
        d["config"],
        split=d["split"],
        streaming=streaming,
        cache_dir=cfg.get("paths", {}).get("cache_dir"),
    )

    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(f"dataset schema changed; missing columns: {sorted(missing)}")

    if pilot is not None and not streaming:
        ds = ds.select(range(min(pilot, len(ds))))

    return ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))


# ----------------------------------------------------------------- decoding


def extract_samples(audio: Any) -> tuple[np.ndarray, int]:
    """Return (mono float32 1-D array, sample_rate) from any datasets Audio value.

    Handles the torchcodec AudioDecoder (datasets>=4) and the legacy dict form so
    the loader does not silently break on a datasets upgrade.
    """
    if hasattr(audio, "get_all_samples"):  # torchcodec AudioDecoder
        s = audio.get_all_samples()
        arr = s.data.numpy()
        sr = int(s.sample_rate)
    elif isinstance(audio, dict) and "array" in audio:  # legacy datasets<4
        arr = np.asarray(audio["array"])
        sr = int(audio["sampling_rate"])
    else:
        raise TypeError(f"unsupported audio value of type {type(audio).__name__}")

    if arr.ndim == 2:  # [C, T] -> mono
        arr = arr.mean(axis=0)
    elif arr.ndim != 1:
        raise ValueError(f"unexpected audio array shape {arr.shape}")

    return np.ascontiguousarray(arr, dtype=np.float32), sr


def expected_frames(n_samples: int) -> int:
    """Token count NeuCodec will emit for a raw clip of n_samples."""
    return max(1, n_samples // HOP)


def _preprocess_batch(batch: dict, store_int16: bool = False) -> dict:
    wavs, n_samples = [], []
    for audio in batch["audio"]:
        arr, sr = extract_samples(audio)
        if sr != TARGET_SR:
            raise ValueError(f"expected {TARGET_SR} Hz after cast_column, got {sr}")
        if store_int16:
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767.0).astype(np.int16)
        wavs.append(arr)
        n_samples.append(int(len(arr)))

    return {
        "wav": wavs,
        "n_samples": n_samples,
        "expected_frames": [expected_frames(n) for n in n_samples],
    }


def preprocess(ds: Dataset, cfg: dict) -> Dataset:
    """CPU-only preprocessing: decode -> mono -> 16 kHz float32.

    Runs with num_proc>1 (HLD §1: fork for CPU, never for GPU). The `audio`
    column is removed so AudioDecoder objects are never pickled or re-serialised
    into the cache.

    Disk note: 80h of float32 @16 kHz is ~18 GB in the arrow cache. Set
    preprocess.store_int16 to halve that at the cost of a lossless
    int16 round-trip at encode time.
    """
    p = cfg.get("preprocess", {})
    store_int16 = bool(p.get("store_int16", False))

    return ds.map(
        _preprocess_batch,
        batched=True,
        batch_size=p.get("map_batch_size", 64),
        num_proc=p.get("num_proc", 4),
        remove_columns=["audio"],
        writer_batch_size=p.get("writer_batch_size", 200),
        fn_kwargs={"store_int16": store_int16},
        desc="decode -> mono -> 16 kHz",
    )


def as_float32(wav: np.ndarray) -> np.ndarray:
    """Undo store_int16. No-op for float input."""
    if wav.dtype == np.int16:
        return wav.astype(np.float32) / 32768.0
    return np.asarray(wav, dtype=np.float32)


# ---------------------------------------------------------------- batching


def exact_length_groups(
    n_samples: list[int], max_batch: int = 1
) -> Iterator[list[int]]:
    """Yield index groups whose clips share an IDENTICAL raw sample count.

    Exact equality is required, not approximate bucketing: padding a clip to
    match a longer batch-mate perturbed 86% of its tokens (T3 measurement).
    Order is deterministic (sorted by length, then index) so a resumed run
    reproduces the same grouping — otherwise tokens would depend on run history.
    """
    by_len: dict[int, list[int]] = {}
    for i, n in enumerate(n_samples):
        by_len.setdefault(int(n), []).append(i)

    for length in sorted(by_len):
        idxs = by_len[length]
        for s in range(0, len(idxs), max_batch):
            yield idxs[s : s + max_batch]