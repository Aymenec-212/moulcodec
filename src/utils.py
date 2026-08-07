"""Validation + config helpers (HLD §2.4/§2.6, spec T3b).

The load-bearing invariant of this project is transcript-token pairing: TTS
training fails SILENTLY on misalignment, so validation runs twice — once when
rows are built, once on the serialised parquet before push (HLD §2.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

HOP = 320
FRAME_RATE = 50
FSQ_VOCAB = 65_536
ENCODER_TAG = "neucodec-v1"

# Release schema (HLD §2.4, revised after the real MoulSot schema was verified):
#   audio_id   <- dataset `id`
#   channel    <- dataset `channel`. A YouTube channel, NOT a speaker identity.
#   n_samples  added so consumers can re-derive the token count without audio.
RELEASE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "audio_id": str,
    "text": str,
    "channel": str,
    "duration": float,
    "n_samples": int,
    "tokens": (list, tuple),
    "encoder": str,
    "sampling_rate": int,
}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------ token sanity


def expected_frames(n_samples: int) -> int:
    """Frames NeuCodec emits for a raw clip. Mirrors src/models.frames_for_samples."""
    return max(1, n_samples // HOP)


def check_token_count(
    n_tokens: int,
    n_samples: int,
    tolerance: int = 1,
) -> tuple[bool, str]:
    """Sanity-check a token sequence length against the source audio length.

    Uses n_samples, not `duration`: the dataset's duration is a rounded float and
    the frame count is exactly n_samples // 320. Returns (ok, reason).
    """
    want = expected_frames(n_samples)
    delta = n_tokens - want
    if abs(delta) <= tolerance:
        return True, ""
    return False, (
        f"token count {n_tokens} != expected {want} (delta {delta:+d}, "
        f"tolerance ±{tolerance}); n_samples={n_samples}"
    )


def check_token_values(tokens: Sequence[int], vocab: int = FSQ_VOCAB) -> tuple[bool, str]:
    if len(tokens) == 0:
        return False, "empty token sequence"
    lo, hi = min(tokens), max(tokens)
    if lo < 0 or hi >= vocab:
        return False, f"tokens out of range [0,{vocab}): min={lo} max={hi}"
    return True, ""


# --------------------------------------------------------- schema validation


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_ok: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def ok(self) -> bool:
        return self.n_rows > 0 and not self.errors

    def add(self, row_id: Any, reason: str) -> None:
        self.errors.append(f"{row_id}: {reason}")

    def summary(self) -> str:
        return f"{self.n_ok}/{self.n_rows} rows valid, {len(self.errors)} errors"


def validate_row(row: dict, tolerance: int = 1) -> tuple[bool, str]:
    """Validate one release row against the §2.4 schema."""
    for field, typ in RELEASE_FIELDS.items():
        if field not in row:
            return False, f"missing field '{field}'"
        if not isinstance(row[field], typ):
            got = type(row[field]).__name__
            return False, f"field '{field}' has type {got}, expected {typ}"

    if not row["text"].strip():
        return False, "empty transcript — pairing invariant broken"
    if row["sampling_rate"] != 16_000:
        return False, f"sampling_rate {row['sampling_rate']} != 16000"
    if row["encoder"] != ENCODER_TAG:
        return False, f"encoder tag '{row['encoder']}' != '{ENCODER_TAG}'"

    ok, why = check_token_values(row["tokens"])
    if not ok:
        return False, why
    return check_token_count(len(row["tokens"]), row["n_samples"], tolerance)


def validate_rows(rows: Iterable[dict], tolerance: int = 1) -> ValidationReport:
    rep = ValidationReport()
    seen: set[str] = set()
    for row in rows:
        rep.n_rows += 1
        rid = row.get("audio_id", f"<row {rep.n_rows}>")
        if rid in seen:
            rep.add(rid, "duplicate audio_id")
            continue
        seen.add(rid)
        ok, why = validate_row(row, tolerance)
        if ok:
            rep.n_ok += 1
        else:
            rep.add(rid, why)
    return rep


def validate_parquet(path: str | Path, tolerance: int = 1) -> ValidationReport:
    """Re-validate the serialised release before push (HLD §2.4, spec T8)."""
    import pandas as pd

    df = pd.read_parquet(path)
    rows = df.to_dict(orient="records")
    for r in rows:  # parquet gives numpy arrays for list columns
        r["tokens"] = [int(t) for t in r["tokens"]]
        r["duration"] = float(r["duration"])
        r["n_samples"] = int(r["n_samples"])
        r["sampling_rate"] = int(r["sampling_rate"])
    return validate_rows(rows, tolerance)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)