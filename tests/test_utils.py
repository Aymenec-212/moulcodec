"""T3b tests for src/utils.py — schema + token sanity validation.

    uv run pytest tests/test_utils.py -v
"""

from __future__ import annotations

import pytest

from src.utils import (
    ENCODER_TAG,
    FSQ_VOCAB,
    check_token_count,
    check_token_values,
    expected_frames,
    validate_parquet,
    validate_row,
    validate_rows,
)


def good_row(**over) -> dict:
    row = {
        "audio_id": "moulsot_001_002",
        "text": "واخا نبداو العملية ديال اليوم",
        "channel": "some_channel",
        "duration": 1.0,
        "n_samples": 16_000,
        "tokens": list(range(50)),
        "encoder": ENCODER_TAG,
        "sampling_rate": 16_000,
    }
    row.update(over)
    return row


# ------------------------------------------------------------ token sanity


def test_expected_frames_matches_ground_truth():
    assert expected_frames(16_000) == 50
    assert expected_frames(21_920) == 68
    assert expected_frames(10) == 1


def test_check_token_count_accepts_exact():
    ok, why = check_token_count(50, 16_000)
    assert ok and why == ""


def test_check_token_count_accepts_within_tolerance():
    assert check_token_count(51, 16_000)[0]
    assert check_token_count(49, 16_000)[0]


def test_check_token_count_rejects_outside_tolerance():
    ok, why = check_token_count(40, 16_000)
    assert not ok and "expected 50" in why


def test_check_token_count_reports_direction():
    _, why = check_token_count(60, 16_000)
    assert "+10" in why


def test_check_token_values_rejects_empty():
    ok, why = check_token_values([])
    assert not ok and "empty" in why


def test_check_token_values_rejects_out_of_vocab():
    assert not check_token_values([0, FSQ_VOCAB])[0]
    assert not check_token_values([-1, 5])[0]


def test_check_token_values_accepts_edges():
    assert check_token_values([0, FSQ_VOCAB - 1])[0]


# ---------------------------------------------------------------- schema


def test_valid_row_passes():
    ok, why = validate_row(good_row())
    assert ok, why


@pytest.mark.parametrize(
    "field", ["audio_id", "text", "channel", "duration", "n_samples", "tokens", "encoder"]
)
def test_missing_field_fails(field):
    row = good_row()
    del row[field]
    ok, why = validate_row(row)
    assert not ok and field in why


def test_empty_transcript_fails_pairing_invariant():
    ok, why = validate_row(good_row(text="   "))
    assert not ok and "pairing" in why


def test_wrong_sampling_rate_fails():
    assert not validate_row(good_row(sampling_rate=24_000))[0]


def test_wrong_encoder_tag_fails():
    assert not validate_row(good_row(encoder="distill-neucodec"))[0]


def test_token_count_mismatch_fails():
    assert not validate_row(good_row(tokens=list(range(10))))[0]


def test_wrong_type_fails():
    assert not validate_row(good_row(duration="1.0"))[0]


# ---------------------------------------------------------------- batches


def test_validate_rows_counts_and_reports():
    rep = validate_rows([good_row(), good_row(audio_id="b", text="")])
    assert rep.n_rows == 2 and rep.n_ok == 1
    assert not rep.ok and "b:" in rep.errors[0]


def test_validate_rows_catches_duplicate_ids():
    rep = validate_rows([good_row(), good_row()])
    assert not rep.ok and "duplicate" in rep.errors[0]


def test_validate_rows_empty_is_not_ok():
    assert not validate_rows([]).ok


def test_validate_parquet_roundtrip(tmp_path):
    import pandas as pd

    path = tmp_path / "release.parquet"
    pd.DataFrame([good_row(), good_row(audio_id="x")]).to_parquet(path)
    rep = validate_parquet(path)
    assert rep.ok, rep.errors


def test_validate_parquet_catches_broken_pairing(tmp_path):
    import pandas as pd

    path = tmp_path / "bad.parquet"
    pd.DataFrame([good_row(), good_row(audio_id="x", text="")]).to_parquet(path)
    assert not validate_parquet(path).ok