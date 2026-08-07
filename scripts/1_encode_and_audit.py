#!/usr/bin/env python
"""Stage A: encode once, fork to audit + release (HLD §2.1, spec T4).

Runs in the CODEC environment (transformers <4.47). ASR does NOT run here:
qwen-asr pins transformers==4.57.6, which breaks neucodec's HubertModel import,
so transcription is Stage B in an isolated env. See scripts/2_asr_transcribe.py.

    uv run python scripts/1_encode_and_audit.py --config configs/pipeline_config.yaml
    uv run python scripts/1_encode_and_audit.py --pilot 100

Outputs:
    <token_store>/tokens-*.parquet   token store — SINGLE SOURCE OF TRUTH
    <audit_dir>/signal_metrics.csv   per-clip Mel-L1 / SI-SDR / speaker-sim
    <audit_dir>/audit_manifest.json  sampled ids, in nested prefix order
    <export_wav_dir>/orig|recon/     16 kHz wavs for Stage B
    <release_dir>/moulsot_tokens.parquet

Resumable: already-encoded audio_ids are read from existing shards and skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import as_float32, exact_length_groups, load_subset, preprocess  # noqa: E402
from src.metrics import SR, mel_l1, si_sdr, to_16k  # noqa: E402
from src.models import DECODE_SR, NeuCodecWrapper  # noqa: E402
from src.sampling import select_audit_sample, stratum_counts  # noqa: E402
from src.utils import ENCODER_TAG, load_config, validate_rows, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoulCodec unified encode + audit pipeline")
    p.add_argument("--config", default="configs/pipeline_config.yaml")
    p.add_argument("--pilot", type=int, default=None, help="limit to first N dataset rows")
    p.add_argument("--audit-sample", type=int, default=None, help="override audit.sample_size")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--skip-release", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--no-resume", action="store_true", help="ignore existing token shards")
    return p.parse_args()


# --------------------------------------------------------------- token store


def existing_ids(store: Path) -> set[str]:
    ids: set[str] = set()
    for shard in sorted(store.glob("tokens-*.parquet")):
        ids.update(pd.read_parquet(shard, columns=["audio_id"])["audio_id"].tolist())
    return ids


def next_shard_index(store: Path) -> int:
    shards = sorted(store.glob("tokens-*.parquet"))
    return len(shards)


def flush_shard(rows: list[dict], store: Path, index: int) -> None:
    if not rows:
        return
    path = store / f"tokens-{index:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  wrote {path.name} ({len(rows)} rows)")


# ------------------------------------------------------------------- encode


def encode_all(ds, codec: NeuCodecWrapper, cfg: dict, store: Path, resume: bool) -> None:
    store.mkdir(parents=True, exist_ok=True)
    done = existing_ids(store) if resume else set()
    if done:
        print(f"resuming: {len(done)} clips already encoded")

    n_samples = ds["n_samples"]
    audio_ids = ds["id"]
    max_batch = int(cfg["encode"].get("batch_size", 1))
    shard_rows = int(cfg["encode"].get("shard_rows", 2000))

    shard_idx = next_shard_index(store)
    buffer: list[dict] = []
    n_done = 0

    for group in exact_length_groups(n_samples, max_batch=max_batch):
        group = [i for i in group if audio_ids[i] not in done]
        if not group:
            continue

        rows_g = [ds[i] for i in group]
        wavs = [torch.from_numpy(as_float32(np.asarray(r["wav"]))) for r in rows_g]
        tokens = codec.encode_batch(wavs, strict=True)

        for row, tok in zip(rows_g, tokens):
            buffer.append(
                {
                    "audio_id": row["id"],
                    "tokens": tok.tolist(),
                    "n_samples": int(row["n_samples"]),
                    "duration": float(row["duration"]),
                    "text": row["text"],
                    "channel": row["channel"],
                    "expected_frames": int(row["expected_frames"]),
                }
            )
        n_done += len(group)

        if len(buffer) >= shard_rows:
            flush_shard(buffer, store, shard_idx)
            shard_idx += 1
            buffer = []
        if n_done % 500 == 0:
            print(f"  encoded {n_done} clips")

    flush_shard(buffer, store, shard_idx)
    print(f"encode complete: {n_done} new clips")


def load_token_store(store: Path) -> pd.DataFrame:
    shards = sorted(store.glob("tokens-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no token shards in {store}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


# -------------------------------------------------------------- audit fork


def run_audit(df: pd.DataFrame, ds, codec: NeuCodecWrapper, cfg: dict, n_sample: int) -> None:
    import soundfile as sf

    a = cfg["audit"]
    audit_dir = Path(cfg["paths"]["audit_dir"])
    wav_dir = Path(a["export_wav_dir"])
    (wav_dir / "orig").mkdir(parents=True, exist_ok=True)
    (wav_dir / "recon").mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    picks = select_audit_sample(
        df["audio_id"].tolist(),
        df["duration"].tolist(),
        df["channel"].tolist(),
        n=n_sample,
        seed=int(a.get("seed", 42)),
        edges=tuple(a.get("strata_duration_edges", (2.0, 4.0, 6.0, 8.0, 10.0))),
    )
    print(f"audit sample: {len(picks)} clips (nested prefix, seed={a.get('seed', 42)})")

    wav_by_id = {aid: i for i, aid in enumerate(ds["id"])}   # single column only
    speaker = None
    if a.get("speaker_similarity", True):
        from src.metrics import SpeakerSimilarity

        speaker = SpeakerSimilarity(model_id=cfg["speaker"]["model_id"])

    out_rows = []
    for k, idx in enumerate(picks, 1):
        row = df.iloc[idx]
        aid = row["audio_id"]
        ref = torch.from_numpy(as_float32(np.asarray(ds[wav_by_id[aid]]["wav"])))
        recon24 = codec.decode_one(row["tokens"])

        rec = {
            "audio_id": aid,
            "duration": float(row["duration"]),
            "channel": row["channel"],
            "n_frames": len(row["tokens"]),
            "mel_l1": mel_l1(ref, recon24),
            "si_sdr": si_sdr(ref, recon24),
        }
        if speaker is not None:
            rec["speaker_sim"] = speaker(ref, recon24)
        out_rows.append(rec)

        # 16 kHz wavs for Stage B (ASR expects 16 kHz mono)
        sf.write(wav_dir / "orig" / f"{aid}.wav", ref.numpy(), SR)
        sf.write(wav_dir / "recon" / f"{aid}.wav", to_16k(recon24, DECODE_SR).numpy(), SR)

        if k % 200 == 0:
            print(f"  audited {k}/{len(picks)}")

    csv_path = audit_dir / "signal_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {csv_path}")

    write_json(
        {
            "seed": int(a.get("seed", 42)),
            "n_sampled": len(picks),
            "audio_ids": [df.iloc[i]["audio_id"] for i in picks],
            "stratum_counts": {
                f"{k[0]}|{k[1]}": v
                for k, v in stratum_counts(
                    df["duration"].tolist(), df["channel"].tolist(), picks
                ).items()
            },
        },
        audit_dir / "audit_manifest.json",
    )


# ------------------------------------------------------------ release fork


def run_release(df: pd.DataFrame, cfg: dict) -> None:
    release_dir = Path(cfg["paths"]["release_dir"])
    release_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "audio_id": r["audio_id"],
            "text": r["text"],
            "channel": r["channel"],
            "duration": float(r["duration"]),
            "n_samples": int(r["n_samples"]),
            "tokens": [int(t) for t in r["tokens"]],
            "encoder": ENCODER_TAG,
            "sampling_rate": SR,
        }
        for _, r in df.iterrows()
    ]

    rep = validate_rows(rows, tolerance=int(cfg["validation"].get("frame_tolerance", 1)))
    print(f"release validation: {rep.summary()}")
    if not rep.ok:
        for e in rep.errors[:10]:
            print(f"  ERROR {e}")
        if cfg["validation"].get("fail_fast", True):
            raise SystemExit("release validation failed — refusing to serialise")

    path = release_dir / "moulsot_tokens.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"wrote {path} ({len(rows)} rows)")


# ------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print("loading dataset ...")
    ds = load_subset(cfg, pilot=args.pilot)
    print(f"  {len(ds)} rows")

    print("CPU preprocessing ...")
    ds = preprocess(ds, cfg)
    ds = ds.with_format("numpy")     # row access returns np.ndarray directly

    print(f"loading codec on {args.device or 'auto'} ...")
    codec = NeuCodecWrapper(model_id=cfg["codec"]["model_id"], device=args.device)

    store = Path(cfg["paths"]["token_store"])
    print("encoding ...")
    encode_all(ds, codec, cfg, store, resume=not args.no_resume)

    df = load_token_store(store)
    print(f"token store: {len(df)} clips")

    bad = df[df["tokens"].map(len) != df["expected_frames"]]
    print(f"frame-count exact matches: {len(df) - len(bad)}/{len(df)}")

    if not args.skip_audit:
        n = args.audit_sample or int(cfg["audit"]["sample_size"])
        run_audit(df, ds, codec, cfg, n_sample=n)

    if not args.skip_release:
        run_release(df, cfg)

    print("Stage A done. Next: scripts/2_asr_transcribe.py in the isolated ASR env.")


if __name__ == "__main__":
    main()