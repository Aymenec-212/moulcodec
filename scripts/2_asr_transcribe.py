#!/usr/bin/env python
"""Stage B: ASR transcription in an ISOLATED environment (HLD §2.3).

WHY THIS IS A SEPARATE SCRIPT AND A SEPARATE ENV:
    qwen-asr==0.0.6 pins transformers==4.57.6
    neucodec breaks on transformers>=4.47 (HubertModel import fails)
There is no version that satisfies both, so this never runs in the project env.

    uv run --no-project \
        --with "qwen-asr==0.0.6" \
        python scripts/2_asr_transcribe.py \
            --wav-dir outputs/audit_wavs \
            --out-dir audit \
            --language Arabic

STDLIB ONLY (plus qwen_asr). Do NOT import from src/.

Resumable: transcripts already present in the output JSONL are skipped, so an
interrupted run or an expanded audit sample costs only the new clips.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe originals and reconstructions")
    p.add_argument("--wav-dir", required=True, help="dir containing orig/ and recon/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="atlasia/moulsot.v0.3")
    p.add_argument("--language", default="Arabic")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on N clips per side")
    p.add_argument("--sides", default="orig,recon")
    return p.parse_args()


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["audio_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a killed run
    return done


def transcribe_side(model, wav_dir: Path, out_path: Path, language: str, limit) -> int:
    done = load_done(out_path)
    wavs = sorted(wav_dir.glob("*.wav"))
    todo = [w for w in wavs if w.stem not in done]
    if limit:
        todo = todo[:limit]

    print(f"{wav_dir.name}: {len(wavs)} wavs, {len(done)} already done, {len(todo)} to go")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n, t0 = 0, time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for w in todo:
            try:
                result = model.transcribe(audio=str(w), language=language)
                text = getattr(result, "text", None)
                if text is None:  # API surface only documented via the model card
                    text = result["text"] if isinstance(result, dict) else str(result)
                rec = {"audio_id": w.stem, "text": text}
            except Exception as exc:  # noqa: BLE001 — one bad clip must not kill the run
                rec = {"audio_id": w.stem, "text": "", "error": repr(exc)}
                print(f"  ERROR {w.stem}: {exc!r}")

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()  # crash-safe: never lose more than the current clip
            n += 1
            if n % 100 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)} ({rate:.2f} clips/s)")
    return n


def main() -> None:
    args = parse_args()
    wav_root = Path(args.wav_dir)
    out_dir = Path(args.out_dir)

    from qwen_asr import Qwen3ASRModel

    print(f"loading {args.model} on {args.device} ({args.dtype}) ...")
    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=args.dtype, device_map=args.device
    )

    total = 0
    for side in [s.strip() for s in args.sides.split(",") if s.strip()]:
        side_dir = wav_root / side
        if not side_dir.is_dir():
            raise SystemExit(f"missing {side_dir} — run Stage A first")
        total += transcribe_side(
            model, side_dir, out_dir / f"hyps_{side}.jsonl", args.language, args.limit
        )

    try:
        import transformers

        tv = transformers.__version__
    except Exception:  # noqa: BLE001
        tv = "unknown"

    meta_path = out_dir / "asr_meta.json"
    meta = {
        "model": args.model,
        "language": args.language,
        "device": args.device,
        "dtype": args.dtype,
        "transformers": tv,
        "engine": "qwen_asr",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\ntranscribed {total} clips this run; wrote {meta_path}")
    print("Next: scripts/3_build_audit_report.py in the project env.")


if __name__ == "__main__":
    main()