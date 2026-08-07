#!/usr/bin/env python
"""T8: validate the serialised release and push it to the Hub (HLD §2.5).

    uv run python scripts/4_push_to_hub.py --config configs/pipeline_config.yaml
    uv run python scripts/4_push_to_hub.py --dry-run          # validate + print card
    uv run python scripts/4_push_to_hub.py --confirm          # actually push

Nothing is pushed without --confirm. The default is a dry run, because a
dataset push is hard to undo and the card must be read before it ships.

Validation runs TWICE by design (HLD §2.4): once inside Stage A when rows are
built, and again here on the deserialised parquet. Transcript-token pairing is
the load-bearing invariant — TTS training fails silently on misalignment, so it
is checked after every serialisation boundary, not once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ENCODER_TAG, load_config, validate_parquet  # noqa: E402

CARD_TEMPLATE = """---
license: {license}
task_categories:
- text-to-speech
- automatic-speech-recognition
language:
- ary
- ar
tags:
- darija
- moroccan-arabic
- speech
- neural-audio-codec
- neucodec
- fsq
- speech-tokens
size_categories:
- {size_category}
---

# {repo_name}

Discrete speech tokens for Moroccan Darija, produced by encoding the transcribed
subset of [`{source_dataset}`](https://huggingface.co/datasets/{source_dataset})
with [`{codec}`](https://huggingface.co/{codec}).

Each row pairs a transcript with the token sequence for the same clip, ready for
training a text-to-speech model that predicts tokens directly.

## Provenance and caveats

Read these before training on this data.

1. **Tokens come from `{codec}`** — the full encoder, not a distilled variant.
   The encode path (Wav2Vec2-BERT semantic encoder, BigCodec acoustic encoder,
   FSQ quantiser) was frozen throughout; no encoder weights were modified, so
   these tokens are reproducible against the upstream checkpoint.
2. **Transcripts are machine-generated.** The `text` field originates from
   `{source_dataset}`, where transcripts were produced by Gemini 2.5 Pro and are
   **not human-verified**. Expect transcription noise.
3. **Licensing is inherited from `{source_dataset}`**, which is sourced from
   YouTube. Verify the upstream terms apply to your use case before
   redistributing or training commercially.
4. **`channel` is not a speaker ID.** It is the source YouTube channel. One
   channel may contain many speakers and one speaker may appear across
   channels. Do not use it as a speaker label for multi-speaker conditioning
   without further diarisation.

## Reproducing the tokens

Token IDs depend on the exact input given to the encoder. To reproduce them,
pass **raw mono 16 kHz float32 audio** — unpadded — to `encode_code`, one clip
per call:

```python
import torch
from neucodec import NeuCodec

model = NeuCodec.from_pretrained("{codec}").eval()
wav = ...  # 1-D float32 @ 16 kHz, raw length, NO padding
codes = model.encode_code(wav.reshape(1, 1, -1))
```

Two details matter and are easy to get wrong:

- **Do not pre-pad.** NeuCodec pads internally. Adding your own padding first
  changes the semantic encoder's per-utterance feature normalisation and shifts
  token identity — measured at ~85% of frames changed by only 10 ms of
  appended silence.
- **Do not batch clips of differing lengths.** `encode_code` applies no
  attention mask, so padding a short clip up to a longer batch-mate perturbs
  ~86% of its frames. Batch only clips of identical sample length, or use
  batch size 1.

Frame count follows `n_frames == n_samples // 320` (50 Hz).

## Schema

| Field | Type | Description |
|---|---|---|
| `audio_id` | string | Clip identifier, matching the source dataset |
| `text` | string | Machine-generated Darija transcript (see caveat 2) |
| `channel` | string | Source YouTube channel — NOT a speaker ID (caveat 4) |
| `duration` | float | Clip duration in seconds |
| `n_samples` | int | Raw sample count at 16 kHz; `n_samples // 320 == len(tokens)` |
| `tokens` | list[int] | FSQ token IDs in `[0, 65536)`, 50 tokens/second |
| `encoder` | string | `{encoder_tag}` |
| `sampling_rate` | int | 16000 (encoder input rate) |

## Statistics

- Clips: **{n_clips}**
- Total duration: **{total_hours:.1f} hours**
- Total tokens: **{n_tokens}**
- Distinct token IDs used: **{n_distinct}** of 65,536 ({coverage:.1f}% coverage)

## Decoding

The NeuCodec decoder outputs **24 kHz** audio; there is no 16 kHz output path.

```python
audio24k = model.decode_code(torch.tensor(tokens).reshape(1, 1, -1))
```

## Audit

Codec quality on this corpus was measured zero-shot using delta-WER against
[`atlasia/moulsot.v0.3`](https://huggingface.co/atlasia/moulsot.v0.3) — the
difference between ASR error on reconstructed audio and on the original.
Absolute WER is not a meaningful codec signal here, because the references are
themselves machine-generated. See the project repository for the full audit
report.

## Citation

Produced by the MoulCodec project, contributing open speech tooling to the
Moroccan AI ecosystem.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate and push MoulSot-Tokens")
    p.add_argument("--config", default="configs/pipeline_config.yaml")
    p.add_argument("--parquet", default=None)
    p.add_argument("--repo", default=None)
    p.add_argument("--license", default="cc-by-nc-4.0", help="must match upstream terms")
    p.add_argument("--confirm", action="store_true", help="actually push (default: dry run)")
    p.add_argument("--private", action="store_true", default=None)
    return p.parse_args()


def size_category(n: int) -> str:
    for bound, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < bound:
            return label
    return "100K<n<1M"


def build_card(df: pd.DataFrame, cfg: dict, repo: str, license_: str) -> str:
    lens = df["tokens"].map(len)
    distinct = len({int(t) for row in df["tokens"] for t in row})
    return CARD_TEMPLATE.format(
        license=license_,
        size_category=size_category(len(df)),
        repo_name=repo.split("/")[-1],
        source_dataset=cfg["dataset"]["id"],
        codec=cfg["codec"]["model_id"],
        encoder_tag=ENCODER_TAG,
        n_clips=f"{len(df):,}",
        total_hours=float(df["duration"].sum()) / 3600.0,
        n_tokens=f"{int(lens.sum()):,}",
        n_distinct=f"{distinct:,}",
        coverage=100.0 * distinct / 65_536,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    parquet = Path(args.parquet or Path(cfg["paths"]["release_dir"]) / "moulsot_tokens.parquet")
    repo = args.repo or cfg["hub"]["dataset_repo"]
    private = cfg["hub"].get("private", True) if args.private is None else args.private

    if not parquet.exists():
        raise SystemExit(f"{parquet} not found — run Stage A first")

    print(f"validating {parquet} ...")
    rep = validate_parquet(parquet, tolerance=int(cfg["validation"].get("frame_tolerance", 1)))
    print(f"  {rep.summary()}")
    if not rep.ok:
        for e in rep.errors[:20]:
            print(f"  ERROR {e}")
        raise SystemExit("validation failed — refusing to push")

    df = pd.read_parquet(parquet)
    card = build_card(df, cfg, repo, args.license)

    card_path = parquet.parent / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"wrote {card_path}")

    if not args.confirm:
        print("\n" + "=" * 70)
        print(card)
        print("=" * 70)
        print(f"\nDRY RUN. Target: {repo} (private={private})")
        print("Read the card above, then re-run with --confirm to push.")
        return

    from datasets import Dataset
    from huggingface_hub import HfApi

    print(f"pushing {len(df)} rows to {repo} (private={private}) ...")
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(repo, private=private)

    HfApi().upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"pushed. Verify with: load_dataset('{repo}')")
    print("Then re-run validation on the downloaded copy (spec T8 done-when).")


if __name__ == "__main__":
    main()