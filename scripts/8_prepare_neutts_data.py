#!/usr/bin/env python3
"""Build NeuTTS Air training tensors from Tilas/MoulSot-Tokens-v1.

CPU only - no GPU, no model weights. Run this on a laptop so the GPU box is rented
only for GPU work. Output is a `datasets` directory of input_ids ready for Trainer.

    python scripts/8_prepare_neutts_data.py --verify 200          # check, don't write
    python scripts/8_prepare_neutts_data.py --num-proc 8 --out data/neutts_darija

Why piecewise ID construction
-----------------------------
Upstream builds one big string and calls tokenizer.encode() on it, which means running
the tokenizer over 18.3M `<|speech_i|>` substrings. Preflight confirmed those ids are
contiguous, so speech ids are just `SPEECH_BASE + code` and the surrounding template
pieces are constant. That is orders of magnitude faster - but only valid if the
tokenizer really does split cleanly on the added special tokens, so `--verify` builds
both ways and asserts they are identical before trusting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.darija_g2p import DarijaPhonemizer, filter_reason, to_phones  # noqa: E402

REPO_MODEL = "neuphonic/neutts-air"
REPO_DATA = "Tilas/MoulSot-Tokens-v1"

# Exactly the template in neuphonic/neutts examples/finetune.py, which is also the
# prefix NeuTTS._apply_chat_template builds at inference in "phonemes" mode.
PREFIX_TEXT = "user: Convert the text to speech:"
INFIX_TEXT = "\nassistant:"

# One phonemizer per worker process; EspeakBackend is not picklable.
_G2P = None


def _g2p() -> DarijaPhonemizer:
    global _G2P
    if _G2P is None:
        _G2P = DarijaPhonemizer()
    return _G2P


def build_pieces(tok):
    """Constant id segments plus the speech-token base offset."""
    sid = tok.convert_tokens_to_ids
    pieces = {
        "prefix": tok.encode(PREFIX_TEXT, add_special_tokens=False),
        "text_start": [sid("<|TEXT_PROMPT_START|>")],
        "text_end": [sid("<|TEXT_PROMPT_END|>")],
        "infix": tok.encode(INFIX_TEXT, add_special_tokens=False),
        "speech_start": [sid("<|SPEECH_GENERATION_START|>")],
        "speech_end": [sid("<|SPEECH_GENERATION_END|>")],
    }
    base = sid("<|speech_0|>")
    if sid("<|speech_1|>") != base + 1 or sid(f"<|speech_{65535}|>") != base + 65535:
        raise SystemExit("speech token ids are not contiguous; rerun the preflight")
    return pieces, base


def assemble(pieces, base, phone_ids, codes):
    """input_ids for one sample, plus the index where supervision starts."""
    head = (
        pieces["prefix"]
        + pieces["text_start"]
        + phone_ids
        + pieces["text_end"]
        + pieces["infix"]
    )
    speech_start_idx = len(head)
    ids = (
        head
        + pieces["speech_start"]
        + [base + int(c) for c in codes]
        + pieces["speech_end"]
    )
    return ids, speech_start_idx


def verify(tok, pieces, base, ds, n: int) -> None:
    """Assert the fast path reproduces upstream's string-encoding byte for byte."""
    print(f"\nVerifying piecewise construction against string encoding on {n} samples...")
    g2p = _g2p()
    checked = 0
    for i in range(min(n, len(ds))):
        row = ds[i]
        text, codes = row["text"], row["tokens"]
        if filter_reason(text, len(codes), max_speech_tokens=10**9) is not None:
            continue
        phones = to_phones(g2p, text)
        if not phones:
            continue

        fast, _ = assemble(pieces, base, tok.encode(phones, add_special_tokens=False), codes)

        codes_str = "".join(f"<|speech_{c}|>" for c in codes)
        chat = (
            f"{PREFIX_TEXT}<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>"
            f"{INFIX_TEXT}<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"
        )
        slow = tok.encode(chat)

        if fast != slow:
            first = next(
                (j for j, (a, b) in enumerate(zip(fast, slow)) if a != b), min(len(fast), len(slow))
            )
            raise SystemExit(
                f"MISMATCH on row {i} at index {first}\n"
                f"  fast[{first-3}:{first+3}] = {fast[max(0,first-3):first+3]}\n"
                f"  slow[{first-3}:{first+3}] = {slow[max(0,first-3):first+3]}\n"
                f"  lengths: fast={len(fast)} slow={len(slow)}\n"
                "The tokenizer is not splitting cleanly on added tokens; fall back to "
                "string encoding."
            )
        checked += 1
    print(f"  identical on all {checked} filtered samples.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/neutts_darija"))
    ap.add_argument("--max-seq-len", type=int, default=896,
                    help="drop samples longer than this (preflight p99 was 859)")
    ap.add_argument("--min-speech-tokens", type=int, default=50,
                    help="1.0 s, matching min_new_tokens=50 at inference")
    ap.add_argument("--eval-size", type=int, default=1000)
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--verify", type=int, default=200,
                    help="samples to cross-check; 0 to skip")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N rows; for smoke-testing the "
                         "num_proc map path before the full run")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO_MODEL)
    pieces, base = build_pieces(tok)
    overhead = sum(len(v) for v in pieces.values())
    print(f"speech token base id: {base}")
    print(f"template overhead:    {overhead} tokens")
    print(f"max_speech_tokens:    {args.max_seq_len - overhead} "
          f"({(args.max_seq_len - overhead) / 50:.1f} s) before phonemes")

    ds = load_dataset(REPO_DATA, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"--limit active: using first {len(ds):,} rows (SMOKE TEST, not a real run)")
    print(f"loaded {len(ds):,} rows")

    if args.verify:
        verify(tok, pieces, base, ds, args.verify)
    if args.verify_only:
        return 0

    def process(batch):
        g2p = _g2p()
        out = {"input_ids": [], "speech_start_idx": [], "length": [],
               "audio_id": [], "reject": []}
        for audio_id, text, codes in zip(batch["audio_id"], batch["text"], batch["tokens"]):
            reason = filter_reason(
                text,
                n_speech_tokens=len(codes),
                # leave room for phonemes; the exact total is re-checked below
                max_speech_tokens=args.max_seq_len - overhead,
                min_speech_tokens=args.min_speech_tokens,
            )
            if reason is None:
                phones = to_phones(g2p, text)
                if not phones:
                    reason = "empty_phonemes"
            if reason is None:
                phone_ids = tok.encode(phones, add_special_tokens=False)
                ids, start = assemble(pieces, base, phone_ids, codes)
                if len(ids) > args.max_seq_len:
                    reason = "too_long_with_phonemes"
            if reason is not None:
                out["input_ids"].append([])
                out["speech_start_idx"].append(-1)
                out["length"].append(0)
                out["audio_id"].append(audio_id)
                out["reject"].append(reason)
                continue
            out["input_ids"].append(ids)
            out["speech_start_idx"].append(start)
            out["length"].append(len(ids))
            out["audio_id"].append(audio_id)
            out["reject"].append("")
        return out

    print(f"\nphonemizing and tokenizing with num_proc={args.num_proc}...")
    proc = ds.map(
        process,
        batched=True,
        batch_size=256,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="prepare",
    )

    from collections import Counter

    rejects = Counter(r for r in proc["reject"] if r)
    kept = proc.filter(lambda r: r == "", input_columns="reject", num_proc=args.num_proc)
    kept = kept.remove_columns(["reject"])

    print(f"\nkept {len(kept):,} of {len(ds):,} ({100 * len(kept) / len(ds):.1f}%)")
    for reason, count in rejects.most_common():
        print(f"  {reason:<24} {count:,}")

    lens = kept["length"]
    lens_sorted = sorted(lens)
    pct = lambda p: lens_sorted[int(p * (len(lens_sorted) - 1))]  # noqa: E731
    print(f"\nsequence length: p50 {pct(.5)}  p95 {pct(.95)}  p99 {pct(.99)}  max {lens_sorted[-1]}")
    speech_tokens = sum(l - s - 2 for l, s in zip(lens, kept["speech_start_idx"]))
    print(f"speech tokens kept: {speech_tokens:,} ({speech_tokens / 50 / 3600:.1f} h)")
    print(f"padding waste at max_seq_len={args.max_seq_len}: "
          f"{100 * (1 - sum(lens) / (len(lens) * args.max_seq_len)):.0f}% "
          f"(this is why the collator pads dynamically instead)")

    eval_size = min(args.eval_size, max(1, len(kept) // 10))
    if eval_size != args.eval_size:
        print(f"eval split reduced to {eval_size} (dataset has only {len(kept):,} rows)")
    split = kept.train_test_split(test_size=eval_size, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(args.out))

    meta = {
        "source": REPO_DATA,
        "tokenizer": REPO_MODEL,
        "speech_token_base": base,
        "template_overhead": overhead,
        "max_seq_len": args.max_seq_len,
        "min_speech_tokens": args.min_speech_tokens,
        "strip_diacritics": True,
        "n_train": len(split["train"]),
        "n_eval": len(split["test"]),
        "rejects": dict(rejects),
        "length_p50": pct(.5), "length_p95": pct(.95), "length_p99": pct(.99),
        "seed": args.seed,
    }
    (args.out / "prepare_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out}  (train {len(split['train']):,} / eval {len(split['test']):,})")
    print(f"wrote {args.out / 'prepare_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())