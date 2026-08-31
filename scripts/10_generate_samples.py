#!/usr/bin/env python3
"""Synthesize Darija speech from the fine-tuned NeuTTS Air model.

The acceptance test for this project is whether the audio is intelligible Darija, not
what the loss says. Run this before deciding whether to train longer.

    # unconditioned - exactly the training distribution, no reference voice
    python scripts/10_generate_samples.py --model Tilas/neutts-air-darija-v1 --no-ref

    # voice-cloned from a dataset row (tokens + text, no audio download needed)
    python scripts/10_generate_samples.py --model Tilas/neutts-air-darija-v1 --ref-row 42

    # your own sentences
    python scripts/10_generate_samples.py --model ... --text "شنو كتدير هنا؟" "الله يعطيك الصحة"

Training used no reference prefix, so --no-ref is the in-distribution case and the
cleanest read on whether the model learned Darija phonetics. Reference conditioning
exercises the cloning behaviour inherited from the base model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.darija_g2p import DarijaPhonemizer, to_phones  # noqa: E402

REPO_DATA = "Tilas/MoulSot-Tokens-v1"
PREFIX_TEXT = "user: Convert the text to speech:"
INFIX_TEXT = "\nassistant:"

DEFAULT_SENTENCES = [
    "شنو كتدير هنا؟",
    "الله يعطيك الصحة أخويا",
    "غادي نمشي للدار دابا",
    "بزاف ديال الناس كايقولو هاد الشي",
    "واخا، صافي، غدا نتلاقاو.",
]


def build_prompt(tok, g2p, text, ref_text=None, ref_codes=None):
    """Mirror NeuTTS._apply_chat_template for the 'phonemes' input format."""
    sid = tok.convert_tokens_to_ids
    base = sid("<|speech_0|>")

    phones = to_phones(g2p, text)
    if ref_text:
        phones = f"{to_phones(g2p, ref_text)} {phones}"

    ids = (
        tok.encode(PREFIX_TEXT, add_special_tokens=False)
        + [sid("<|TEXT_PROMPT_START|>")]
        + tok.encode(phones, add_special_tokens=False)
        + [sid("<|TEXT_PROMPT_END|>")]
        + tok.encode(INFIX_TEXT, add_special_tokens=False)
        + [sid("<|SPEECH_GENERATION_START|>")]
    )
    if ref_codes is not None:
        ids += [base + int(c) for c in ref_codes]
    return ids, base, phones


def pick_device(requested: str | None, requested_dtype: str | None):
    """CUDA > MPS > CPU.

    NOT fp16. NeuTTS Air is Qwen2-architecture, and Qwen2 activations routinely exceed
    fp16's 65504 ceiling -> inf logits -> nan probabilities -> multinomial raises. The
    model was trained in fp32 under bf16 autocast, so bf16 (same exponent range as fp32)
    is the faithful choice and half the memory of fp32.
    """
    if requested:
        dev = requested
    elif torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    if requested_dtype:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[requested_dtype]
    else:
        dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    return dev, dtype


def logits_are_finite(model, ids, device) -> bool:
    """One forward pass to catch dtype range problems before generate() does."""
    try:
        with torch.no_grad():
            out = model(torch.tensor([ids[:64]], dtype=torch.long, device=device))
        return bool(torch.isfinite(out.logits).all().item())
    except Exception as exc:  # noqa: BLE001
        print(f"  probe forward failed ({type(exc).__name__}: {exc})")
        return False


def decode_to_wav(codec, codes, device):
    """neucodec.decode_code wants a LongTensor; be tolerant about the exact rank."""
    t = torch.tensor(codes, dtype=torch.long, device=device)
    for shaped in (t.view(1, 1, -1), t.view(1, -1), t):
        try:
            with torch.no_grad():
                wav = codec.decode_code(shaped)
            return wav.squeeze().float().cpu().numpy()
        except Exception:  # noqa: BLE001, PERF203
            continue
    raise RuntimeError("decode_code rejected every shape tried; check the neucodec API")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--text", nargs="*", default=None)
    ap.add_argument("--no-ref", action="store_true", help="no reference voice prefix")
    ap.add_argument("--ref-row", type=int, default=7,
                    help="row of MoulSot-Tokens-v1 to use as the reference voice")
    ap.add_argument("--max-new-tokens", type=int, default=900, help="50 = 1.0 s")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                    help="default: cuda > mps > cpu. Force cpu if MPS misbehaves.")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None,
                    help="default: bf16 on gpu, fp32 on cpu. float16 WILL produce nan "
                         "on this Qwen2-based model.")
    args = ap.parse_args()

    import soundfile as sf
    from neucodec import NeuCodec
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = pick_device(args.device, args.dtype)
    # NeuCodec decodes through a Vocos head that hits unimplemented Metal ops; it is
    # fast enough on CPU, so only follow the model onto the GPU when that GPU is CUDA.
    codec_device = device if device == "cuda" else "cpu"
    print(f"device: {device} ({dtype}), codec on {codec_device}")
    if device != "cuda":
        print("  note: expect roughly 10-60 s per sample rather than 1-2 s")
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    # Training set use_cache=False for gradient checkpointing and it persisted into the
    # saved config. Generation without a KV cache is quadratic - turn it back on.
    model.config.use_cache = True
    model.eval()

    # Verify the dtype actually holds this model's activation range before generating.
    probe = tok.encode("user: Convert the text to speech:", add_special_tokens=False)
    if not logits_are_finite(model, probe, device):
        print(f"  non-finite logits in {dtype}; reloading in float32")
        del model
        if device == "mps":
            torch.mps.empty_cache()
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        model.config.use_cache = True
        model.eval()
        if not logits_are_finite(model, probe, device):
            raise SystemExit(
                f"Logits are non-finite even in float32 on {device}. Retry with "
                "--device cpu, and if that also fails the checkpoint itself is suspect."
            )
    print(f"  logits finite in {dtype}")

    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to(codec_device)
    g2p = DarijaPhonemizer()

    speech_end = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    ref_text = ref_codes = None
    if not args.no_ref:
        from datasets import load_dataset

        row = load_dataset(REPO_DATA, split="train")[args.ref_row]
        ref_text, ref_codes = row["text"], row["tokens"]
        print(f"reference row {args.ref_row}: {row['audio_id']}  "
              f"{len(ref_codes)} codes ({len(ref_codes) / 50:.1f} s)")
        print(f"  text: {ref_text}")

    sentences = args.text or DEFAULT_SENTENCES
    manifest = []

    for i, text in enumerate(sentences):
        ids, base, phones = build_prompt(tok, g2p, text, ref_text, ref_codes)
        n_prompt = len(ids)
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                inp,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=50,        # matches NeuTTS._infer_torch
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=speech_end,
                pad_token_id=tok.pad_token_id,
            )

        new = out[0, n_prompt:].tolist()
        # Keep only real speech tokens, stopping at the end marker.
        codes = []
        for t in new:
            if t == speech_end:
                break
            if base <= t < base + 65536:
                codes.append(t - base)

        if len(codes) < 25:
            print(f"[{i}] {text}\n    only {len(codes)} codes generated - skipping")
            continue

        wav = decode_to_wav(codec, codes, codec_device)
        path = args.out / f"{i:02d}_{'noref' if args.no_ref else 'ref'}.wav"
        sf.write(path, wav, 24000)

        dur = len(codes) / 50
        print(f"[{i}] {text}")
        print(f"    phones: {phones[-90:] if len(phones) > 90 else phones}")
        print(f"    {len(codes)} codes -> {dur:.1f} s  ->  {path}")
        manifest.append({"text": text, "phones": phones, "n_codes": len(codes),
                         "duration_s": round(dur, 2), "file": str(path)})

    (args.out / "manifest.json").write_text(
        json.dumps({"model": args.model, "ref_row": None if args.no_ref else args.ref_row,
                    "samples": manifest}, ensure_ascii=False, indent=2)
    )
    print(f"\nwrote {len(manifest)} wav files to {args.out}")
    print("Listen before deciding whether to train longer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())