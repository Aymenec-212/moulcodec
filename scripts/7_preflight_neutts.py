#!/usr/bin/env python3
"""Pre-flight checks for NeuTTS Air fine-tuning on MoulSot-Tokens-v1.

Answers, empirically, every number the training script needs to be written correctly.
Nothing here trains; the expensive part is one optional 3-step VRAM probe.

    python scripts/7_preflight_neutts.py                      # checks only
    python scripts/7_preflight_neutts.py --vram-probe         # + 3 real training steps
    python scripts/7_preflight_neutts.py --n-text-samples 8000

Writes preflight_neutts.json next to the report. Exit code is nonzero if any hard
check fails, so it is safe to chain before a training launch.
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Python puts THIS FILE's directory on sys.path[0], not the CWD, so `src` is not
# importable when this is run as `python scripts/7_preflight_neutts.py`. Adding the
# repo root explicitly makes the script work from any working directory. `python -m`
# is not an option here: the leading digit makes this an invalid module name.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.darija_g2p import (
        USING_NEUTTS_BASE,
        DarijaPhonemizer,
        filter_reason,
        to_phones,
    )
except ModuleNotFoundError as exc:
    if "darija_g2p" not in str(exc) and "src" not in str(exc):
        raise
    sys.exit(
        f"Could not import src.darija_g2p (looked under {REPO_ROOT}).\n"
        f"Expected file: {REPO_ROOT / 'src' / 'darija_g2p.py'}\n"
        f"Exists: {(REPO_ROOT / 'src' / 'darija_g2p.py').exists()}\n"
        "Place darija_g2p.py in src/ and rerun."
    )

REPO_MODEL = "neuphonic/neutts-air"
REPO_DATA = "Tilas/MoulSot-Tokens-v1"
CODEBOOK_SIZE = 65536

SPECIAL_TOKENS = [
    "<|TEXT_REPLACE|>",
    "<|TEXT_PROMPT_START|>",
    "<|TEXT_PROMPT_END|>",
    "<|SPEECH_REPLACE|>",
    "<|SPEECH_GENERATION_START|>",
    "<|SPEECH_GENERATION_END|>",
]

# Exactly the template examples/finetune.py builds, with the payload removed.
TEMPLATE_SHELL = (
    "user: Convert the text to speech:<|TEXT_PROMPT_START|><|TEXT_PROMPT_END|>\n"
    "assistant:<|SPEECH_GENERATION_START|><|SPEECH_GENERATION_END|>"
)

results: dict = {}
failures: list[str] = []
warnings_: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    warnings_.append(msg)


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ======================================================================================
# 1. Environment
# ======================================================================================


def check_environment() -> None:
    section("1. ENVIRONMENT")

    import torch
    import transformers

    print(f"  python       {sys.version.split()[0]}")
    print(f"  torch        {torch.__version__}   cuda={torch.cuda.is_available()}")
    print(f"  transformers {transformers.__version__}")
    results["torch"] = torch.__version__
    results["transformers"] = transformers.__version__

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"  gpu          {props.name}  {total_gb:.1f} GiB")
        results["gpu"] = {"name": props.name, "total_gib": round(total_gb, 1)}

    # transformers 5.x removed Trainer(tokenizer=...) in favour of processing_class=.
    # examples/finetune.py still passes tokenizer=, so it TypeErrors on the version
    # the neutts package itself pins (5.1.0).
    from transformers import Trainer

    params = inspect.signature(Trainer.__init__).parameters
    results["trainer_kwarg"] = "tokenizer" if "tokenizer" in params else "processing_class"
    if "tokenizer" in params:
        ok("Trainer accepts tokenizer= (transformers 4.x style)")
    elif "processing_class" in params:
        warn("Trainer requires processing_class=, NOT tokenizer= "
             "(upstream finetune.py would TypeError here)")
    else:
        fail("Trainer accepts neither tokenizer= nor processing_class=")

    # Phonemizer parity: bundled espeak-ng vs system espeak-ng.
    ph = DarijaPhonemizer()
    version = getattr(ph, "espeak_version", None)
    results["espeak_version"] = str(version)
    results["espeak_bundled"] = bool(USING_NEUTTS_BASE)
    if USING_NEUTTS_BASE:
        ok(f"using neutts BasePhonemizer, espeak-ng {version}")
    else:
        warn(f"using SYSTEM espeak-ng {version}; install neutts before generating "
             "training data or train/inference phonemes will diverge")

    # neucodec must import under this transformers version (it broke above 4.47 before).
    try:
        from neucodec import NeuCodec  # noqa: F401

        ok("neucodec imports cleanly")
        results["neucodec_import"] = True
    except Exception as exc:  # noqa: BLE001
        warn(f"neucodec import failed ({type(exc).__name__}: {exc}); needed to decode "
             "generated tokens to audio, not to train")
        results["neucodec_import"] = False

    return ph


# ======================================================================================
# 2. Model and tokenizer
# ======================================================================================


def check_model(load_weights: bool) -> tuple:
    section("2. MODEL / TOKENIZER")

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO_MODEL)
    cfg = AutoConfig.from_pretrained(REPO_MODEL)

    vocab = len(tok)
    tied = getattr(cfg, "tie_word_embeddings", None)
    hidden = getattr(cfg, "hidden_size", None)
    print(f"  vocab_size (len(tokenizer))   {vocab:,}")
    print(f"  config.vocab_size             {getattr(cfg, 'vocab_size', None):,}")
    print(f"  hidden_size                   {hidden}")
    print(f"  tie_word_embeddings           {tied}")
    results["vocab_size"] = vocab
    results["hidden_size"] = hidden
    results["tie_word_embeddings"] = tied

    if tied is False:
        ok("embeddings untied -> lm_head is a separate vocab x hidden matrix "
           "(this is what makes the logits tensor the VRAM bottleneck)")

    # config.neuphonic.input_format decides which chat template inference uses.
    neuphonic_cfg = getattr(cfg, "neuphonic", None) or {}
    fmt = neuphonic_cfg.get("input_format", "phonemes")
    results["input_format"] = fmt
    if fmt == "phonemes":
        ok("config.neuphonic.input_format = 'phonemes' (matches our espeak plan)")
    else:
        fail(f"input_format is '{fmt}', not 'phonemes'; the training template must match")

    # --- special tokens -----------------------------------------------------------
    missing = [t for t in SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    if missing:
        fail(f"special tokens missing from vocab: {missing}")
    else:
        ok(f"all {len(SPECIAL_TOKENS)} NeuTTS special tokens present")

    # --- speech token contiguity --------------------------------------------------
    # If contiguous we can map code -> id by arithmetic instead of running the tokenizer
    # over 18.3M added-token strings, which is the difference between minutes and hours.
    names = [f"<|speech_{i}|>" for i in range(CODEBOOK_SIZE)]
    ids = tok.convert_tokens_to_ids(names)
    base = ids[0]
    contiguous = ids == list(range(base, base + CODEBOOK_SIZE))
    results["speech_token_base_id"] = base
    results["speech_tokens_contiguous"] = contiguous
    if contiguous:
        ok(f"<|speech_i|> ids contiguous from {base} to {base + CODEBOOK_SIZE - 1} "
           f"-> use id = {base} + code")
    else:
        bad = [i for i, v in enumerate(ids) if v != base + i][:5]
        fail(f"speech token ids are NOT contiguous (first offenders: {bad}); "
             "must tokenize the code string instead of using arithmetic")

    if tok.unk_token_id in ids:
        fail("some <|speech_i|> tokens resolve to unk")

    # A speech token must encode as exactly one id, not be split into BPE pieces.
    probe = tok.encode("<|speech_0|><|speech_65535|>", add_special_tokens=False)
    if probe == [ids[0], ids[-1]]:
        ok("speech tokens encode 1:1 (no BPE splitting)")
    else:
        fail(f"speech token encoding split unexpectedly: {probe}")

    # --- pad token ----------------------------------------------------------------
    print(f"  pad_token / id                {tok.pad_token!r} / {tok.pad_token_id}")
    print(f"  eos_token / id                {tok.eos_token!r} / {tok.eos_token_id}")
    results["pad_token_id"] = tok.pad_token_id
    if tok.pad_token_id is None:
        fail("tokenizer has no pad_token; the collator needs one")

    # --- template overhead --------------------------------------------------------
    overhead = len(tok.encode(TEMPLATE_SHELL))
    results["template_overhead_tokens"] = overhead
    ok(f"chat template overhead = {overhead} tokens")

    # Exact parameter count without downloading ~1.5 GB of weights: instantiate on the
    # meta device, where tensors carry shape but no storage.
    n_params = None
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            meta = AutoModelForCausalLM.from_config(cfg)
        n_params = sum(p.numel() for p in meta.parameters())
        del meta
        results["n_params"] = n_params
        ok(f"{n_params / 1e6:.1f}M parameters (meta device, nothing downloaded)")
    except Exception as exc:  # noqa: BLE001
        warn(f"meta-device parameter count failed: {exc}")

    # --- tied or untied, really? ---------------------------------------------------
    # The meta-device count above reflects config.tie_word_embeddings. If the config
    # says tied but the checkpoint stores a separate lm_head.weight (or vice versa),
    # the effective trainable parameter count - and ~3 GiB of AdamW state - changes.
    # get_safetensors_metadata reads only the file header, so this costs no download.
    n_params_effective = n_params
    try:
        from huggingface_hub import get_safetensors_metadata

        meta_st = get_safetensors_metadata(REPO_MODEL)
        weight_map = getattr(meta_st, "weight_map", {}) or {}
        has_lm_head = "lm_head.weight" in weight_map
        stored = getattr(meta_st, "parameter_count", None)
        stored_total = sum(stored.values()) if isinstance(stored, dict) else None
        results["checkpoint_has_lm_head"] = has_lm_head
        results["checkpoint_param_count"] = stored_total
        if stored_total:
            print(f"  params stored in checkpoint    {stored_total / 1e6:.1f}M")
        print(f"  separate lm_head.weight        {has_lm_head}")

        if tied and has_lm_head:
            warn("config.tie_word_embeddings=True BUT the checkpoint stores a separate "
                 "lm_head.weight - from_pretrained will tie them on load, so effective "
                 "trainable params are lower than the meta-device count suggests")
            n_params_effective = n_params - (vocab * hidden)
        elif tied and not has_lm_head:
            ok("tied, and no separate lm_head in the checkpoint (self-consistent)")
            if n_params and abs(n_params - (n_params - vocab * hidden)) > 0:
                n_params_effective = n_params - (vocab * hidden)
        else:
            ok("untied - lm_head is a genuine second matrix")
    except Exception as exc:  # noqa: BLE001
        warn(f"could not read safetensors header ({exc}); VRAM estimate assumes "
             f"{(n_params or 0) / 1e6:.1f}M trainable params")

    if n_params_effective != n_params:
        ok(f"effective trainable params for VRAM sizing: "
           f"{n_params_effective / 1e6:.1f}M (vs {n_params / 1e6:.1f}M counted)")
    results["n_params_effective"] = n_params_effective

    model = None
    if load_weights:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(REPO_MODEL, dtype=torch.bfloat16)
        ok("weights loaded (only needed for --vram-probe)")

    return tok, overhead, model, n_params_effective


# ======================================================================================
# 3. Dataset
# ======================================================================================


def check_dataset(tok, phonemizer, overhead: int, n_text_samples: int) -> None:
    section("3. DATASET")

    from datasets import load_dataset

    ds = load_dataset(REPO_DATA, split="train")
    n = len(ds)
    print(f"  rows: {n:,}   columns: {ds.column_names}")
    results["n_rows"] = n
    results["columns"] = ds.column_names

    if "codes" not in ds.column_names and "tokens" in ds.column_names:
        ok("column is 'tokens' (upstream expects 'codes' - rename in preprocessing)")

    # --- frame formula holds on every row (cheap, no audio needed) -----------------
    # Lengths come from Arrow list offsets. `ds["tokens"]` would materialise all
    # 18.3M codes as Python int objects (~1 GB resident), which is the difference
    # between fine and swapping on an 8 GB laptop.
    n_samples = ds["n_samples"]
    lengths: list[int] = []
    try:
        import pyarrow.compute as pc

        for batch in ds.select_columns(["tokens"]).with_format("arrow").iter(batch_size=8192):
            lengths.extend(pc.list_value_length(batch["tokens"]).to_pylist())
    except Exception as exc:  # noqa: BLE001
        warn(f"Arrow fast path unavailable ({exc}); scanning row by row instead")
        lengths = [len(row["tokens"]) for row in ds]
    mismatches = sum(1 for ns, ln in zip(n_samples, lengths) if ns // 320 != ln)
    if mismatches == 0:
        ok(f"n_samples // 320 == len(tokens) on {n:,}/{n:,} rows")
    else:
        fail(f"frame formula violated on {mismatches:,} rows")
    results["frame_formula_mismatches"] = mismatches

    # --- speech-token length distribution -----------------------------------------
    qs = statistics.quantiles(lengths, n=100)
    dist = {
        "min": min(lengths), "p50": qs[49], "p90": qs[89],
        "p95": qs[94], "p99": qs[98], "max": max(lengths),
    }
    results["speech_token_lengths"] = dist
    print("\n  speech tokens per clip (50 = 1.0 s):")
    for k, v in dist.items():
        print(f"    {k:>4} {int(v):>6}   ({int(v) / 50:>6.1f} s)")

    # --- character inventory ------------------------------------------------------
    # Tells us what actually needs normalising; drives _PUNCT_MAP in darija_g2p.py.
    # Even coverage across the WHOLE corpus. `range(0, n, n // k)` looks equivalent but
    # stops ~n/k short of the end: at n=79,641 / k=4,000 it reaches row 75,981 and misses
    # the last 4.6%. The release is written shortest-first within exact-length groups, so
    # that tail is precisely where the longest clips are - which silently reported zero
    # too_long rejections and understated the sequence-length percentiles.
    k = min(n_text_samples, n)
    sample_idx = [(i * n) // k for i in range(k)]
    sample_texts = ds.select(sample_idx)["text"]
    chars = Counter()
    diacritics = 0
    for t in sample_texts:
        if not t:
            continue
        chars.update(t)
        diacritics += sum(1 for c in t if unicodedata.category(c) == "Mn")
    non_letter = {
        c: k for c, k in chars.most_common()
        if not unicodedata.category(c).startswith("L") and not c.isspace()
    }
    print(f"\n  non-letter characters in {len(sample_texts):,} sampled transcripts:")
    for c, k in list(non_letter.items())[:20]:
        try:
            name = unicodedata.name(c)
        except ValueError:
            name = "?"
        print(f"    U+{ord(c):04X} {c!r:>8} x{k:<8} {name}")
    results["diacritic_marks_in_sample"] = diacritics
    results["non_letter_chars"] = {f"U+{ord(c):04X}": k for c, k in list(non_letter.items())[:40]}
    if diacritics:
        warn(f"{diacritics:,} combining diacritics in sample - espeak uses harakat for "
             "vowelisation, so partial diacritisation makes G2P inconsistent across rows")

    # --- phonemize a sample, measure real sequence lengths ------------------------
    print(f"\n  phonemizing {len(sample_texts):,} transcripts...")
    n_sampled = len(sample_texts)
    totals, phone_lens, empty_phones = [], [], 0
    reasons = Counter()

    # Provisional cap: p99 speech length. Refined below once totals are known.
    provisional_cap = int(dist["p99"])

    for pos, text in enumerate(sample_texts):
        n_tok = lengths[sample_idx[pos]]
        reason = filter_reason(text, n_tok, max_speech_tokens=provisional_cap)
        if reason:
            reasons[reason] += 1
            continue
        phones = to_phones(phonemizer, text)
        if not phones.strip():
            empty_phones += 1
            reasons["empty_phonemes"] += 1
            continue
        p_len = len(tok.encode(phones, add_special_tokens=False))
        phone_lens.append(p_len)
        totals.append(overhead + p_len + n_tok)

    kept = len(totals)
    print(f"\n  filter outcome on {n_sampled:,} sampled rows:")
    print(f"    kept                 {kept:,}  ({100 * kept / n_sampled:.1f}%)")
    for reason, k in reasons.most_common():
        print(f"    {reason:<20} {k:,}")
    results["filter"] = {"sampled": n_sampled, "kept": kept, "rejected": dict(reasons)}
    results["empty_phonemes"] = empty_phones

    if kept / max(n_sampled, 1) < 0.5:
        warn(f"filter keeps only {100 * kept / n_sampled:.1f}% of rows - inspect the "
             "rejection breakdown before accepting this")

    if not totals:
        fail("no rows survived the filter; cannot recommend max_seq_len")
        return

    tq = statistics.quantiles(totals, n=100)
    pq = statistics.quantiles(phone_lens, n=100)
    results["phone_token_lengths"] = {"p50": pq[49], "p95": pq[94], "p99": pq[98]}
    results["total_seq_lengths"] = {
        "p50": tq[49], "p90": tq[89], "p95": tq[94], "p99": tq[98], "max": max(totals),
    }

    print("\n  phoneme tokens (text side):")
    print(f"    p50 {int(pq[49]):>5}   p95 {int(pq[94]):>5}   p99 {int(pq[98]):>5}")
    print("\n  TOTAL sequence length (overhead + phones + speech):")
    for label, v in [("p50", tq[49]), ("p90", tq[89]), ("p95", tq[94]),
                     ("p99", tq[98]), ("max", max(totals))]:
        print(f"    {label:>4} {int(v):>6}")

    # Round p99 up to a multiple of 64 for tensor-core friendliness.
    rec = int(((tq[98] + 63) // 64) * 64)
    results["recommended_max_seq_len"] = rec
    covered = 100 * sum(1 for t in totals if t <= rec) / len(totals)
    print(f"\n  recommended max_seq_len = {rec}  (covers {covered:.1f}% of kept rows)")
    print(f"  upstream default is 2048 -> {100 * (1 - tq[49] / 2048):.0f}% of every "
          "sequence would be padding, and pad positions carry loss in finetune.py")

    hours = sum(lengths) / 50 / 3600
    print(f"\n  corpus: {hours:.1f} h of speech, {sum(lengths):,} speech tokens")
    results["total_hours"] = round(hours, 1)


# ======================================================================================
# 4. VRAM
# ======================================================================================


def estimate_vram(n_params: int, vocab: int, recommended_seq: int) -> None:
    """Analytic estimate, runnable without a GPU.

    Two terms dominate and both are exactly determined by n_params and vocab:

      optimizer state  n_params x 16 B   (fp32 master + grad + Adam m + v)
      logits           B x L x V x 10 B  (bf16 logits + fp32 CE copy + fp32 grad)

    Activations sit on top and depend on gradient checkpointing, so treat these as a
    floor, not a budget. The --vram-probe on a CUDA box is the number that decides.
    """
    section("4. VRAM ESTIMATE (analytic - no GPU required)")

    if not n_params:
        warn("parameter count unavailable; skipping estimate")
        return

    state = n_params * 16 / 1024**3
    print(f"  optimizer state (AdamW, fp32): {state:.2f} GiB   [independent of seq/batch]")
    print(f"  vocab = {vocab:,}  ->  logits cost {vocab * 10 / 1024**3 * 1024:.2f} MiB "
          "per 1k positions\n")

    seq_lens = sorted({256, 512, recommended_seq or 512, 1024, 2048})
    print(f"  {'seq_len':>8} {'batch':>6} {'logits':>9} {'floor':>9}   vs 24 GiB 3090")
    grid = []
    for L in seq_lens:
        for B in (1, 2, 4):
            logits = B * L * vocab * 10 / 1024**3
            floor = state + logits
            if floor > 22:
                verdict = "OOM"
            elif floor > 17:
                verdict = "tight"
            else:
                verdict = "ok"
            marker = "  <- recommended" if L == recommended_seq and B == 2 else ""
            print(f"  {L:>8} {B:>6} {logits:>8.2f}G {floor:>8.2f}G   {verdict}{marker}")
            grid.append({"seq_len": L, "batch": B, "floor_gib": round(floor, 2),
                         "verdict": verdict})
    results["vram_estimate"] = {"optimizer_state_gib": round(state, 2), "grid": grid}
    print("\n  Activations add on top. Use gradient checkpointing and confirm with "
          "--vram-probe on the GPU box.")


def vram_probe(model, tok, seq_len: int, batch_size: int, grad_checkpointing: bool) -> None:
    section(f"5. VRAM PROBE  (seq_len={seq_len}, batch_size={batch_size}, "
            f"grad_checkpointing={grad_checkpointing})")

    import torch

    if not torch.cuda.is_available():
        warn("no CUDA device - probe skipped. Everything above is still valid; only "
             "this measured number has to wait for a GPU box. MPS is not a substitute: "
             "a full fine-tune of this model does not fit in 8 GB of unified memory, "
             "and MPS memory accounting would not transfer to the 3090 anyway.")
        return
    if model is None:
        warn("model not loaded (--no-weights); skipping VRAM probe")
        return

    model = model.to("cuda")
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()

    vocab = len(tok)
    for step in range(3):
        ids = torch.randint(0, vocab, (batch_size, seq_len), device="cuda")
        labels = ids.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  step {step}: loss {loss.item():.3f}   peak {peak:.2f} GiB")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    results["vram_probe"] = {
        "seq_len": seq_len, "batch_size": batch_size,
        "grad_checkpointing": grad_checkpointing,
        "peak_gib": round(peak, 2), "total_gib": round(total, 1),
    }
    headroom = total - peak
    if headroom < 2.0:
        warn(f"peak {peak:.2f} GiB of {total:.1f} GiB - under 2 GiB headroom, "
             "expect OOM under fragmentation")
    else:
        ok(f"peak {peak:.2f} GiB of {total:.1f} GiB ({headroom:.1f} GiB headroom)")


# ======================================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-text-samples", type=int, default=4000,
                    help="transcripts to phonemize for length/filter stats")
    ap.add_argument("--no-weights", action="store_true",
                    help="skip loading model weights (config/tokenizer only)")
    ap.add_argument("--vram-probe", action="store_true")
    ap.add_argument("--probe-seq-len", type=int, default=0,
                    help="0 = use the recommended max_seq_len from section 3")
    ap.add_argument("--probe-batch-size", type=int, default=2)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("preflight_neutts.json"))
    args = ap.parse_args()

    phonemizer = check_environment()
    tok, overhead, model, n_params = check_model(load_weights=not args.no_weights)
    check_dataset(tok, phonemizer, overhead, args.n_text_samples)

    estimate_vram(n_params, len(tok), results.get("recommended_max_seq_len", 0))

    if args.vram_probe:
        seq = args.probe_seq_len or results.get("recommended_max_seq_len", 512)
        vram_probe(model, tok, seq, args.probe_batch_size, not args.no_grad_checkpointing)

    section("SUMMARY")
    results["failures"] = failures
    results["warnings"] = warnings_
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  wrote {args.out}")
    print(f"  {len(failures)} failure(s), {len(warnings_)} warning(s)")
    for f in failures:
        print(f"    FAIL: {f}")
    for w in warnings_:
        print(f"    WARN: {w}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())