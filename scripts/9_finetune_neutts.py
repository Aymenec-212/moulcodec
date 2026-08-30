#!/usr/bin/env python3
"""Fine-tune NeuTTS Air on Moroccan Darija speech tokens.

Consumes scripts/8_prepare_neutts_data.py output. GPU required.

    python scripts/9_finetune_neutts.py --data data/neutts_darija --inspect
    python scripts/9_finetune_neutts.py --data data/neutts_darija --max-steps 30 \
        --out /tmp/smoke --check-tie --no-push
    python scripts/9_finetune_neutts.py --data data/neutts_darija \
        --epochs 2 --hub-repo Tilas/neutts-air-darija-v1

Differences from upstream examples/finetune.py, all measured rather than assumed:

  1. processing_class=, not tokenizer=  - removed from Trainer in transformers 5.x,
     which is what the neutts package itself pins.
  2. Dynamic padding to the longest sequence in each batch, not a fixed 2048. With
     group_by_length this cuts padding overhead from 62% to ~1%.
  3. Pad positions get label -100. Upstream sets labels[speech_start:] = input_ids[...]
     including padding, so most of the loss lands on pad tokens.
  4. No truncation. Upstream does ids[:max_len], which cuts the code stream and drops
     <|SPEECH_GENERATION_END|>, teaching the model never to stop. Over-long samples are
     filtered in step 8 instead.
  5. torch_compile off by default - dynamic shapes would trigger constant recompiles.

Everything that could waste GPU hours is checked BEFORE the model loads: CUDA, bf16
support, free disk for checkpoints, and Hub write access. A pod without a network volume
is destroyed when credits hit zero, so discovering a bad token after hours of training
would lose the run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import TrainerCallback

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_MODEL = "neuphonic/neutts-air"
REPO_DATA = "Tilas/MoulSot-Tokens-v1"
IGNORE_INDEX = -100


# ======================================================================================
# Collator
# ======================================================================================


@dataclass
class NeuTTSCollator:
    """Pad to the longest sequence in the batch; supervise only the speech span."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            longest = ((longest + m - 1) // m) * m

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            start = int(f["speech_start_idx"])
            n_pad = longest - len(ids)

            input_ids.append(ids + [self.pad_token_id] * n_pad)
            # Supervise from <|SPEECH_GENERATION_START|> onward. The phoneme prompt is
            # masked (we are not training a text generator) and so is the padding - the
            # stop signal is <|SPEECH_GENERATION_END|>, which sits inside the real
            # sequence, so masking pads costs nothing.
            labels.append([IGNORE_INDEX] * start + ids[start:] + [IGNORE_INDEX] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def inspect_batch(ds, tok, collator, n: int = 4) -> None:
    """Print one collated batch. Cheapest possible check that labels are right."""
    batch = collator([ds[i] for i in range(n)])
    print("\n" + "=" * 78 + "\nBATCH INSPECTION\n" + "=" * 78)
    for k, v in batch.items():
        print(f"  {k:<16} {tuple(v.shape)}  {v.dtype}")

    ids, labels, attn = batch["input_ids"][0], batch["labels"][0], batch["attention_mask"][0]
    supervised = labels != IGNORE_INDEX
    first = int(supervised.nonzero()[0])
    real = int(attn.sum())

    print(f"\n  sample 0: {real} real tokens, {len(ids) - real} pad")
    print(f"  supervision starts at index {first} "
          f"({int(supervised.sum())} of {len(labels)} positions)")
    print(f"  token at that index: {tok.convert_ids_to_tokens([int(ids[first])])[0]}")
    print(f"\n  prompt (masked):\n    {tok.decode(ids[:first])!r}")
    print(f"  first 6 supervised: {tok.convert_ids_to_tokens(ids[first:first + 6].tolist())}")
    print(f"  last 3 real:        {tok.convert_ids_to_tokens(ids[real - 3:real].tolist())}")

    assert labels[:first].eq(IGNORE_INDEX).all(), "prompt is not fully masked"
    assert labels[real:].eq(IGNORE_INDEX).all(), "padding is not masked"
    assert torch.equal(labels[first:real], ids[first:real]), "supervised span mismatch"
    print("\n  [ok] prompt masked, padding masked, speech span supervised")


class MemoryCallback(TrainerCallback):
    """Report peak VRAM early, so an OOM surfaces in seconds not hours."""

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step in (1, 5, 20, 50, 100):
            peak = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  [mem] step {state.global_step}: peak {peak:.2f} / {total:.1f} GiB")


# ======================================================================================
# Pre-flight - everything that could waste GPU hours, checked before loading weights
# ======================================================================================


def preflight(args, n_train: int) -> dict:
    print("=" * 78 + "\nPRE-FLIGHT\n" + "=" * 78)
    facts = {}

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This script needs a GPU.")
    props = torch.cuda.get_device_properties(0)
    print(f"  gpu            {props.name}  {props.total_memory / 1024**3:.1f} GiB")
    facts["gpu"] = props.name

    # bf16 needs Ampere or newer. On Turing/Volta (T4, V100) fall back to fp16.
    use_bf16 = torch.cuda.is_bf16_supported()
    if not use_bf16:
        print("  [WARN] bf16 unsupported on this GPU; falling back to fp16 + grad scaler")
    print(f"  precision      {'bf16' if use_bf16 else 'fp16'}")
    facts["precision"] = "bf16" if use_bf16 else "fp16"

    # A Trainer checkpoint is weights (fp32) + AdamW m,v (fp32) ~= params x 12 bytes.
    per_ckpt = 553e6 * 12 / 1024**3
    need = per_ckpt * args.save_total_limit + 6  # + final model, hub cache, model cache
    args.out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out).free / 1024**3
    print(f"  disk free      {free:.1f} GiB  (need ~{need:.1f} GiB for "
          f"{args.save_total_limit} checkpoints + final model)")
    if free < need:
        raise SystemExit(
            "Not enough disk. Increase the pod's container disk, lower "
            "--save-total-limit, or point --out at a larger volume."
        )
    facts["disk_free_gib"] = round(free, 1)

    # Hub write access, checked NOW rather than after training.
    if args.hub_repo and not args.no_push:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            who = api.whoami()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Not logged in to the Hub ({exc}). Run: hf auth login")
        print(f"  hub user       {who.get('name')}")
        try:
            url = api.create_repo(args.hub_repo, exist_ok=True, private=args.hub_private)
            print(f"  hub repo       {url} (write access confirmed)")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Cannot create or write {args.hub_repo}: {exc}\n"
                "Check the token has write scope and the namespace is yours."
            )
        facts["hub_repo"] = args.hub_repo

    eff = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, n_train // eff)
    total = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    print(f"  effective batch {eff}   steps/epoch {steps_per_epoch:,}   total steps {total:,}")
    facts.update(effective_batch=eff, steps_per_epoch=steps_per_epoch, total_steps=total)
    return facts


# ======================================================================================
# Model card
# ======================================================================================


def build_model_card(args, meta: dict, facts: dict, metrics: dict) -> str:
    kept = meta.get("n_train", 0) + meta.get("n_eval", 0)
    eval_loss = metrics.get("eval_loss")
    eval_loss = f"{eval_loss:.4f}" if isinstance(eval_loss, float) else "n/a"
    return f"""---
license: apache-2.0
base_model: {args.model}
datasets:
  - {REPO_DATA}
language:
  - ary
  - ar
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - darija
  - moroccan-arabic
  - neucodec
  - neutts
---

# NeuTTS Air - Moroccan Darija

`{args.model}` fine-tuned on Moroccan Darija speech tokens from
[`{REPO_DATA}`](https://huggingface.co/datasets/{REPO_DATA}).

Speech is represented as NeuCodec tokens (single codebook, 65,536 codes, 50 tokens/s,
0.8 kbps). Generated tokens must be decoded with `neuphonic/neucodec`.

## Usage

Darija text is phonemised with espeak-ng's `ar` voice through a custom phonemizer that
maps Arabic punctuation to Latin equivalents (espeak silently drops `؟` and `،`) and
strips harakat. **The same phonemizer must be used at inference**, or the model sees a
different phoneme distribution than it was trained on.

```python
import src.darija_g2p          # registers DarijaPhonemizer as CUSTOM_PHONEMIZERS["ar"]
from neutts import NeuTTS

tts = NeuTTS(backbone_repo="{args.hub_repo}", language="ar")
wav = tts.infer("شنو كتدير هنا؟", ref_codes=ref_codes, ref_text=ref_text)
```

`language="ar"` must be passed explicitly: this repo is not in `BACKBONE_LANGUAGE_MAP`,
and `_load_phonemizer` raises rather than guessing.

## Training

| | |
|---|---|
| Base model | `{args.model}` ({facts.get('trainable_m', '?')}M trainable params) |
| Training samples | {meta.get('n_train', 0):,} |
| Eval samples | {meta.get('n_eval', 0):,} |
| Speech | ~94.8 h after filtering (101.7 h before) |
| Max sequence length | {meta.get('max_seq_len', '?')} tokens |
| Effective batch | {facts.get('effective_batch', '?')} |
| Learning rate | {args.lr} (cosine, {args.warmup_ratio} warmup) |
| Epochs | {args.epochs} |
| Precision | {facts.get('precision', '?')} |
| GPU | {facts.get('gpu', '?')} |
| Final eval loss | {eval_loss} |

Supervision covers only the speech span; the phoneme prompt and padding are masked with
`-100`. Sequences are dynamically padded with length-grouped batching.

## Data filtering

{kept:,} of 79,641 source rows were kept.
Rejections: `{json.dumps(meta.get('rejects', {}), ensure_ascii=False)}`

Clips under 1.0 s were dropped to match `min_new_tokens=50` at inference. Clips
containing digits were dropped because espeak verbalises numerals into Modern Standard
Arabic while the audio contains the Darija form.

## Limitations

- **Phonemisation is a baseline, not a Darija G2P.** espeak-ng's `ar` voice applies MSA
  morphophonology and reads short vowels from diacritics that undiacritised Darija does
  not carry, producing MSA-flavoured output and vowel-less consonant clusters
  (`نبداو` → `mbdˈaːw`, `فالسبعة` → `fassbʕt`). It is deterministic and collision-free on
  the pairs tested, so the model can learn a consistent mapping, but the IPA is
  frequently wrong. A dedicated Darija phonemiser is the obvious next improvement.
- **French loanwords** get English-flavoured IPA under `language_switch="remove-flags"`.
- **Transcripts are Gemini-2.5-Pro generated**, not human-verified, and Darija has no
  standardised orthography.
- **Source audio is YouTube**, so acoustic conditions vary and background noise is
  learnable. Voice quality at inference depends heavily on the reference clip.
- **The eval split is a random sample of training channels**, so eval loss is optimistic.
  It is a training-health signal, not a quality benchmark.

## Provenance

Tokens were produced by `neuphonic/neucodec` with the encode path frozen, at
`batch_size=1` and with no pre-padding. Both are required for token stability, since
Wav2Vec2-BERT applies per-utterance feature normalisation and `encode_code` passes no
attention mask.
"""


# ======================================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/neutts-darija-v1"))
    ap.add_argument("--model", default=REPO_MODEL)

    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--torch-compile", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore checkpoints in --out and start fresh")

    ap.add_argument("--hub-repo", default="Tilas/neutts-air-darija-v1")
    ap.add_argument("--hub-private", action="store_true")
    ap.add_argument("--no-push", action="store_true")

    ap.add_argument("--inspect", action="store_true", help="show one batch and exit")
    ap.add_argument("--check-tie", action="store_true",
                    help="compare stored lm_head.weight against embed_tokens.weight")
    args = ap.parse_args()

    if args.save_steps % args.eval_steps != 0:
        raise SystemExit(
            f"--save-steps ({args.save_steps}) must be a multiple of --eval-steps "
            f"({args.eval_steps}) for load_best_model_at_end"
        )

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    # ---- data -----------------------------------------------------------------
    ds = load_from_disk(str(args.data))
    meta_path = args.data / "prepare_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    print(f"train {len(ds['train']):,}   eval {len(ds['test']):,}")

    tok = AutoTokenizer.from_pretrained(args.model)

    # If this tokenizer disagrees with the one used in step 8, every speech token is
    # silently wrong and training would produce noise with a perfectly plausible loss.
    base_now = tok.convert_tokens_to_ids("<|speech_0|>")
    base_then = meta.get("speech_token_base")
    if base_then is not None and base_now != base_then:
        raise SystemExit(
            f"speech token base mismatch: data built with {base_then}, "
            f"tokenizer for {args.model} gives {base_now}. Rebuild step 8."
        )
    print(f"speech token base {base_now} matches prepared data")

    collator = NeuTTSCollator(pad_token_id=tok.pad_token_id)

    if args.inspect:
        inspect_batch(ds["train"], tok, collator)
        return 0

    facts = preflight(args, len(ds["train"]))

    # ---- model ----------------------------------------------------------------
    # float32 weights + autocast keeps fp32 master weights, matching the VRAM estimate.
    # Loading in bf16 would make the optimizer track bf16 masters and is less stable.
    print(f"\nloading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    tied = model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    facts["trainable_m"] = round(trainable / 1e6, 1)
    print(f"  tied after load:  {tied}")
    print(f"  trainable params: {trainable / 1e6:.1f}M")

    if args.check_tie:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        try:
            sd = load_file(hf_hub_download(args.model, "model.safetensors"))
            if "lm_head.weight" in sd and "model.embed_tokens.weight" in sd:
                same = torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
                print(f"  stored lm_head == stored embed_tokens: {same}")
                facts["stored_lm_head_equals_embed"] = bool(same)
                if not same:
                    print("  WARNING: the checkpoint carries a distinct lm_head that "
                          "tie_word_embeddings=True discards on load. This affects every "
                          "transformers user of this model - worth reporting upstream.")
            del sd
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] tie check skipped: {exc}")

    # Inference reads config.neuphonic.input_format to pick the chat template.
    neuphonic = getattr(model.config, "neuphonic", None) or {}
    neuphonic.setdefault("input_format", "phonemes")
    model.config.neuphonic = neuphonic
    # Incompatible with gradient checkpointing; warns loudly every step otherwise.
    model.config.use_cache = False

    report_to = []
    try:
        import tensorboard  # noqa: F401

        report_to = ["tensorboard"]
    except ImportError:
        print("  [note] tensorboard not installed; logging to stdout only")

    use_bf16 = facts["precision"] == "bf16"
    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=not args.no_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,          # makes dynamic padding actually pay off
        length_column_name="length",
        remove_unused_columns=False,   # speech_start_idx must reach the collator
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        torch_compile=args.torch_compile,
        seed=args.seed,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        processing_class=tok,  # NOT tokenizer= - removed in transformers 5.x
        callbacks=[MemoryCallback()],
    )

    resume = None
    if not args.no_resume and any(args.out.glob("checkpoint-*")):
        resume = True
        print(f"\nresuming from the latest checkpoint in {args.out}")

    try:
        trainer.train(resume_from_checkpoint=resume)
    except torch.cuda.OutOfMemoryError:
        print(
            "\nCUDA OOM. Halve --batch-size and double --grad-accum to keep the same "
            f"effective batch. Checkpoints in {args.out} are intact; rerun the same "
            "command to resume.",
            file=sys.stderr,
        )
        raise

    metrics = trainer.evaluate()
    print(f"\nfinal eval: {metrics}")

    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "train_config.json").write_text(
        json.dumps({"args": vars(args), "facts": facts, "metrics": metrics},
                   indent=2, default=str)
    )
    card_meta = {**meta, "n_train": len(ds["train"]), "n_eval": len(ds["test"])}
    (args.out / "README.md").write_text(build_model_card(args, card_meta, facts, metrics))
    print(f"saved to {args.out}")

    # ---- push -----------------------------------------------------------------
    if args.hub_repo and not args.no_push:
        from huggingface_hub import HfApi

        try:
            HfApi().upload_folder(
                repo_id=args.hub_repo,
                folder_path=str(args.out),
                # Checkpoints are optimizer state, not something anyone downloads.
                ignore_patterns=["checkpoint-*", "runs/*", "*.pt"],
                commit_message=f"Darija fine-tune, eval_loss "
                               f"{metrics.get('eval_loss', float('nan')):.4f}",
            )
            print(f"\npushed to https://huggingface.co/{args.hub_repo}")
        except Exception as exc:  # noqa: BLE001
            print(f"\nPUSH FAILED: {exc}\n"
                  f"The model is saved at {args.out}. Push manually with:\n"
                  f"  hf upload {args.hub_repo} {args.out} "
                  f"--exclude 'checkpoint-*' 'runs/*'",
                  file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())