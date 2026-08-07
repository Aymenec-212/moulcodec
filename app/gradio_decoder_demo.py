#!/usr/bin/env python
"""T10: Gradio decoder sanity check (HLD §4.2).

Lets anyone paste a token sequence — or pull one from the released dataset —
and hear it. Proves the vocoder works locally and makes the token space
audible rather than abstract.

    uv run --extra app python app/gradio_decoder_demo.py

Add to pyproject.toml:

    [project.optional-dependencies]
    app = ["gradio>=4.0"]

Deployed as a Space, rename this to app.py and add gradio + neucodec to
requirements.txt there.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import DECODE_SR, NeuCodecWrapper  # noqa: E402

DATASET_REPO = "atlasia/MoulSot-Tokens-v1"
VOCAB = 65_536
MAX_TOKENS = 3_000  # 60 s at 50 Hz — keeps a shared Space responsive

_codec: NeuCodecWrapper | None = None
_samples: list[dict] = []


def codec() -> NeuCodecWrapper:
    global _codec
    if _codec is None:
        _codec = NeuCodecWrapper()
    return _codec


def parse_tokens(raw: str) -> list[int]:
    """Accept JSON arrays, comma-separated, or whitespace-separated integers."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("no tokens given")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(t) for t in parsed]
    except json.JSONDecodeError:
        pass

    parts = [p for p in re.split(r"[,\s]+", raw) if p]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"could not parse tokens: {exc}") from exc


def validate(tokens: list[int]) -> None:
    if not tokens:
        raise ValueError("empty token sequence")
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"{len(tokens)} tokens exceeds the {MAX_TOKENS} limit for this demo")
    lo, hi = min(tokens), max(tokens)
    if lo < 0 or hi >= VOCAB:
        raise ValueError(f"tokens must be in [0, {VOCAB}); got min {lo}, max {hi}")


def decode(raw: str):
    try:
        tokens = parse_tokens(raw)
        validate(tokens)
    except ValueError as exc:
        return None, f"❌ {exc}"

    wav = codec().decode_one(tokens).numpy().astype(np.float32)
    peak = float(np.abs(wav).max())
    if peak > 1.0:
        wav = wav / peak

    info = (
        f"✅ decoded {len(tokens)} tokens → {wav.size / DECODE_SR:.2f} s @ {DECODE_SR} Hz\n"
        f"distinct tokens: {len(set(tokens))} | range [{min(tokens)}, {max(tokens)}]"
    )
    return (DECODE_SR, wav), info


def load_samples(n: int = 20) -> str:
    global _samples
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET_REPO, split=f"train[:{n}]")
        _samples = [
            {"id": r["audio_id"], "text": r["text"], "tokens": [int(t) for t in r["tokens"]]}
            for r in ds
        ]
        return f"loaded {len(_samples)} samples from {DATASET_REPO}"
    except Exception as exc:  # noqa: BLE001
        return f"could not load {DATASET_REPO}: {exc}"


def pick_sample(index: int):
    if not _samples:
        return "", "load samples first"
    s = _samples[int(index) % len(_samples)]
    return json.dumps(s["tokens"]), f"**{s['id']}**\n\n{s['text']}"


def build_ui():
    import gradio as gr

    with gr.Blocks(title="MoulCodec Decoder") as demo:
        gr.Markdown(
            "# MoulCodec Decoder\n"
            "Turn Darija speech tokens back into audio. Tokens come from "
            f"[`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO}), "
            "encoded with `neuphonic/neucodec` (FSQ, 50 tokens/second).\n\n"
            "The decoder outputs **24 kHz** even though the encoder takes 16 kHz — "
            "that is the codec's only output path, not a bug."
        )

        with gr.Row():
            with gr.Column():
                tokens_in = gr.Textbox(
                    label="Token IDs",
                    placeholder="[1243, 8421, 9901, ...]  — JSON, commas, or spaces",
                    lines=8,
                )
                decode_btn = gr.Button("Decode", variant="primary")

                with gr.Accordion("Load from the released dataset", open=False):
                    load_btn = gr.Button("Fetch 20 samples")
                    load_status = gr.Markdown()
                    idx = gr.Slider(0, 19, value=0, step=1, label="Sample index")
                    pick_btn = gr.Button("Use this sample")
                    transcript = gr.Markdown()

            with gr.Column():
                audio_out = gr.Audio(label="Decoded audio", type="numpy")
                info_out = gr.Markdown()

        decode_btn.click(decode, inputs=tokens_in, outputs=[audio_out, info_out])
        load_btn.click(load_samples, outputs=load_status)
        pick_btn.click(pick_sample, inputs=idx, outputs=[tokens_in, transcript])

    return demo


if __name__ == "__main__":
    build_ui().launch()