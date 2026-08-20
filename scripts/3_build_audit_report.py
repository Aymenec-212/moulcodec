#!/usr/bin/env python
"""Stage C: build the audit report and the Phase 2 verdict (HLD §2.5, spec T7).

Runs in the PROJECT env. Joins Stage A's signal metrics with Stage B's ASR
hypotheses, computes delta-WER/CER with bootstrap CIs, and states an
unambiguous go/no-go.

    uv run python scripts/3_build_audit_report.py --config configs/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import (  # noqa: E402
    EXPAND,
    bootstrap_ci,
    clip_counts,
    corpus_rates,
    decide,
    delta_cer_points,
    delta_wer_points,
    mean,
)
from src.utils import load_config, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the MoulCodec audit report")
    p.add_argument("--config", default="configs/pipeline_config.yaml")
    p.add_argument("--audit-dir", default=None)
    return p.parse_args()


def read_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["audio_id"]] = rec.get("text", "")
    return out


def load_references(cfg: dict) -> dict[str, str]:
    """References come from the token store — the same rows that were encoded."""
    store = Path(cfg["paths"]["token_store"])
    shards = sorted(store.glob("tokens-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no token shards in {store}")
    df = pd.concat(
        [pd.read_parquet(s, columns=["audio_id", "text"]) for s in shards], ignore_index=True
    )
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    return dict(zip(df["audio_id"], df["text"]))


def fmt_ci(ci: tuple[float, float], places: int = 2) -> str:
    return f"[{ci[0]:.{places}f}, {ci[1]:.{places}f}]"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    a = cfg["audit"]
    audit_dir = Path(args.audit_dir or cfg["paths"]["audit_dir"])

    refs = load_references(cfg)
    hyp_o = read_jsonl(audit_dir / "hyps_orig.jsonl")
    hyp_r = read_jsonl(audit_dir / "hyps_recon.jsonl")
    signals = pd.read_csv(audit_dir / "signal_metrics.csv").set_index("audio_id")

    ids = sorted(set(hyp_o) & set(hyp_r) & set(refs))
    missing = (set(hyp_o) | set(hyp_r)) - set(ids)
    if not ids:
        raise SystemExit("no clips have both hypotheses and a reference")
    print(f"{len(ids)} clips with complete data ({len(missing)} incomplete, skipped)")

    counts = [clip_counts(i, refs[i], hyp_o[i], hyp_r[i]) for i in ids]
    empty_refs = [c.audio_id for c in counts if c.w_ref == 0]
    counts = [c for c in counts if c.w_ref > 0]
    if empty_refs:
        print(f"dropped {len(empty_refs)} clips with empty references")

    rates = corpus_rates(counts)
    iters = int(a.get("bootstrap_iters", 2000))
    seed = int(a.get("seed", 42))
    wer_ci = bootstrap_ci(counts, delta_wer_points, iters=iters, seed=seed)
    cer_ci = bootstrap_ci(counts, delta_cer_points, iters=iters, seed=seed)

    spk_ci = None
    spk_vals: list[float] = []
    if "speaker_sim" in signals.columns:
        spk_vals = [
            float(signals.loc[c.audio_id, "speaker_sim"])
            for c in counts
            if c.audio_id in signals.index
        ]
        if len(spk_vals) >= 2:
            spk_ci = bootstrap_ci(spk_vals, mean, iters=iters, seed=seed)

    verdict, reasons = decide(
        wer_ci,
        spk_ci,
        thr_delta_wer=float(a.get("trigger_delta_wer", 5.0)),
        thr_speaker=float(a.get("trigger_speaker_sim", 0.80)),
    )

    rows = []
    for c in counts:
        r = {
            "audio_id": c.audio_id,
            "w_err_orig": c.w_err_orig,
            "w_err_recon": c.w_err_recon,
            "w_ref": c.w_ref,
            "c_err_orig": c.c_err_orig,
            "c_err_recon": c.c_err_recon,
            "c_ref": c.c_ref,
        }
        if c.audio_id in signals.index:
            for col in ("mel_l1", "si_sdr", "speaker_sim", "duration", "channel", "n_frames"):
                if col in signals.columns:
                    r[col] = signals.loc[c.audio_id, col]
        rows.append(r)
    per_clip = audit_dir / "per_clip.csv"
    pd.DataFrame(rows).to_csv(per_clip, index=False)

    meta = {}
    meta_path = audit_dir / "asr_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    mel = [r["mel_l1"] for r in rows if "mel_l1" in r]
    sisdr = [r["si_sdr"] for r in rows if "si_sdr" in r]

    def dist(xs: list[float], places: int = 3) -> str:
        if not xs:
            return "n/a"
        q = statistics.quantiles(xs, n=20) if len(xs) >= 20 else [min(xs), max(xs)]
        return (
            f"mean {mean(xs):.{places}f}, median {statistics.median(xs):.{places}f}, "
            f"p5 {q[0]:.{places}f}, p95 {q[-1]:.{places}f}"
        )

    lines = [
        "# MoulCodec Zero-Shot Audit Report",
        "",
        f"**Verdict: {verdict}**",
        "",
        *[f"- {r}" for r in reasons],
        "",
        "## Scope",
        "",
        f"- Clips audited: **{len(counts)}** (nested stratified prefix, seed {seed})",
        f"- Reference words: {int(rates['n_ref_words'])}",
        f"- ASR: `{meta.get('model', 'unknown')}` via {meta.get('engine', '?')}, "
        f"transformers {meta.get('transformers', '?')}",
        f"- Codec: `{cfg['codec']['model_id']}` (full encoder, encode path frozen)",
        "",
        "## Intelligibility (primary)",
        "",
        "| Metric | Original | Reconstruction | Delta | 95% CI |",
        "|---|---|---|---|---|",
        f"| WER (pts) | {rates['wer_orig']:.2f} | {rates['wer_recon']:.2f} | "
        f"**{rates['delta_wer']:+.2f}** | {fmt_ci(wer_ci)} |",
        f"| CER (pts) | {rates['cer_orig']:.2f} | {rates['cer_recon']:.2f} | "
        f"**{rates['delta_cer']:+.2f}** | {fmt_ci(cer_ci)} |",
        "",
        "Absolute WER is high by construction: references are Gemini-2.5-Pro "
        "transcripts and the ASR baseline is ~39 WER on Darija. Only the delta "
        "isolates codec-induced damage (HLD §2.3).",
        "",
        "## Speaker fidelity (primary)",
        "",
        f"- ECAPA cosine similarity: {dist(spk_vals) if spk_vals else 'not computed'}",
        f"- 95% CI on the mean: {fmt_ci(spk_ci, 3) if spk_ci else 'n/a'}",
        "",
        "## Signal distortion",
        "",
        f"- **Mel-L1 (primary):** {dist(mel)}",
        f"- SI-SDR dB (diagnostic only): {dist(sisdr, 2)}",
        "",
        "> SI-SDR is reported for bug detection, not quality. NeuCodec decodes "
        "through a Vocos head, which reconstructs perceptually and does not "
        "preserve phase, so low SI-SDR on good-sounding audio is expected. "
        "Judge signal quality by Mel-L1.",
        "",
        "## Phase 2 decision",
        "",
    ]

    if verdict == EXPAND:
        ladder = a.get("expand_to", [])
        nxt = next((n for n in ladder if n > len(counts)), None)
        lines += [
            "The interval straddles a trigger threshold — the sample cannot "
            "resolve the decision yet.",
            "",
            f"Expand the audit to **{nxt}** clips and re-run Stages A/B/C. The sample is a "
            "nested prefix, so only the new clips need transcribing."
            if nxt
            else "No further rung on the expansion ladder; widen `audit.expand_to` or accept "
            "the ambiguity explicitly.",
        ]
    elif verdict == "TRIGGER":
        lines += [
            "Degradation is real. Proceed to Phase 2 (HLD §3): freeze the entire "
            "encode path, train decoder + Vocos only. Scope the X-Codec2.0 "
            "training-harness reconstruction (§3.3) before writing code.",
        ]
    else:
        lines += [
            "Zero-shot quality is sufficient. Skip Phase 2 and proceed to the "
            "dataset release (spec T8).",
        ]

    lines += ["", f"Per-clip data: `{per_clip.name}`", ""]

    report = audit_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    write_json(
        {
            "verdict": verdict,
            "reasons": reasons,
            "n_clips": len(counts),
            **{k: v for k, v in rates.items()},
            "delta_wer_ci": list(wer_ci),
            "delta_cer_ci": list(cer_ci),
            "speaker_sim_ci": list(spk_ci) if spk_ci else None,
            "asr": meta,
        },
        audit_dir / "verdict.json",
    )

    print(f"\n=== {verdict} ===")
    for r in reasons:
        print(f"  {r}")
    print(f"\nwrote {report} and {per_clip}")


if __name__ == "__main__":
    main()