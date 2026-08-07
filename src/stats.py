"""Corpus-level WER/CER aggregation, bootstrap CIs, and the Phase 2 verdict.

Separate from metrics.py on purpose: metrics.py MEASURES one clip, this module
AGGREGATES across clips and draws an inference from the aggregate.

WHY PER-CLIP COUNTS AND NOT PER-CLIP WER: corpus WER is total_errors /
total_reference_words, not the mean of per-clip WERs. A 1-word clip and a
40-word clip do not carry equal weight. Storing counts also means the audit can
EXPAND its sample without re-running ASR on clips already transcribed — just
append rows and recompute.

DECISION RULE (HLD §3 trigger, amended for sampling):
    delta-WER 95% CI entirely below 5 pts  -> NO_TRIGGER
    delta-WER 95% CI entirely above 5 pts  -> TRIGGER
    CI straddles the threshold             -> EXPAND the nested prefix
  Same shape for mean speaker similarity against 0.80.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from src.metrics import normalize_ar


@dataclass(frozen=True)
class ClipCounts:
    """Error/reference counts for one clip, on both the original and the recon."""

    audio_id: str
    w_err_orig: int
    w_err_recon: int
    w_ref: int
    c_err_orig: int
    c_err_recon: int
    c_ref: int


def _word_counts(ref: str, hyp: str) -> tuple[int, int]:
    import jiwer

    o = jiwer.process_words(ref, hyp)
    errors = o.substitutions + o.deletions + o.insertions
    n_ref = o.substitutions + o.deletions + o.hits
    return int(errors), int(n_ref)


def _char_counts(ref: str, hyp: str) -> tuple[int, int]:
    import jiwer

    o = jiwer.process_characters(ref, hyp)
    errors = o.substitutions + o.deletions + o.insertions
    n_ref = o.substitutions + o.deletions + o.hits
    return int(errors), int(n_ref)


def clip_counts(
    audio_id: str,
    ref: str,
    hyp_orig: str,
    hyp_recon: str,
    normalize: bool = True,
) -> ClipCounts:
    """Count errors for one clip. Normalisation is applied identically to all three."""
    f = normalize_ar if normalize else (lambda s: s)
    r, ho, hr = f(ref), f(hyp_orig), f(hyp_recon)

    w_err_o, w_ref = _word_counts(r, ho)
    w_err_r, _ = _word_counts(r, hr)
    c_err_o, c_ref = _char_counts(r, ho)
    c_err_r, _ = _char_counts(r, hr)

    return ClipCounts(audio_id, w_err_o, w_err_r, w_ref, c_err_o, c_err_r, c_ref)


# --------------------------------------------------------------- aggregation


def corpus_rates(counts: Sequence[ClipCounts]) -> dict[str, float]:
    """Corpus WER/CER in PERCENTAGE POINTS, plus the deltas."""
    w_ref = sum(c.w_ref for c in counts)
    c_ref = sum(c.c_ref for c in counts)
    if w_ref == 0 or c_ref == 0:
        raise ValueError("no reference words/characters — cannot compute rates")

    wer_o = 100.0 * sum(c.w_err_orig for c in counts) / w_ref
    wer_r = 100.0 * sum(c.w_err_recon for c in counts) / w_ref
    cer_o = 100.0 * sum(c.c_err_orig for c in counts) / c_ref
    cer_r = 100.0 * sum(c.c_err_recon for c in counts) / c_ref

    return {
        "wer_orig": wer_o,
        "wer_recon": wer_r,
        "delta_wer": wer_r - wer_o,
        "cer_orig": cer_o,
        "cer_recon": cer_r,
        "delta_cer": cer_r - cer_o,
        "n_clips": float(len(counts)),
        "n_ref_words": float(w_ref),
    }


def delta_wer_points(counts: Sequence[ClipCounts]) -> float:
    return corpus_rates(counts)["delta_wer"]


def delta_cer_points(counts: Sequence[ClipCounts]) -> float:
    return corpus_rates(counts)["delta_cer"]


# ---------------------------------------------------------------- bootstrap


def bootstrap_ci(
    items: Sequence,
    stat: Callable[[Sequence], float],
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling CLIPS with replacement.

    Resampling clips (not words) is what makes the interval honest: errors
    within a clip are correlated, so treating words as independent would give a
    spuriously tight interval.
    """
    if len(items) < 2:
        raise ValueError("need at least 2 items to bootstrap")
    rng = random.Random(seed)
    n = len(items)
    draws: list[float] = []
    for _ in range(iters):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        try:
            draws.append(stat(sample))
        except ValueError:
            continue  # degenerate resample (e.g. all-empty refs); skip
    if not draws:
        raise ValueError("every bootstrap resample was degenerate")
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return lo, hi


def mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x is not None]
    if not vals:
        raise ValueError("empty sequence")
    return sum(vals) / len(vals)


# ------------------------------------------------------------------ verdict

TRIGGER = "TRIGGER"
NO_TRIGGER = "NO_TRIGGER"
EXPAND = "EXPAND"


def decide(
    delta_wer_ci: tuple[float, float],
    speaker_ci: tuple[float, float] | None,
    thr_delta_wer: float = 5.0,
    thr_speaker: float = 0.80,
) -> tuple[str, list[str]]:
    """Phase 2 go/no-go from confidence intervals. Returns (verdict, reasons)."""
    reasons: list[str] = []
    lo, hi = delta_wer_ci

    wer_fails = lo > thr_delta_wer
    wer_passes = hi < thr_delta_wer
    if wer_fails:
        reasons.append(f"delta-WER CI [{lo:.2f}, {hi:.2f}] entirely above {thr_delta_wer} pts")
    elif wer_passes:
        reasons.append(f"delta-WER CI [{lo:.2f}, {hi:.2f}] entirely below {thr_delta_wer} pts")
    else:
        reasons.append(f"delta-WER CI [{lo:.2f}, {hi:.2f}] straddles {thr_delta_wer} pts")

    spk_fails = spk_passes = False
    if speaker_ci is not None:
        slo, shi = speaker_ci
        spk_fails = shi < thr_speaker
        spk_passes = slo > thr_speaker
        if spk_fails:
            reasons.append(f"speaker-sim CI [{slo:.3f}, {shi:.3f}] entirely below {thr_speaker}")
        elif spk_passes:
            reasons.append(f"speaker-sim CI [{slo:.3f}, {shi:.3f}] entirely above {thr_speaker}")
        else:
            reasons.append(f"speaker-sim CI [{slo:.3f}, {shi:.3f}] straddles {thr_speaker}")
    else:
        spk_passes = True
        reasons.append("speaker similarity not computed")

    if wer_fails or spk_fails:
        return TRIGGER, reasons
    if wer_passes and spk_passes:
        return NO_TRIGGER, reasons
    return EXPAND, reasons