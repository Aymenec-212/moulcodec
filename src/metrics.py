"""Audit metrics (HLD §2.3, spec T3b).

CONTRACT: every signal metric takes (reference @16 kHz, estimate @24 kHz) and
resamples the estimate to 16 kHz internally. Callers never resample — that rule
lives here so nobody double-resamples or compares across rates (HLD §1).

LENGTH: a reconstruction of F frames is F*480 samples @24 kHz -> F*320 @16 kHz.
Since F = n_samples // 320, the reconstruction is up to 319 samples SHORTER than
the original. Everything below trims to the common length; a phantom offset
would wreck SI-SDR in particular.

METRIC WEIGHTING — read before interpreting a report:
  ΔWER / ΔCER   primary. Isolates codec damage from transcript noise.
  speaker sim   primary. ECAPA cosine, original vs reconstruction.
  Mel-L1        primary signal metric. Phase-insensitive, so it is meaningful
                for a generative vocoder.
  SI-SDR        DIAGNOSTIC ONLY. NeuCodec decodes through a Vocos head, which
                reconstructs perceptually and does NOT preserve phase. Expect
                values near 0 dB or negative on audio that sounds fine. Use it
                to catch gross alignment/padding bugs, not to judge quality.
  PESQ / STOI   optional, intrusive, also phase-sensitive. Same caveat as SI-SDR.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import torch
import torchaudio

SR = 16_000
DECODE_SR = 24_000
SI_SDR_CAP_DB = 100.0  # identical signals -> +inf; cap so aggregates stay finite


# ------------------------------------------------------------------ helpers


def to_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Resample a 1-D waveform to 16 kHz. No-op when already there."""
    if wav.dim() != 1:
        raise ValueError(f"expected 1-D waveform, got {tuple(wav.shape)}")
    if sr == SR:
        return wav.float()
    return torchaudio.functional.resample(wav.float(), sr, SR)


def align(ref: torch.Tensor, est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim both signals to their common length (see LENGTH note above)."""
    n = min(ref.numel(), est.numel())
    if n == 0:
        raise ValueError("empty signal after alignment")
    return ref[:n].float(), est[:n].float()


def prepare(ref16k: torch.Tensor, est24k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard entry point: resample the estimate to 16 kHz, then align."""
    return align(ref16k.float(), to_16k(est24k, DECODE_SR))


# ---------------------------------------------------------- signal metrics


def si_sdr(ref16k: torch.Tensor, est24k: torch.Tensor) -> float:
    """Scale-invariant SDR in dB. DIAGNOSTIC ONLY — see module docstring."""
    ref, est = prepare(ref16k, est24k)
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = ref.pow(2).sum()
    if denom == 0:
        raise ValueError("reference is silent; SI-SDR undefined")
    proj = (est @ ref) / denom * ref
    noise = est - proj
    if noise.pow(2).sum() == 0:
        return SI_SDR_CAP_DB
    val = 10 * torch.log10(proj.pow(2).sum() / noise.pow(2).sum())
    return float(torch.clamp(val, max=SI_SDR_CAP_DB))


_MEL = None


def _mel_transform() -> torchaudio.transforms.MelSpectrogram:
    global _MEL
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=256, n_mels=80, power=1.0
        )
    return _MEL


def mel_l1(ref16k: torch.Tensor, est24k: torch.Tensor, eps: float = 1e-5) -> float:
    """L1 distance between log-mel magnitude spectra. PRIMARY signal metric."""
    ref, est = prepare(ref16k, est24k)
    mel = _mel_transform()
    a = torch.log(mel(ref) + eps)
    b = torch.log(mel(est) + eps)
    return float((a - b).abs().mean())


def pesq_stoi(ref16k: torch.Tensor, est24k: torch.Tensor) -> dict[str, float | None]:
    """Optional intrusive metrics. Returns None entries if deps are absent."""
    ref, est = prepare(ref16k, est24k)
    out: dict[str, float | None] = {"pesq": None, "stoi": None}
    try:
        from pesq import pesq as _pesq

        out["pesq"] = float(_pesq(SR, ref.numpy(), est.numpy(), "wb"))
    except Exception:
        pass
    try:
        from pystoi import stoi as _stoi

        out["stoi"] = float(_stoi(ref.numpy(), est.numpy(), SR, extended=False))
    except Exception:
        pass
    return out


# --------------------------------------------------------- speaker fidelity


class SpeakerSimilarity:
    """ECAPA-TDNN cosine similarity, original vs reconstruction (HLD §2.3).

    Lazy-loaded: constructing this downloads speechbrain weights.
    """

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = torch.device(device)
        self.model = EncoderClassifier.from_hparams(
            source=model_id, run_opts={"device": str(self.device)}
        )

    @torch.inference_mode()
    def _embed(self, wav16k: torch.Tensor) -> torch.Tensor:
        emb = self.model.encode_batch(wav16k.unsqueeze(0).to(self.device))
        return emb.squeeze().float().cpu()

    def __call__(self, ref16k: torch.Tensor, est24k: torch.Tensor) -> float:
        ref, est = prepare(ref16k, est24k)
        a, b = self._embed(ref), self._embed(est)
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


# ------------------------------------------------------------ transcription

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]", flags=re.UNICODE)


def normalize_ar(text: str) -> str:
    """Light Arabic/Darija normalisation applied IDENTICALLY to ref and both hyps.

    Strips diacritics and tatweel, unifies alef/ya/ta-marbuta variants, drops
    punctuation, collapses whitespace. Deliberately conservative: aggressive
    normalisation would mask real codec-induced errors.
    """
    t = unicodedata.normalize("NFKC", text)
    t = _ARABIC_DIACRITICS.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    # 5. Drop ALL punctuation (Unicode category starting with 'P')
    # This catches ASCII punctuation (!, ?, .) AND Arabic punctuation (،, ؛, ؟)
    t = "".join(c for c in t if not unicodedata.category(c).startswith("P"))
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


@dataclass(frozen=True)
class DeltaWER:
    wer_orig: float
    wer_recon: float
    cer_orig: float
    cer_recon: float

    @property
    def delta_wer(self) -> float:
        return self.wer_recon - self.wer_orig

    @property
    def delta_cer(self) -> float:
        return self.cer_recon - self.cer_orig

    def as_dict(self) -> dict[str, float]:
        return {
            "wer_orig": self.wer_orig,
            "wer_recon": self.wer_recon,
            "delta_wer": self.delta_wer,
            "cer_orig": self.cer_orig,
            "cer_recon": self.cer_recon,
            "delta_cer": self.delta_cer,
        }


def delta_wer(
    refs: list[str],
    hyps_orig: list[str],
    hyps_recon: list[str],
    normalize: bool = True,
) -> DeltaWER:
    """ΔWER/ΔCER per HLD §2.3.

    `refs` are the dataset's Gemini transcripts; `hyps_orig` and `hyps_recon` are
    moulsot.v0.3 output on the original and reconstructed audio. The delta is
    what isolates codec damage from transcript noise — never report the absolutes
    alone (HLD §1).
    """
    import jiwer

    if not (len(refs) == len(hyps_orig) == len(hyps_recon)):
        raise ValueError(
            f"length mismatch: refs={len(refs)} orig={len(hyps_orig)} recon={len(hyps_recon)}"
        )
    if not refs:
        raise ValueError("no transcripts supplied")

    f = normalize_ar if normalize else (lambda s: s)
    r = [f(x) for x in refs]
    ho = [f(x) for x in hyps_orig]
    hr = [f(x) for x in hyps_recon]

    return DeltaWER(
        wer_orig=float(jiwer.wer(r, ho)),
        wer_recon=float(jiwer.wer(r, hr)),
        cer_orig=float(jiwer.cer(r, ho)),
        cer_recon=float(jiwer.cer(r, hr)),
    )