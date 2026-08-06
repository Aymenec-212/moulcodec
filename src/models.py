"""NeuCodec wrapper: batched encode/decode with padding-aware truncation.

HLD refs:
  §1  encode path (semantic encoder, acoustic encoder, FSQ) is FROZEN everywhere.
      This module never trains and never mutates weights.
  §1  decoder emits 24 kHz ONLY. Resampling to 16 kHz for metrics belongs in
      metrics.py, not here — this module returns raw 24 kHz.
  §2.3 token sanity: n_frames == n_samples // 320 (see FRAME FORMULA below).

FRAME FORMULA (empirically established, T0 smoke test + tests/test_models.py):
  NeuCodec._prepare_audio pads unconditionally: pad = 320 - (T % 320), so a
  T that is already a multiple of 320 still gains a full 320 samples. That
  makes the ACOUSTIC branch emit T//320 + 1 frames. But the SEMANTIC branch
  (Wav2Vec2-BERT: 10 ms hop, stride-2 stacking) emits T//320, and encode_code
  clamps both to min_len. Net result:

      n_frames == n_samples // 320        (== floor(50 * duration_seconds))

  Verified: 16000 samples -> 50 frames -> 24000 decoded samples @ 24 kHz.
"""

from __future__ import annotations

from typing import Sequence

import torch
from neucodec import NeuCodec

SAMPLE_RATE = 16_000
DECODE_SR = 24_000
HOP = 320                                   # 16000 / 50 Hz
FRAME_RATE = 50
SAMPLES_PER_FRAME_24K = DECODE_SR // FRAME_RATE   # 480


def pick_device(pref: str | None = None) -> torch.device:
    """CUDA when present, else CPU.

    MPS is deliberately NOT auto-selected: op coverage gaps in the codec's
    conv/weight_norm stack make it a debugging trap. Pass device="mps"
    explicitly (with PYTORCH_ENABLE_MPS_FALLBACK=1) if you want to try it.
    """
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def frames_for_samples(n_samples: int) -> int:
    """Expected token count for a 16 kHz clip of n_samples. See FRAME FORMULA."""
    return max(1, n_samples // HOP)


def pad_to_hop(wav: torch.Tensor, hop: int = HOP) -> torch.Tensor:
    """Right-pad a 1-D waveform so its length is a multiple of `hop`.

    NeuCodec pads internally anyway; doing it up front makes the frame count
    deterministic and keeps bucketing arithmetic honest.
    """
    if wav.dim() != 1:
        raise ValueError(f"expected 1-D waveform, got shape {tuple(wav.shape)}")
    rem = wav.numel() % hop
    if rem == 0:
        return wav
    return torch.nn.functional.pad(wav, (0, hop - rem))


class NeuCodecWrapper:
    """Inference-only wrapper around neuphonic/neucodec.

    Contract:
      encode_batch(wavs) -> list[LongTensor[F_i]]   tokens, per-item truncated
      decode_batch(toks) -> list[FloatTensor[T_i]]  24 kHz audio, per-item trimmed

    Mixed-length batches are REJECTED by default. encode_code applies no
    attention mask, so padding a short clip up to the batch max lets the
    semantic transformer attend over trailing silence, which can perturb the
    tokens of the real frames. That would make tokens a function of batch
    composition — fatal for reproducibility and for resumable runs (HLD §2.1:
    the token store is the single source of truth). Bucket by exact length in
    the pipeline; pass strict=False only if the characterization test in
    tests/test_models.py shows divergence is acceptable for your use.
    """

    def __init__(
        self,
        model_id: str = "neuphonic/neucodec",
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = pick_device(device)
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.model = NeuCodec.from_pretrained(model_id, **kwargs).eval().to(self.device)
        # Belt and braces on the frozen-encode-path invariant (HLD §1).
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- encode

    @torch.inference_mode()
    def encode_batch(
        self,
        wavs: Sequence[torch.Tensor],
        strict: bool = True,
    ) -> list[torch.Tensor]:
        """Encode a batch of 1-D float32 16 kHz waveforms to FSQ tokens.

        Returns one LongTensor per input, truncated to frames_for_samples(len).
        """
        if not wavs:
            return []
        for w in wavs:
            if w.dim() != 1:
                raise ValueError(f"expected 1-D waveforms, got {tuple(w.shape)}")

        lengths = {int(w.numel()) for w in wavs}
        if len(lengths) > 1:
            if strict:
                raise ValueError(
                    f"mixed-length batch {sorted(lengths)}; bucket by exact length "
                    "or pass strict=False (see class docstring)"
                )
            t_max = max(lengths)
            wavs = [torch.nn.functional.pad(w, (0, t_max - w.numel())) for w in wavs]

        batch = torch.stack([w.to(torch.float32).reshape(1, -1) for w in wavs])  # [B,1,T]
        codes = self.model.encode_code(batch.to(self.device))                    # [B,1,F]
        codes = codes.squeeze(1).cpu().long()

        out: list[torch.Tensor] = []
        for i, w in enumerate(wavs):
            n = min(frames_for_samples(int(w.numel())), codes.shape[-1])
            out.append(codes[i, :n].clone())
        return out

    def encode_one(self, wav: torch.Tensor) -> torch.Tensor:
        """Reference path: encode a single clip with no batch neighbours."""
        return self.encode_batch([wav])[0]

    # ---------------------------------------------------------------- decode

    @torch.inference_mode()
    def decode_batch(
        self,
        token_seqs: Sequence[Sequence[int] | torch.Tensor],
        strict: bool = True,
    ) -> list[torch.Tensor]:
        """Decode token sequences to 24 kHz waveforms (NOT resampled — HLD §1).

        Same batch-composition caveat as encode: sequences of unequal length are
        zero-padded and the decoded tail is trimmed, but the decoder's receptive
        field means the junk region can bleed into the real tail. Bucket here too.
        """
        if not token_seqs:
            return []
            
        lengths = {len(t) for t in token_seqs}
        if len(lengths) > 1 and strict:
            raise ValueError(
                f"mixed-length token batch {sorted(lengths)}; bucket by exact "
                "length or pass strict=False"
            )

        f_max = max(lengths)
        padded = torch.zeros(len(token_seqs), 1, f_max, dtype=torch.long)
        for i, t in enumerate(token_seqs):
            padded[i, 0, : len(t)] = torch.as_tensor(t, dtype=torch.long)

        wav = self.model.decode_code(padded.to(self.device))     # [B,1,T] @ 24 kHz
        wav = wav.squeeze(1).float().cpu()
        return [
            wav[i, : len(t) * SAMPLES_PER_FRAME_24K].clone() for i, t in enumerate(token_seqs)
        ]

    def decode_one(self, tokens: Sequence[int]) -> torch.Tensor:
        return self.decode_batch([tokens])[0]