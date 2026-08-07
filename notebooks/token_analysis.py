# %% [markdown]
# # MoulCodec Token Analysis (T10, HLD §4.1)
#
# Written as a `# %%` cell script rather than a `.ipynb`: VS Code runs it as an
# interactive notebook, and it diffs cleanly in git — a real advantage for a
# file that will be reviewed in PRs.
#
# Two constraints from HLD §4.1 are honoured here:
#
# - **No t-SNE on the raw 65,536-token vocabulary.** Token IDs are arbitrary
#   integer labels; embedding them shows nothing about phonetics.
# - **t-SNE runs on the continuous pre-quantisation projections**, obtained via
#   `NeuCodecWrapper.encode_latents` (see the models.py addendum).
#
# **Honest caveat on "phoneme clustering":** the HLD asks the t-SNE to show
# phoneme structure, but this corpus has no phoneme alignments and no reliable
# Darija phonemiser exists. Colouring by an unlabelled proxy is the best
# available option — below, frames are coloured by a coarse acoustic class
# (silence / low-frequency-dominant / high-frequency-dominant) derived from
# energy and zero-crossing rate. Read the plot as "does the latent space
# organise acoustically", not as verified phoneme clusters.

# %%
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from src.data import as_float32, load_subset, preprocess  # noqa: E402
from src.models import NeuCodecWrapper  # noqa: E402
from src.utils import load_config  # noqa: E402

CFG = load_config("configs/pipeline_config.yaml")
RELEASE = Path(CFG["paths"]["release_dir"]) / "moulsot_tokens.parquet"
N_LATENT_CLIPS = 40  # clips to encode for the latent-space plots

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## 1. Load the released tokens

# %%
import pandas as pd

df = pd.read_parquet(RELEASE)
print(f"{len(df):,} clips | {df['duration'].sum() / 3600:.1f} hours")
print(f"{int(df['tokens'].map(len).sum()):,} tokens total")
df.head(3)[["audio_id", "duration", "n_samples", "channel"]]

# %% [markdown]
# ## 2. Codebook utilisation
#
# The single most informative plot for an FSQ codec. FSQ cannot collapse the way
# VQ does, but *coverage* still tells you how much of the code space this
# language actually occupies — and low coverage on Darija specifically would be
# the strongest quantitative argument for domain adaptation.

# %%
counts = Counter()
for toks in df["tokens"]:
    counts.update(int(t) for t in toks)

total = sum(counts.values())
used = len(counts)
freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
p = freqs / total
entropy = float(-(p * np.log2(p)).sum())

print(f"distinct tokens used : {used:,} / 65,536  ({100 * used / 65536:.2f}%)")
print(f"entropy              : {entropy:.2f} bits (max {np.log2(65536):.0f})")
print(f"perplexity           : {2 ** entropy:,.0f}")
print(f"top-1000 share       : {100 * freqs[:1000].sum() / total:.1f}% of all tokens")

fig, ax = plt.subplots(1, 2)
ax[0].loglog(np.arange(1, len(freqs) + 1), freqs)
ax[0].set(xlabel="token rank", ylabel="count", title="Token frequency (log-log)")
ax[0].grid(alpha=0.3)

cum = np.cumsum(freqs) / total
ax[1].plot(np.arange(1, len(freqs) + 1), cum)
ax[1].set(xlabel="tokens included", ylabel="cumulative share", title="Concentration")
ax[1].grid(alpha=0.3)
plt.tight_layout()

# %% [markdown]
# ## 3. Per-dimension FSQ histograms
#
# Token IDs are decomposed into per-dimension codes using the quantiser's own
# `indices_to_codes`, not manual base arithmetic — the level configuration is
# read from the model rather than assumed.

# %%
codec = NeuCodecWrapper(device=None)
levels = codec.fsq_levels()
print(f"FSQ levels: {levels}  ->  {int(np.prod(levels)):,} codes")

sample_tokens = [int(t) for toks in df["tokens"].head(500) for t in toks]
codes = codec.tokens_to_codes(sample_tokens).numpy()  # [F, D]
n_dims = codes.shape[1]

fig, axes = plt.subplots(1, n_dims, figsize=(2.2 * n_dims, 3), sharey=True)
for d, ax in enumerate(np.atleast_1d(axes)):
    ax.hist(codes[:, d], bins=levels[d] if d < len(levels) else 20, color="#4C72B0")
    ax.set(title=f"dim {d}", xlabel="code value")
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel("frames")
plt.suptitle("FSQ per-dimension occupancy", y=1.04)
plt.tight_layout()

# %% [markdown]
# A dimension that is heavily skewed toward one level is carrying little
# information for this language. Flat-ish histograms across dimensions mean the
# code space is being used efficiently on Darija.

# %% [markdown]
# ## 4. Frame-rate sanity
#
# Re-checks the invariant the whole pipeline rests on: `len(tokens) ==
# n_samples // 320`. Any drift here means a padding or alignment bug reached the
# release.

# %%
expected = df["n_samples"] // 320
delta = df["tokens"].map(len) - expected
print(f"exact matches: {(delta == 0).sum():,} / {len(df):,}")
print(f"within ±1    : {(delta.abs() <= 1).sum():,} / {len(df):,}")
if (delta.abs() > 1).any():
    print("\nVIOLATIONS — investigate before trusting this release:")
    print(df.loc[delta.abs() > 1, ["audio_id", "n_samples"]].head(10))

# %% [markdown]
# ## 5. t-SNE on pre-quantisation latents
#
# This needs audio, so it loads a small slice of the source dataset and encodes
# it. Frames are coloured by a coarse acoustic proxy — see the caveat at the top
# about why this is not phoneme labelling.

# %%
ds = preprocess(load_subset(CFG, pilot=N_LATENT_CLIPS), CFG)

latents, classes = [], []
for i in range(len(ds)):
    wav = torch.from_numpy(as_float32(np.asarray(ds[i]["wav"])))
    z = codec.encode_latents(wav)              # [D, F] or [F, D]
    z = z.T if z.shape[0] < z.shape[1] else z  # -> [F, D]
    latents.append(z.numpy())

    # coarse acoustic class per frame, at the 320-sample hop
    n_frames = z.shape[0]
    frames = wav[: n_frames * 320].reshape(n_frames, 320).numpy()
    energy = (frames**2).mean(axis=1)
    zcr = (np.diff(np.sign(frames), axis=1) != 0).mean(axis=1)
    thresh = np.percentile(energy, 20)
    cls = np.where(energy < thresh, 0, np.where(zcr < np.median(zcr), 1, 2))
    classes.append(cls)

Z = np.concatenate(latents, axis=0)
C = np.concatenate(classes, axis=0)
print(f"{Z.shape[0]:,} frames x {Z.shape[1]} latent dims")

# %%
from sklearn.manifold import TSNE

idx = np.random.default_rng(42).choice(len(Z), size=min(5000, len(Z)), replace=False)
emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Z[idx])

labels = {0: "low energy", 1: "LF-dominant", 2: "HF-dominant"}
plt.figure(figsize=(7, 6))
for k, name in labels.items():
    m = C[idx] == k
    plt.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, label=name)
plt.legend(markerscale=4)
plt.title("Pre-quantisation latent space (t-SNE)\ncoloured by coarse acoustic class, not phonemes")
plt.tight_layout()

# %% [markdown]
# ## 6. Duration and channel coverage

# %%
fig, ax = plt.subplots(1, 2)
ax[0].hist(df["duration"], bins=40, color="#55A868")
ax[0].set(xlabel="seconds", ylabel="clips", title="Clip duration")
ax[0].grid(alpha=0.3)

top = df["channel"].value_counts().head(15)
ax[1].barh(top.index[::-1].astype(str), top.values[::-1], color="#C44E52")
ax[1].set(xlabel="clips", title="Top source channels")
plt.tight_layout()

print(f"{df['channel'].nunique()} distinct channels")
print(f"largest channel is {100 * top.iloc[0] / len(df):.1f}% of the corpus")