"""Nested stratified sampling for the audit fork (HLD §2.3, amended).

WHY NESTED: the audit runs on a sample first and expands only if the result is
statistically borderline. Expansion must REUSE the clips already transcribed —
ASR is the dominant cost (§2.2) and re-running it would defeat the point.

HOW: rather than allocating per-stratum counts (which can shrink a stratum's
allocation as n grows, breaking reuse), this module builds ONE deterministic
stratum-balanced global ordering. Every sample is a prefix of it, so:

    select(n1) is a subset of select(n2)   whenever n1 <= n2

holds by construction, and any prefix is approximately proportional across
strata. Strata are (duration bucket x channel), which covers the two axes most
likely to correlate with codec difficulty.

The seed is part of the contract: changing it reshuffles the ordering and
invalidates every previously transcribed clip.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

DEFAULT_DURATION_EDGES = (2.0, 4.0, 6.0, 8.0, 10.0)


def _u01(seed: int, key: str) -> float:
    """Deterministic uniform in [0,1) from a stable hash. Not Python's hash():
    that is salted per process and would break reproducibility across runs."""
    h = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def duration_bucket(duration: float, edges: Sequence[float] = DEFAULT_DURATION_EDGES) -> int:
    """Index of the duration bucket a clip falls into."""
    for i, e in enumerate(edges):
        if duration < e:
            return i
    return len(edges)


def stratified_order(
    ids: Sequence[str],
    durations: Sequence[float],
    channels: Sequence[str],
    seed: int = 42,
    edges: Sequence[float] = DEFAULT_DURATION_EDGES,
) -> list[int]:
    """Return ALL indices in a stratum-balanced deterministic order.

    Each item gets key (within_stratum_rank + offset_s) / stratum_size, so large
    and small strata interleave proportionally. Sorting by that key and taking a
    prefix yields a stratified sample of any size.
    """
    n = len(ids)
    if not (len(durations) == len(channels) == n):
        raise ValueError("ids, durations and channels must be the same length")
    if n == 0:
        return []

    strata: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i in range(n):
        strata[(duration_bucket(float(durations[i]), edges), str(channels[i]))].append(i)

    keyed: list[tuple[float, str, int]] = []
    for stratum, members in strata.items():
        # deterministic shuffle inside the stratum
        members = sorted(members, key=lambda i: (_u01(seed, str(ids[i])), str(ids[i])))
        size = len(members)
        offset = _u01(seed, f"stratum:{stratum[0]}:{stratum[1]}")
        for rank, idx in enumerate(members):
            keyed.append(((rank + offset) / size, str(ids[idx]), idx))

    keyed.sort(key=lambda t: (t[0], t[1]))  # id breaks ties -> total order
    return [idx for _, _, idx in keyed]


def select_audit_sample(
    ids: Sequence[str],
    durations: Sequence[float],
    channels: Sequence[str],
    n: int,
    seed: int = 42,
    edges: Sequence[float] = DEFAULT_DURATION_EDGES,
) -> list[int]:
    """Indices of the first `n` clips in the stratified order (nested prefix)."""
    if n <= 0:
        return []
    return stratified_order(ids, durations, channels, seed=seed, edges=edges)[:n]


def stratum_counts(
    durations: Sequence[float],
    channels: Sequence[str],
    indices: Sequence[int] | None = None,
    edges: Sequence[float] = DEFAULT_DURATION_EDGES,
) -> dict[tuple[int, str], int]:
    """Population or sample counts per stratum — for reporting sample balance."""
    idxs = range(len(durations)) if indices is None else indices
    out: dict[tuple[int, str], int] = defaultdict(int)
    for i in idxs:
        out[(duration_bucket(float(durations[i]), edges), str(channels[i]))] += 1
    return dict(out)