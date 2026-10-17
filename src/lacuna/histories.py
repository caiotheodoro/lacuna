"""Capture histories: which witnesses saw an action.

An action seen by witnesses {0, 2} out of K=3 has history bits 0b101 = 5.
History 0, seen by nobody, is the cell this project exists to estimate. It is
never observed, so a count table is a length-2**K integer array whose entry 0
is ignored on input.

Witness order is fixed so that "W1" in the docs is bit 0 here, everywhere.
"""

from __future__ import annotations

import numpy as np

WITNESSES = ("trace", "proxy", "env", "auditor")


def bits(h: int, k: int) -> np.ndarray:
    """The K capture indicators of history h, witness 0 first."""
    return np.array([(h >> j) & 1 for j in range(k)], dtype=float)


def n_witnesses(counts: np.ndarray) -> int:
    k = int(np.log2(len(counts)))
    if 2**k != len(counts) or k < 2:
        raise ValueError(f"count table length {len(counts)} is not 2**K with K >= 2")
    return k


def observed_total(counts: np.ndarray) -> int:
    """Distinct actions seen by at least one witness. Never counts cell 0."""
    return int(np.asarray(counts)[1:].sum())


def seen_by(counts: np.ndarray, witness: int) -> int:
    """Actions witness `witness` saw, whatever the others did."""
    return int(sum(c for h, c in enumerate(counts) if h and (h >> witness) & 1))


def keep_witnesses(counts: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    """Marginalise the table onto a subset of witnesses, in the given order.

    Actions seen only by a dropped witness fall into the new cell 0, which is
    exactly what happens to an auditor who does not have that witness.
    """
    out = np.zeros(2 ** len(keep), dtype=np.int64)
    for h, c in enumerate(counts):
        if h == 0:
            continue
        g = sum(((h >> w) & 1) << i for i, w in enumerate(keep))
        out[g] += c
    out[0] = 0
    return out
