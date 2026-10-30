"""Thin a config's logs at chosen loss rates and count the capture histories.

Standard library only, and it does not import lacuna: it ships inside the
dataset and runs from the dataset root.

    from loss_sim import load, thin

    rows = load("confirm")                      # episodes/confirm.jsonl
    out = thin(rows, drop={0: 0.5, 1: 0.5}, env_view="financial", seed=4242)
    out["n_true"], out["cells"], out["unseen"]

`thin` keeps or drops each record of `records` on its own (independent,
per-record loss), links the survivors to the ledger by `record.oracle`, and
counts ledger actions by which witnesses still hold them. A history is a bit
mask, bit w set when witness w recorded the action: 1 = W1 trace, 2 = W2
gateway, 4 = W3 env. Cell 0, the actions no witness kept, is not observable;
it is returned as `unseen`, which is the number a capture-recapture estimator
tries to recover.

Witness 3 (the LLM auditor) is in `auditor_records`, not `records`, and is
never counted here. The data holds a run's complete logs; every loss is
injected by this function, after the fact.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

CELLS = (1, 2, 3, 4, 5, 6, 7)
# The env diff's view: None sees every operation, a tuple only those operations.
ENV_VIEWS = {
    "full": None,
    "financial": ("issue_refund", "cancel_order", "send_email"),
}
ENV = 2  # the witness the view applies to


def load(config: str, root: str | Path = ".") -> list[dict]:
    """The rows of one episode config, e.g. load("confirm") from the dataset root."""
    path = Path(root) / "episodes" / f"{config}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def thin(
    rows: Iterable[dict],
    drop: Mapping[int, float] | None = None,
    env_view: str | Iterable[str] = "full",
    seed: int = 0,
) -> dict:
    """Capture-history counts after dropping records at `drop`, per witness.

    drop      witness id -> probability that one of its records is lost
              (0 = trace, 1 = gateway, 2 = env); a missing witness loses none.
    env_view  "full", "financial", or an iterable of operation names: the env
              witness (2) keeps only records of those operations.
    seed      the draw is reproducible from it.

    Returns {"n_true", "cells": {1..7: count}, "unseen"}. n_true is the sum of
    `n_true` over the rows, the ledger length; unseen is n_true minus the seven
    cells.
    """
    drop = dict(drop or {})
    view = ENV_VIEWS[env_view] if isinstance(env_view, str) else tuple(env_view)
    rng = random.Random(seed)
    cells = dict.fromkeys(CELLS, 0)
    n_true = 0
    for row in rows:
        seen = {e["id"]: 0 for e in row["ledger"]}
        for r in row["records"]:
            if r["oracle"] not in seen:
                continue
            if drop.get(r["witness"], 0.0) > 0 and rng.random() < drop[r["witness"]]:
                continue
            if r["witness"] == ENV and view is not None and r["op"] not in view:
                continue
            seen[r["oracle"]] |= 1 << r["witness"]
        for history in seen.values():
            if history:
                cells[history] += 1
        n_true += row["n_true"]
    return {"n_true": n_true, "cells": cells, "unseen": n_true - sum(cells.values())}
