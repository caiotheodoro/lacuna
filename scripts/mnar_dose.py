#!/usr/bin/env python3
"""Dose response for the two MNAR mechanisms of section 7 (D44).

The paper's MNAR results were one point each: a 1.8x drop multiplier. This
sweeps the multiplier, under both linkages with the same random numbers, and
separates the hidden-key configuration into its two parts: W3 blind to the
keyed actions, and the logs dropping them more often.

Op-keyed: both logs' drop rate times m for update_customer and write_file,
fin_mid base (0.5/0.5, financial env view). Hidden-key: every third ledger
id, full env view, with W3 blind to them or not, logs' drop rate times m.

Exploratory, after the confirmatory run.

Usage: uv run python scripts/mnar_dose.py
"""

from __future__ import annotations

import json
from pathlib import Path

from replay import replicates, summarise
from thin_episodes import load

from lacuna.thinning import Config

ROOT = Path(__file__).resolve().parents[1]
REPS = 300
EST = ("ll_independence", "ll_bic", "strat_bic")


def op_keyed(m: float) -> Config:
    bias = (("update_customer", m), ("write_file", m)) if m != 1.0 else ()
    return Config(f"op_x{m:g}", 0.5, 0.5, "financial", op_bias=bias)


def hidden_key(m: float, blind: bool) -> Config:
    return Config(
        f"hidden_x{m:g}{'_w3blind' if blind else ''}",
        0.5,
        0.5,
        "full",
        hidden_mult=m,
        hidden_env_blind=blind,
    )


def main() -> None:
    episodes = load("confirm")
    out: dict = {"run": "confirm", "reps": REPS, "op_keyed": {}, "hidden_key": {}}
    for i, m in enumerate((1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8)):
        for linkage in ("oracle", "signature"):
            rows = replicates(episodes, op_keyed(m), linkage, REPS, 20_000 + i, EST)
            out["op_keyed"][f"{m:g}/{linkage}"] = {e: summarise(rows, e) for e in EST}
            print(
                "op",
                m,
                linkage,
                {e: out["op_keyed"][f"{m:g}/{linkage}"][e]["coverage"] for e in EST},
                flush=True,
            )
    for i, (m, blind) in enumerate(
        [
            (1.0, True),
            (1.2, True),
            (1.4, True),
            (1.8, True),
            (1.2, False),
            (1.4, False),
            (1.8, False),
        ]
    ):
        rows = replicates(episodes, hidden_key(m, blind), "signature", REPS, 21_000 + i, EST)
        key = f"{m:g}/{'w3_blind' if blind else 'w3_sees'}"
        out["hidden_key"][key] = {e: summarise(rows, e) for e in EST}
        print("hidden", key, {e: out["hidden_key"][key][e]["coverage"] for e in EST}, flush=True)
    dst = ROOT / "artifacts" / "robustness" / "mnar-dose.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
