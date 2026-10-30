"""The bypass cells of scripts/baselines_battery.py and artifacts/robustness/baselines-bypass.json.

`bypass_b*` (baselines.json) removes only the gateway record of a bypassed action. In the
`backoffice_bypass` env the harness trace misses those actions too, so the `bypass_both_b*`
cells drop both, and plant canaries after the bypass (an auditor's canaries are never
bypassed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "robustness"

sys.path.insert(0, str(ROOT / "scripts"))
import baselines_battery as bb  # noqa: E402


def test_main_bypass_cells_are_gateway_only_and_the_new_ones_drop_both():
    assert all(not c.bypass_trace for c in bb.cells())
    new = bb.bypass_cells()
    assert [c.bypass for c in new] == [0.0, 0.05, 0.1, 0.2, 0.3]
    assert all(c.bypass_trace and c.canary_variants == {"episode_mix": {}} for c in new)
    assert {c.name for c in new}.isdisjoint(c.name for c in bb.cells())


def test_a_full_bypass_leaves_only_the_env_witness_and_never_touches_canaries():
    # Every real action bypasses W1 and W2, so only W3 (and the canaries) see anything.
    cell = bb.Cell(
        "all", "", bb._cfg(bb.Config("none", 0.0, 0.0, "full")), bypass=1.0, bypass_trace=True
    )
    rows = bb.one_rep(cell, np.random.default_rng(0))
    plain = rows["oracle/plain"][0]
    canary = rows["oracle/canary_episode_mix"][0]
    assert plain["union"] > 0
    # Under no loss and a full bypass the union is what W3 alone saw; the canary world's
    # union equals it, so the canaries (which are never bypassed) do not leak into it.
    assert canary["union"] == plain["union"]
    chapman = plain["chapman_w1w2"]
    assert chapman.n_observed == 0


def test_the_artifact_has_the_five_cells_and_b0_matches_per_record():
    art = json.loads((ART / "baselines-bypass.json").read_text())
    base = json.loads((ART / "baselines.json").read_text())
    assert art["set"] == "bypass" and len(art["cells"]) == 5
    assert all(
        c["bypass"] == b for c, b in zip(art["cells"].values(), bb.BYPASS_RATES, strict=True)
    )
    for lk in ("oracle", "signature"):
        got = art["cells"]["bypass_both_b0/fin_mid"]["linkage"][lk]["estimators"]["ll_bic"]
        want = base["cells"]["per_record"]["linkage"][lk]["estimators"]["ll_bic"]
        assert got == want
