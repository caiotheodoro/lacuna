"""scripts/bypass_analysis.py: the frozen analysis for the `backoffice_bypass` study.

What is pinned:
- `callid_table` moved from scripts/linkage_arms.py to lacuna.linkage without a change in
  behaviour (golden digest taken from the script version before the move);
- the replicate loop uses the `--bootstrap-episodes` convention of scripts/thin_episodes.py:
  on the oracle and signature arms it reproduces `replay.replicates` draw for draw;
- the bypass accounting is exact on the two free scripted runs and agrees with the run summary;
- one seed is deterministic, and the summary statistics are what they say;
- the committed artifacts record argv and parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

sys.path.insert(0, str(ROOT / "scripts"))
import bypass_analysis as ba  # noqa: E402
import linkage_arms  # noqa: E402
import replay  # noqa: E402

from lacuna import linkage  # noqa: E402
from lacuna.thinning import HELDOUT, Config, thin  # noqa: E402
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record  # noqa: E402

RUN10, RUN30 = "bypass-scripted-10", "bypass-scripted-30"

# sha256 over callid_table(...) for 8 episodes each of `confirm` and `bypass-scripted-30`, two
# loss configs, two keys, computed with the function in scripts/linkage_arms.py at 78948ca.
CALLID_GOLDEN = "3f8eec7308fa07a151860247ed3fac65acecbbdc41afab00603a348a89b00ef9"


def _sig_key(r: Record):
    return linkage.signature(r.op, r.resource, r.value)


def test_callid_table_moved_unchanged():
    assert linkage_arms.callid_table is linkage.callid_table
    h = hashlib.sha256()
    for run in ("confirm", RUN30):
        eps = linkage_arms.episodes(run)
        for cfg in (HELDOUT[0], Config("full_mid", 0.5, 0.5, "full")):
            rng = np.random.default_rng(7)
            for _led, recs in eps[:8]:
                kept = thin(recs, cfg, rng)
                for key in (_sig_key, linkage_arms.digest_signature):
                    h.update(np.asarray(linkage.callid_table(kept, key), dtype=np.int64).tobytes())
    assert h.hexdigest() == CALLID_GOLDEN


def test_callid_table_bypassed_action_is_a_w3_only_cell():
    recs = [
        Record(TRACE, "cancel_order", "O1", "cancelled", oracle=0),
        Record(GATEWAY, "cancel_order", "O1", "cancelled", oracle=0),
        Record(ENV, "cancel_order", "O1", "cancelled", oracle=0),
        Record(ENV, "cancel_order", "O2", "cancelled", oracle=1),  # run_script: no W1/W2 line
    ]
    t = linkage.callid_table(recs, _sig_key)
    assert t.tolist() == [0, 0, 0, 0, 1, 0, 0, 1]


# --- the bypass accounting ---------------------------------------------------------------


@pytest.mark.parametrize("run", [RUN10, RUN30])
def test_descriptive_matches_the_run_summary_and_the_definition(run):
    d = ba.describe(run)
    s = json.loads((ART / "episodes" / run / "summary.json").read_text())["bypass"]
    assert d["n_true"] == s["n_true"]
    assert d["n_bypassed"] == s["n_bypassed"]
    assert d["n_bypassed_from_scripts"] == s["n_bypassed"]
    assert d["bypassed_share"] == pytest.approx(s["n_bypassed"] / s["n_true"], abs=1e-4)
    assert d["n_script_calls"] == s["n_script_calls"]
    assert d["capture"] == {k: s["capture"][k] for k in ("trace", "gateway", "env")}
    assert d["n_bypassed"] == d["n_true"] - d["capture"]["gateway"]
    seen_by_w3 = sum(s["env_saw_gateway_blind_by_op"].values())
    assert d["bypassed_seen_by_w3"] == seen_by_w3
    assert d["bypassed_w3_share"] == pytest.approx(seen_by_w3 / s["n_bypassed"], abs=1e-4)
    assert d["bypassed_by_op"] == s["bypassed_by_op"]


def test_bypass_definition_counts_ledger_actions_without_a_gateway_record():
    ledger = [
        {"id": i, "op": "cancel_order", "resource": f"O{i}", "value": "x", "step": i}
        for i in range(3)
    ]
    gw = [Record(GATEWAY, "cancel_order", "O0", "cancelled", oracle=0)]
    ids = ba.bypassed_ids(ledger, gw)
    assert ids == {1, 2}


# --- the replicate loop ------------------------------------------------------------------


def test_oracle_and_signature_arms_reproduce_replay_replicates():
    eps = ba.load_episodes(RUN30)
    cfg = HELDOUT[0]
    got = ba.run_seed(RUN30, cfg, seed=98_003, reps=4)
    for arm in ("oracle", "signature"):
        want = replay.replicates(
            [(e.ledger, e.records) for e in eps], cfg, arm, 4, 98_003, ("ll_independence", "ll_bic")
        )
        for g, w in zip(got, want, strict=True):
            assert g["n"] == w["n"] and g["union"][arm] == w["union"]
            for name in ("ll_independence", "ll_bic"):
                a, b = g["est"][arm][name], w[name]
                assert (a.point, a.lo, a.hi) == (b.point, b.lo, b.hi)


def test_one_seed_is_deterministic_and_arms_share_the_draw():
    cfg = Config("full_mid", 0.5, 0.5, "full")
    a = ba.run_seed(RUN10, cfg, seed=98_001, reps=3)
    b = ba.run_seed(RUN10, cfg, seed=98_001, reps=3)
    assert [r["n"] for r in a] == [r["n"] for r in b]
    for ra, rb in zip(a, b, strict=True):
        for arm in ba.ARMS:
            for name in ba.ESTIMATORS:
                x, y = ra["est"][arm][name], rb["est"][arm][name]
                assert (x.point, x.lo, x.hi) == (y.point, y.lo, y.hi)
    # every arm saw every estimator
    assert set(a[0]["est"]) == set(ba.ARMS)
    assert all(set(a[0]["est"][arm]) == set(ba.ESTIMATORS) for arm in ba.ARMS)


def test_seq_gap_tail_counts_routed_actions_only():
    eps = ba.load_episodes(RUN30)
    seq = ba.gateway_numbers(eps[0].records)
    n_gateway = sum(1 for r in eps[0].records if r.witness == GATEWAY)
    assert len(seq) == n_gateway < len(eps[0].ledger)
    assert sorted(seq.values()) == list(range(n_gateway))


# --- the summary -------------------------------------------------------------------------


def test_wilson_interval():
    lo, hi = ba.wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-3) and hi == pytest.approx(0.9785, abs=1e-3)
    assert ba.wilson(0, 10)[0] == 0.0 and ba.wilson(10, 10)[1] == 1.0


def test_summarise_fields():
    from lacuna.estimators import Estimate

    rows = [
        {"n": 100, "union": 90, "x": Estimate("x", 90, 100.0, 95.0, 105.0)},  # covers
        {"n": 100, "union": 90, "x": Estimate("x", 90, 94.0, 91.0, 97.0)},  # misses
        {"n": 100, "union": 90, "x": Estimate("x", 90, 90.0, 90.0, math.inf)},  # unbounded
        {"n": 100, "union": 90, "x": Estimate("x", 90, 90.0, 90.0, 90.0)},  # misses
    ]
    s = ba.summarise([r["x"] for r in rows], [r["n"] for r in rows], [r["union"] for r in rows])
    assert s["reps"] == 4
    assert s["coverage"] == 0.5
    assert s["coverage_mcse"] == pytest.approx(0.25)
    assert s["coverage_wilson95"] == [pytest.approx(x, abs=1e-4) for x in ba.wilson(2, 4)]
    assert s["unbounded_rate"] == 0.25
    assert s["mean_width"] == pytest.approx((10 + 6 + 0) / 3, abs=0.01)
    assert s["bias"] == pytest.approx((0 - 6 - 10 - 10) / 4, abs=0.01)
    assert s["rmse"] == pytest.approx(math.sqrt((0 + 36 + 100 + 100) / 4), abs=0.01)
    assert s["detects_gap"] == 0.5  # lo 95 and 91 exceed the union of 90


# --- the CLI and the artifacts -----------------------------------------------------------


def test_defaults_and_configs():
    args = ba.parser().parse_args(["--run", RUN10])
    assert (args.seed0, args.seeds, args.reps) == (98_000, 30, 100)
    names = [c.name for c in ba.CONFIGS]
    assert names == ["fin_mid", "full_mid", "fin_mid_rho100"]
    assert ba.CONFIGS[0] == HELDOUT[0] and ba.CONFIGS[2] == HELDOUT[1]
    assert ba.CONFIGS[1] == Config("full_mid", 0.5, 0.5, "full")
    assert ba.ARMS == ("callid", "signature", "oracle")
    assert ba.ESTIMATORS == (
        "ll_bic",
        "ll_independence",
        "union_count",
        "chapman_w1w2",
        "chapman_w1w3",
        "chapman_w2w3",
        "seq_gap_tail",
    )


def test_seed_base_is_unused_elsewhere():
    taken = ba.SEED0
    for p in (ROOT / "scripts").glob("*.py"):
        if p.name == "bypass_analysis.py":
            continue
        assert f"{taken // 1000}_{taken % 1000:03d}" not in p.read_text(), p.name


def test_main_writes_an_artifact_with_argv_and_parameters(tmp_path):
    out = tmp_path / "x.json"
    ba.main(["--run", RUN10, "--seeds", "2", "--reps", "2", "--processes", "1", "--out", str(out)])
    art = json.loads(out.read_text())
    assert art["run"] == RUN10
    assert art["argv"] == [
        "--run",
        RUN10,
        "--seeds",
        "2",
        "--reps",
        "2",
        "--processes",
        "1",
        "--out",
        str(out),
    ]
    p = art["params"]
    assert p["seed0"] == 98_000 and p["seeds"] == 2 and p["reps_per_seed"] == 2
    assert p["replicates"] == 4 and p["bootstrap_episodes"] is True
    assert set(art["cells"]) == {
        f"{c}/{a}" for c in ("fin_mid", "full_mid", "fin_mid_rho100") for a in ba.ARMS
    }
    cell = art["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]
    for k in (
        "coverage",
        "coverage_mcse",
        "coverage_wilson95",
        "mean_width",
        "bias",
        "rmse",
        "detects_gap",
        "unbounded_rate",
    ):
        assert k in cell
    assert art["descriptive"]["n_true"] > 0


@pytest.mark.parametrize("run", [RUN10, RUN30])
def test_committed_artifacts(run):
    art = json.loads((ART / "robustness" / f"bypass-analysis-{run}.json").read_text())
    assert art["params"]["replicates"] == 3000
    assert art["params"]["seed0"] == 98_000 and art["params"]["seeds"] == 30
    assert art["argv"][:2] == ["--run", run]
    assert art["descriptive"]["n_bypassed"] > 0
    assert len(art["cells"]) == 9
