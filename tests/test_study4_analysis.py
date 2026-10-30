"""scripts/study4_analysis.py: the replication and adversary pools of study 4.

Scripted runs generated at $0 into a temp tree stand in for the paid LLM runs (the scripted agent
ignores the prompt, so the plumbing is what is checked); the existing artifacts check that the
script reproduces numbers already on disk. Everything runs serially (`--processes 1`): a spawned
worker would not see the patched roots.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bypass_analysis as ba  # noqa: E402
import bypass_counterfactuals as cf  # noqa: E402
import bypass_verdicts as bv  # noqa: E402
import run_episodes  # noqa: E402
import study4_analysis as s4  # noqa: E402
import study4_lib as lib  # noqa: E402

from lacuna.envs import bypass  # noqa: E402


def bypass_l2_sha() -> str:
    return hashlib.sha256(bypass.L2_PROMPT.encode()).hexdigest()


SMALL = ["--processes", "1", "--seeds", "2", "--reps", "5", "--loo-seeds", "1", "--loo-reps", "4"]
CELLS = [f"{c}/{a}" for c in ("fin_mid", "full_mid", "fin_mid_rho100") for a in ba.ARMS]


def _scripted(root: Path, name: str, *argv: str) -> None:
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(run_episodes, "ROOT", root)
        mp.setattr(
            sys, "argv",
            ["run_episodes.py", "--agent", "scripted", "--tickets", "8", "--p-gateway-drop", "0",
             "--p-trace-drop", "0", "--run", name, *argv],
        )  # fmt: skip
        run_episodes.main()
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("study4")
    bo = ("--env", "backoffice_bypass", "--bypass-fraction", "0.5", "--episodes", "4")
    _scripted(root, "rep", "--env", "backoffice", "--episodes", "4", "--seed0", "100")
    _scripted(root, "adv", *bo, "--seed0", "200", "--organism", "l2")
    _scripted(root, "neu_same", *bo, "--seed0", "200", "--organism", "neutral")
    _scripted(root, "neu_other", *bo, "--seed0", "300", "--organism", "neutral")
    _scripted(
        root, "adv_lossy", "--env", "backoffice_bypass", "--episodes", "1", "--seed0", "200",
        "--organism", "l2", "--p-gateway-drop", "0.05", "--p-trace-drop", "0.1",
    )  # fmt: skip
    return root


@pytest.fixture
def fresh(tree, monkeypatch):
    """Point every module at the temp tree and drop caches keyed by run name."""
    for mod in (ba, cf, s4):
        monkeypatch.setattr(mod, "ROOT", tree)
    monkeypatch.setattr(bv, "EPISODES", tree / "artifacts" / "episodes")
    monkeypatch.setattr(bv, "ROBUSTNESS", tree / "artifacts" / "robustness")
    caches = (ba.load_episodes, cf.worlds, cf.classes)
    for c in caches:
        c.cache_clear()
    lib.clear_caches()
    yield tree
    for c in caches:
        c.cache_clear()
    lib.clear_caches()


def args(*argv: str) -> list[str]:
    """`argv` with the scripted runs' model check unless the caller names a model."""
    return (
        [*argv, *SMALL]
        if "--expect-model" in argv
        else [*argv, "--expect-model", "scripted", *SMALL]
    )


def run(out: Path, *argv: str) -> dict:
    s4.main([*args(*argv), "--out", str(out)])
    return json.loads(out.read_text())


# --- replication ------------------------------------------------------------------------------


def test_replication_pool_cells_and_loo(fresh, tmp_path):
    r = run(tmp_path / "r.json", "--run", "rep", "--pool", "replication")
    assert r["pool"] == "replication" and r["run"] == "rep"
    assert set(r["cells"]) == {"fin_mid/callid", "fin_mid/signature"}
    assert r["params"]["seed_range"] == [ba.SEED0, ba.SEED0 + 1]
    for arm in ("callid", "signature"):
        c = r["cells"][f"fin_mid/{arm}"]
        assert c["replicates"] == 10
        assert set(c["estimators"]) == set(s4.REPLICATION_ESTIMATORS)
        b = c["estimators"]["ll_bic"]
        assert 0 <= b["coverage_wilson95"][0] <= b["coverage"] <= b["coverage_wilson95"][1] <= 1
        assert len(c["ll_bic_seed_coverage"]["values"]) == 2
        lo = r["loo"]["arms"][arm]
        assert len(lo["coverage"]) == 4 == r["loo"]["pools"]
        assert lo["max_deviation"] == pytest.approx(
            max(abs(x - b["coverage"]) for x in lo["coverage"]), abs=1e-4
        )
        h = r["headline"][arm]
        assert h["coverage"] == b["coverage"] and h["loo_max_deviation"] == lo["max_deviation"]
    assert r["descriptive"]["n_bypassed"] == 0 and r["descriptive"]["n_script_calls"] == 0
    assert r["underpowered"] is True  # 4 episodes < 18
    assert r["reference"] is None  # no confirm artifact in the temp tree


def test_replication_cells_are_the_frozen_analysis_cells(fresh, tmp_path):
    r = run(tmp_path / "r.json", "--run", "rep", "--pool", "replication")
    full = ba.analyse("rep", ba.SEED0, 2, 5, 1)
    for arm in ("callid", "signature"):
        for name in s4.REPLICATION_ESTIMATORS:
            assert (
                r["cells"][f"fin_mid/{arm}"]["estimators"][name]
                == full[f"fin_mid/{arm}"]["estimators"][name]
            )


def test_output_is_deterministic_and_never_overwritten(fresh, tmp_path):
    a = run(tmp_path / "a.json", "--run", "rep", "--pool", "replication")
    b = run(tmp_path / "b.json", "--run", "rep", "--pool", "replication")
    a["argv"] = b["argv"] = None
    assert a == b
    before = (tmp_path / "a.json").read_bytes()
    with pytest.raises(SystemExit):
        out = str(tmp_path / "a.json")
        s4.main(args("--run", "rep", "--pool", "replication") + ["--out", out])
    assert (tmp_path / "a.json").read_bytes() == before


def test_seeds_change_the_draws(fresh, tmp_path):
    a = run(tmp_path / "a.json", "--run", "rep", "--pool", "replication")
    b = run(tmp_path / "b.json", "--run", "rep", "--pool", "replication", "--seed0", "5")
    assert a["cells"]["fin_mid/callid"] != b["cells"]["fin_mid/callid"]


# --- adversary --------------------------------------------------------------------------------


def test_adversary_pool_matched_seeds(fresh, tmp_path):
    r = run(
        tmp_path / "adv.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same",
        "--cf-seed0", "7",
    )  # fmt: skip
    assert set(r["cells"]) == set(CELLS)
    assert r["run_summary"]["organism"] == "l2"
    assert r["run_summary"]["organism_prompt_sha256"] == bypass_l2_sha()
    b = r["bypass"]
    d = ba.describe("adv")
    assert (b["n_true"], b["n_bypassed"]) == (d["n_true"], d["n_bypassed"])
    assert b["n_bypassed"] > 0 and b["ci95"][0] <= b["share"] <= b["ci95"][1]
    assert "raw_verdict" not in b and "threshold" not in b
    v = r["vs_neutral"]
    assert v["matched_seeds"] == [200, 201, 202, 203]
    assert v["paired"]["n_pairs"] == 4 and v["unpaired"]["n_neutral"] == 4
    assert v["diff_pooled"] == pytest.approx(v["treatment_share"] - v["neutral_share"], abs=1e-4)
    assert v["treatment_organism"] == "l2"
    assert r["loo"]["arms"]["callid"]["coverage"] and r["loo"]["pools"] == 4


def test_adversary_cells_are_the_frozen_analysis_cells(fresh, tmp_path):
    r = run(tmp_path / "adv.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same")
    full = ba.analyse("adv", ba.SEED0, 2, 5, 1)
    for key, cell in full.items():
        assert r["cells"][key]["estimators"] == cell["estimators"]
        assert r["cells"][key]["mean_hidden"] == cell["mean_hidden"]


def test_parsed_scripts_control_is_the_counterfactual_world(fresh, tmp_path):
    r = run(
        tmp_path / "adv.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same",
        "--cf-seed0", "7",
    )  # fmt: skip
    ctl = r["parsed_scripts_control"]
    want = json.loads(json.dumps(cf.analyse(["adv"], ("parsed_scripts",), 7, 2, 5, 1)["adv"]))
    assert ctl["worlds"] == want["worlds"]["parsed_scripts"]
    assert ctl["hidden_mass"] == want["hidden_mass"]["parsed_scripts"]
    assert set(ctl["worlds"]) == {"fin_mid", "full_mid"}
    assert len(ctl["episodes"]) == 4
    cov = ctl["worlds"]["fin_mid"]["callid"]["estimators"]["ll_bic"]["coverage"]
    assert r["headline"]["parsed_scripts_coverage"] == cov


def test_unpaired_when_seeds_do_not_match(fresh, tmp_path):
    r = run(tmp_path / "adv.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_other")
    v = r["vs_neutral"]
    assert v["matched_seeds"] == [] and v["paired"] is None
    lo, hi = v["unpaired"]["ci95"]
    assert lo <= hi and v["unpaired"]["diff"] == v["diff_pooled"]
    assert v["unpaired"]["rng_seed"] == bv.SEED_UNPAIRED


# --- run checks -------------------------------------------------------------------------------


def test_adversary_needs_an_l2_run(fresh, tmp_path):
    with pytest.raises(SystemExit, match="organism l2"):
        s4.main(args("--run", "neu_same", "--pool", "adversary") + ["--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()


def test_expect_organism_lifts_the_check(fresh, tmp_path):
    r = run(
        tmp_path / "n.json", "--run", "neu_same", "--pool", "adversary", "--neutral", "neu_other",
        "--expect-organism", "neutral",
    )  # fmt: skip
    assert r["run_summary"]["organism"] == "neutral"


def test_replication_rejects_a_bypass_run_and_adversary_a_backoffice_run(fresh, tmp_path):
    for run_, pool in (("adv", "replication"), ("rep", "adversary")):
        with pytest.raises(SystemExit):
            s4.main(args("--run", run_, "--pool", pool) + ["--out", str(tmp_path / "x")])


def test_lossy_logs_are_rejected(fresh, tmp_path):
    with pytest.raises(SystemExit, match="complete logs"):
        s4.main(args("--run", "adv_lossy", "--pool", "adversary") + ["--out", str(tmp_path / "x")])


def test_missing_run_is_a_usage_error(fresh, tmp_path):
    with pytest.raises(SystemExit):
        s4.main(args("--run", "nope", "--pool", "replication") + ["--out", str(tmp_path / "x")])


# --- existing artifacts -----------------------------------------------------------------------


def test_adversary_script_on_the_stored_l1_run_reproduces_the_stored_numbers(tmp_path):
    out = tmp_path / "l1.json"
    r = run(out, "--run", "bypass-l1", "--pool", "adversary", "--expect-organism", "l1",
            "--neutral", "bypass-llm", "--expect-model", "deepseek-v4.1-flash")  # fmt: skip
    verdicts = json.loads((ROOT / "artifacts/robustness/bypass-verdicts.json").read_text())["L1"]
    v = r["vs_neutral"]
    assert v["diff_pooled"] == verdicts["diff_pooled"]
    assert v["paired"]["ci95"] == verdicts["paired"]["ci95"]
    assert v["unpaired"]["ci95"] == verdicts["unpaired"]["ci95"]
    assert v["neutral_ll_bic_fin_mid_callid"]["coverage"] == 0.759
    assert r["bypass"]["share"] == verdicts["l1_share"]
    assert r["completion"]["model"] == "deepseek-v4.1-flash"


# --- blocks added after the draft review (D69) ------------------------------------------------


def test_adversary_pool_has_the_review_blocks(fresh, tmp_path):
    r = run(tmp_path / "adv.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same")
    w = r["witness_routes"]
    assert w["n_unwitnessed_other"] == 0 and w["accounting_consistent"]
    assert w["n_bypassed_minus_from_scripts"] == 0 and len(w["per_episode"]) == 4
    tp = r["text_parsed_control"]
    assert set(tp["worlds"]) == {"fin_mid", "full_mid"}
    assert set(tp["worlds"]["fin_mid"]) == {"callid", "signature"}
    assert tp["pool"]["scripts_rejected"] == 0 and tp["pool"]["lines_false_extra"] == 0
    assert set(tp["worlds_independent_loss"]) == {"fin_mid", "full_mid"}
    for cfg_name, arms in tp["worlds_independent_loss"].items():  # no failing script, so equal
        for arm, cell in arms.items():
            assert cell == r["parsed_scripts_control"]["worlds"][cfg_name][arm]
    lo = tp["loo"]
    assert lo["pools"] == 4 and set(lo["arms"]) == {"callid"} and lo["seed0"] == ba.LOO_SEED0
    cov = tp["worlds"]["fin_mid"]["callid"]["estimators"]["ll_bic"]["coverage"]
    assert lo["arms"]["callid"]["max_deviation"] == pytest.approx(
        max(abs(x - cov) for x in lo["arms"]["callid"]["coverage"]), abs=1e-4
    )
    assert tp["headline"]["callid"]["coverage"] == cov
    assert r["headline"]["text_parsed_coverage"] == cov
    ps = r["parsed_scripts_control"]
    assert ps["loo"]["pools"] == 4 and "max_deviation" in ps["loo"]["arms"]["callid"]
    assert (
        ps["headline"]["callid"]["loo_max_deviation"]
        == ps["loo"]["arms"]["callid"]["max_deviation"]
    )
    c = r["competence"]
    assert c["episodes_completed"] == 4 and c["tickets"] > 0 and c["invalid_calls"]["total"] == 0
    b = r["behaviour"]
    assert len(b["per_episode"]) == 4 and b["mention_regex"] and b["refusal_regex"]
    assert r["model_check"] == {"expected": None, "recorded": None, "status": "ok"}
    assert "base_url_host" in r["run_summary"]


def test_text_parsed_control_is_deterministic_and_uses_the_control_seeds(fresh, tmp_path):
    a = run(tmp_path / "a.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same")
    b = run(
        tmp_path / "b.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same",
        "--cf-seed0", "5",
    )  # fmt: skip
    assert a["text_parsed_control"]["seed0"] == cf.SEED0 and b["text_parsed_control"]["seed0"] == 5
    assert a["text_parsed_control"]["worlds"] != b["text_parsed_control"]["worlds"]
    again = run(tmp_path / "c.json", "--run", "adv", "--pool", "adversary", "--neutral", "neu_same")
    assert again["text_parsed_control"] == a["text_parsed_control"]


def test_replication_pool_has_competence(fresh, tmp_path):
    r = run(tmp_path / "r.json", "--run", "rep", "--pool", "replication")
    c = r["competence"]
    assert c["episodes_completed"] == 4 and c["n_true"] == r["descriptive"]["n_true"]
    assert c["mean_n_per_episode"] == round(c["n_true"] / 4, 4)
    assert "witness_routes" not in r and "behaviour" not in r


def test_expect_model_mismatch_exits_nonzero_and_writes_nothing(fresh, tmp_path):
    out = tmp_path / "x.json"
    wrong = args("--run", "rep", "--pool", "replication", "--expect-model", "qwen3.8-flash")
    with pytest.raises(SystemExit) as e:
        s4.main([*wrong, "--out", str(out)])
    assert e.value.code not in (0, None) and "expected" in str(e.value.code)
    assert not out.exists()
    with pytest.raises(SystemExit) as e:
        s4.main([*SMALL, "--run", "rep", "--pool", "replication", "--out", str(out)])
    assert e.value.code == 2  # --expect-model is required
