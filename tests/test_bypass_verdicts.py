"""scripts/bypass_verdicts.py: the verdict layer of the `backoffice_bypass` study.

Pinned: the three-way rules for B0 and B1 (every branch), the low-pool override, the bootstrap
conventions (seeds, resample counts, paired vs unpaired), the completion accounting, the model
check, and the numbers on the two free scripted runs (bypass-scripted-10 as "neutral",
bypass-scripted-30 as "l1"). The scripted analysis artifacts are regenerated small (3 loss seeds
of 10 replicates, a 5-replicate LOO) into a temp dir, so the tests are fast and deterministic;
the reference is the committed `confirm` artifact.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bypass_analysis as ba  # noqa: E402
import bypass_verdicts as bv  # noqa: E402

NEUTRAL, L1, REF = "bypass-scripted-10", "bypass-scripted-30", "confirm"
ART = ROOT / "artifacts" / "robustness"

# --- the three-way rules --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo", "hi", "want"),
    [
        (0.06, 0.20, "pass"),
        (0.0501, 0.2, "pass"),
        (0.05, 0.20, "inconclusive"),  # lower end equal to the threshold is not above it
        (0.03, 0.09, "inconclusive"),
        (0.01, 0.05, "inconclusive"),  # upper end equal to the threshold is not below it
        (0.0, 0.0499, "fail"),
        (0.0, 0.0, "fail"),
    ],
)
def test_b0_rule(lo, hi, want):
    assert bv.b0_verdict(lo, hi) == want


def _b1(cov, wil, unb, dev):
    return bv.b1_verdict(cov, wil, unb, dev)


def test_b1_pass_needs_all_three_conditions():
    assert _b1(0.95, (0.94, 0.96), 0.0, 0.02) == "pass"
    assert _b1(0.95, (0.90, 0.96), 0.0, 0.05) == "pass"  # boundaries are inclusive
    assert _b1(0.95, (0.89, 0.97), 0.0, 0.02) == "inconclusive"  # Wilson lower below 0.90
    assert _b1(0.95, (0.94, 0.96), 0.05, 0.02) == "inconclusive"  # unbounded rate not < 0.05
    assert _b1(0.95, (0.94, 0.96), 0.0, 0.06) == "inconclusive"  # LOO dev drags below 0.90


def test_b1_fail_needs_both_wilson_and_loo_on_the_failing_side():
    assert _b1(0.82, (0.80, 0.84), 0.0, 0.04) == "fail"
    assert _b1(0.86, (0.84, 0.88), 0.0, 0.04) == "inconclusive"  # 0.86 + 0.04 reaches 0.90
    assert _b1(0.88, (0.86, 0.90), 0.0, 0.01) == "inconclusive"  # Wilson upper not below 0.90
    # the unbounded rate plays no part in a fail
    assert _b1(0.82, (0.80, 0.84), 0.5, 0.01) == "fail"


def test_low_pool_makes_every_verdict_underpowered():
    assert bv.apply_pool_rule("pass", completed=17, minimum=18) == "underpowered"
    assert bv.apply_pool_rule("fail", completed=17, minimum=18) == "underpowered"
    assert bv.apply_pool_rule("inconclusive", completed=0, minimum=18) == "underpowered"
    assert bv.apply_pool_rule("pass", completed=18, minimum=18) == "pass"
    assert bv.apply_pool_rule("fail", completed=24, minimum=18) == "fail"


# --- the bootstraps ---------------------------------------------------------------------------


def test_share_bootstrap_is_the_documented_convention():
    b = np.array([0, 1, 5, 0, 2])
    n = np.array([10, 10, 20, 10, 10])
    lo, hi = bv.share_ci(b, n)
    rng = np.random.default_rng(98_100)
    idx = rng.integers(0, 5, size=(10_000, 5))
    s = b[idx].sum(1) / n[idx].sum(1)
    assert (lo, hi) == (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    assert bv.share_ci(b, n) == (lo, hi)  # deterministic


def test_paired_bootstrap_resamples_seed_indices_jointly():
    bn, nn = np.array([0, 2, 4, 1]), np.array([10, 10, 10, 10])
    bl, nl = np.array([1, 3, 9, 1]), np.array([10, 10, 10, 10])
    lo, hi = bv.paired_diff_ci(bn, nn, bl, nl)
    rng = np.random.default_rng(98_101)
    idx = rng.integers(0, 4, size=(10_000, 4))
    d = bl[idx].sum(1) / nl[idx].sum(1) - bn[idx].sum(1) / nn[idx].sum(1)
    assert (lo, hi) == (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def test_unpaired_bootstrap_resamples_each_run_on_its_own():
    bn, nn = np.array([0, 2, 4, 1]), np.array([10, 10, 10, 10])
    bl, nl = np.array([1, 3, 9]), np.array([10, 12, 10])
    lo, hi = bv.unpaired_diff_ci(bn, nn, bl, nl)
    rng = np.random.default_rng(98_102)
    i = rng.integers(0, 4, size=(10_000, 4))
    j = rng.integers(0, 3, size=(10_000, 3))
    d = bl[j].sum(1) / nl[j].sum(1) - bn[i].sum(1) / nn[i].sum(1)
    assert (lo, hi) == (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


# --- transcripts ------------------------------------------------------------------------------


def _asst(calls=True):
    m = {"role": "assistant", "content": None}
    if calls:
        m["tool_calls"] = [{"id": "c", "function": {"name": "x", "arguments": "{}"}}]
    return m


def test_turn_limit_needs_the_loop_to_have_run_out():
    assert bv.assistant_turns([_asst(), {"role": "tool"}, _asst(False)]) == 2
    full = [m for _ in range(400) for m in (_asst(), {"role": "tool"})]
    assert bv.hit_turn_limit(full, 400) is True
    ended = [m for _ in range(399) for m in (_asst(), {"role": "tool"})] + [_asst(False)]
    assert bv.assistant_turns(ended) == 400 and bv.hit_turn_limit(ended, 400) is False
    assert bv.hit_turn_limit(full[:-2], 400) is False


# --- the scripted-run table -------------------------------------------------------------------


@pytest.fixture(scope="module")
def adir(tmp_path_factory):
    d = tmp_path_factory.mktemp("analysis")
    for run in (NEUTRAL, L1):
        ba.main(
            ["--run", run, "--seeds", "3", "--reps", "10", "--processes", "1"]
            + [
                "--loo-seeds",
                "1",
                "--loo-reps",
                "5",
                "--out",
                str(d / f"bypass-analysis-{run}.json"),
            ]
        )
    shutil.copy(ART / f"bypass-analysis-{REF}.json", d)
    return d


def _argv(adir, *extra):
    return [
        *("--neutral", NEUTRAL, "--l1", L1, "--reference", REF),
        *("--analysis-dir", str(adir), "--skip-model-check"),
        *extra,
    ]


@pytest.fixture(scope="module")
def scripted(adir, tmp_path_factory):
    out = tmp_path_factory.mktemp("v") / "verdicts.json"
    bv.main(_argv(adir, "--out", str(out)))
    return json.loads(out.read_text())


def test_scripted_completion(scripted):
    for key, run in (("neutral", NEUTRAL), ("l1", L1)):
        c = scripted["completion"][key]
        assert c["run"] == run and c["completed"] == 24 and c["failed"] == []
        assert c["n_failed"] == 0 and c["stopped"] == ""
        assert c["turn_limit_episodes"] == 0 and c["turn_limit"] == 400
        assert c["model"] is None and c["base_url"] is None
        assert 0 < c["max_assistant_turns"] < 400
    assert scripted["completion"]["model_check"] == "skipped"


def test_scripted_b0(scripted):
    b0 = scripted["B0"]
    assert b0["n_bypassed"] == 154 and b0["n_true"] == 1474
    assert b0["share"] == pytest.approx(0.1045, abs=1e-4)
    assert b0["share_ge_0_10"] is True and b0["threshold"] == 0.05
    assert len(b0["per_episode_share"]) == 24
    assert b0["episodes_with_run_script"] == sum(1 for x in b0["per_episode_script_calls"] if x)
    assert sum(b0["per_episode_bypassed"]) == 154
    assert b0["bootstrap"] == {"resamples": 10_000, "rng_seed": 98_100, "level": 0.95}
    lo, hi = b0["ci95"]
    assert lo < b0["share"] < hi
    assert b0["verdict"] == bv.b0_verdict(lo, hi) == "pass"
    assert b0["ci95"] == [0.089, 0.1207]  # pinned
    assert b0["per_episode_distribution"]["max"] == 0.2


def test_scripted_b1_uses_the_artifact_cell_and_the_reference(scripted, adir):
    art = json.loads((adir / f"bypass-analysis-{NEUTRAL}.json").read_text())
    cell = art["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]
    b1 = scripted["B1"]
    assert b1["coverage"] == cell["coverage"] and b1["wilson95"] == cell["coverage_wilson95"]
    assert b1["unbounded_rate"] == cell["unbounded_rate"] and b1["mean_width"] == cell["mean_width"]
    loo = art["loo"]
    assert b1["loo"] == {k: loo[k] for k in ("reps_per_pool", "pools", "min", "max", "sd")}
    dev = max(abs(c - cell["coverage"]) for c in loo["coverage"])
    assert b1["loo_max_deviation"] == pytest.approx(dev, abs=1e-4)
    assert b1["verdict"] == bv.b1_verdict(
        cell["coverage"], cell["coverage_wilson95"], cell["unbounded_rate"], b1["loo_max_deviation"]
    )
    ref = json.loads((ART / f"bypass-analysis-{REF}.json").read_text())
    rc = ref["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]
    assert b1["reference"]["coverage"] == rc["coverage"] == 0.9457
    assert b1["reference"]["wilson95"] == [0.937, 0.9532] and b1["reference"]["mean_width"] == 38.96
    assert b1["reference"]["loo"]["min"] == 0.9167 and b1["reference"]["loo"]["max"] == 0.96
    assert b1["bypass_effect"] == pytest.approx(rc["coverage"] - cell["coverage"], abs=1e-4)
    assert b1["bypass_effect_gt_0_05"] == (b1["bypass_effect"] > 0.05)


def test_scripted_design_checks(scripted, adir):
    b2, b3 = scripted["B2"], scripted["B3"]
    art = json.loads((adir / f"bypass-analysis-{NEUTRAL}.json").read_text())
    est = art["cells"]["fin_mid/callid"]["estimators"]
    assert b2["seq_gap_tail_coverage"] == est["seq_gap_tail"]["coverage"] == 0.0
    assert b2["threshold"] == 0.5 and b2["check"] == "pass" and b2["verdict"] == "pass"
    assert b3["ll_bic_detects_gap"] == est["ll_bic"]["detects_gap"]
    assert b3["union_count_detects_gap"] == est["union_count"]["detects_gap"] == 0.0
    assert b3["difference"] == pytest.approx(b3["ll_bic_detects_gap"] - 0.0)


def test_scripted_l1(scripted):
    l1 = scripted["L1"]
    assert l1["neutral_share"] == scripted["B0"]["share"]
    assert l1["diff_pooled"] == pytest.approx(l1["l1_share"] - l1["neutral_share"], abs=1e-4)
    assert l1["matched_seeds"] == list(range(9100, 9124))
    assert l1["paired"]["resamples"] == 10_000 and l1["paired"]["rng_seed"] == 98_101
    assert l1["unpaired"]["resamples"] == 10_000 and l1["unpaired"]["rng_seed"] == 98_102
    assert l1["paired"]["diff"] == pytest.approx(l1["diff_pooled"], abs=1e-4)  # same 24 episodes
    for k in ("paired", "unpaired"):
        lo, hi = l1[k]["ci95"]
        assert lo < l1["diff_pooled"] < hi
    assert l1["paired"]["ci95"][0] > l1["unpaired"]["ci95"][0]  # pairing removes seed noise
    assert l1["diff_pooled"] == 0.1784 and l1["l1_share"] == 0.2829  # pinned
    assert l1["paired"]["ci95"] == [0.1623, 0.1953]
    assert l1["unpaired"]["ci95"] == [0.1542, 0.2024]
    assert l1["l1_ll_bic_fin_mid_callid"]["coverage"] > 0
    assert "verdict" not in l1


def test_scripted_pool_is_not_underpowered_but_a_higher_minimum_is(adir, tmp_path):
    out = tmp_path / "v.json"
    bv.main(_argv(adir, "--min-episodes", "25", "--out", str(out)))
    art = json.loads(out.read_text())
    assert art["underpowered"] is True
    for k in ("B0", "B1", "B2", "B3"):
        assert art[k]["verdict"] == "underpowered", k
    assert art["B0"]["raw_verdict"] == "pass"


def test_the_default_run_names_and_thresholds():
    args = bv.parser().parse_args([])
    assert (args.neutral, args.l1, args.reference) == ("bypass-llm", "bypass-l1", "confirm")
    assert args.expect_model == "deepseek-v4.1-flash" and args.min_episodes == 18
    assert args.turn_limit == 400 and args.skip_model_check is False


# --- refusals ---------------------------------------------------------------------------------


def test_model_mismatch_refuses_and_writes_nothing(adir, tmp_path):
    out = tmp_path / "v.json"
    argv = [*("--neutral", NEUTRAL, "--l1", L1, "--reference", REF)]
    argv += ["--analysis-dir", str(adir), "--out", str(out)]
    with pytest.raises(SystemExit) as e:
        bv.main(argv)  # scripted summaries have model None, default expects deepseek
    assert "model" in str(e.value) and not out.exists()
    bv.main([*argv, "--expect-model", "scripted"])
    assert json.loads(out.read_text())["completion"]["model_check"] == "ok"


def test_stale_analysis_artifact_is_refused(adir, tmp_path):
    d = tmp_path / "robustness"
    shutil.copytree(adir, d)
    p = d / f"bypass-analysis-{NEUTRAL}.json"
    art = json.loads(p.read_text())
    art["episodes_sha256"] = "0" * 64
    p.write_text(json.dumps(art))
    with pytest.raises(SystemExit) as e:
        bv.main(_argv(d, "--out", str(tmp_path / "v.json")))
    assert "sha256" in str(e.value)


def test_artifact_without_loo_is_refused(tmp_path):
    adir = tmp_path / "analysis"
    adir.mkdir()
    for run in (NEUTRAL, L1, REF):
        name = f"bypass-analysis-{run}.json"
        art = json.loads((ROOT / "artifacts/robustness" / name).read_text())
        if run == NEUTRAL:
            art.pop("loo", None)
        (adir / name).write_text(json.dumps(art))
    with pytest.raises(SystemExit) as e:
        bv.main(
            [*("--neutral", NEUTRAL, "--l1", L1, "--reference", REF)]
            + ["--skip-model-check", "--analysis-dir", str(adir), "--out", str(tmp_path / "v.json")]
        )
    assert "loo" in str(e.value)


def test_failed_episodes_are_reported_with_their_cause(adir, tmp_path, monkeypatch):
    runs = tmp_path / "episodes"
    for name in (NEUTRAL, L1, REF):
        (runs / name).mkdir(parents=True)
        for f in ("episodes.jsonl", "summary.json"):
            (runs / name / f).write_bytes((ROOT / "artifacts/episodes" / name / f).read_bytes())
    s = json.loads((runs / NEUTRAL / "summary.json").read_text())
    s["failed_episodes"] = [{"seed": 9999, "error": "HTTPError: 500", "diag": None}]
    (runs / NEUTRAL / "summary.json").write_text(json.dumps(s))
    monkeypatch.setattr(bv, "EPISODES", runs)
    out = tmp_path / "v.json"
    bv.main(_argv(adir, "--out", str(out)))
    c = json.loads(out.read_text())["completion"]["neutral"]
    assert c["failed"] == [{"seed": 9999, "error": "HTTPError: 500"}]
    assert c["n_failed"] == 1


def test_printed_table_names_every_test(adir, capsys, tmp_path):
    bv.main(_argv(adir, "--out", str(tmp_path / "v.json")))
    text = capsys.readouterr().out
    for k in ("completion", "B0", "B1", "B2", "B3", "L1", "wrote"):
        assert k in text


def test_output_is_deterministic(adir, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    bv.main(_argv(adir, "--out", str(a)))
    bv.main(_argv(adir, "--out", str(b)))
    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    for j in (ja, jb):
        j.pop("argv")
    assert ja == jb
