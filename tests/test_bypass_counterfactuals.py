"""scripts/bypass_counterfactuals.py: the post hoc counterfactuals for the bypass study.

What is pinned:
- the `as_registered` world reproduces the registered cells of bypass-analysis-bypass-llm.json
  exactly (one cell, all 3000 replicates) and the replicate draws of `bypass_analysis.run_seed`;
- `parsed_scripts` adds one W1 and one W2 record per bypassed action, equal to the ledger entry,
  and nothing on a run with no bypass;
- `w3_full_view_on_bypassed` keeps the as-registered loss draws and only puts W3 records back;
- the hidden-mass classes add up to the hidden actions;
- the per-episode accounting agrees with bypass_analysis and the transcript;
- the artifact records argv and parameters and is labelled post hoc.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

sys.path.insert(0, str(ROOT / "scripts"))
import bypass_analysis as ba  # noqa: E402
import bypass_counterfactuals as bc  # noqa: E402

from lacuna.thinning import ENV_VIEWS, HELDOUT  # noqa: E402
from lacuna.witnesses import ENV, GATEWAY, TRACE  # noqa: E402

LLM, L1 = "bypass-llm", "bypass-l1"


def _rows_equal(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x["n"] == y["n"] and x["union"] == y["union"]
        for arm in ba.ARMS:
            for name in bc.ESTIMATORS:
                p, q = x["est"][arm][name], y["est"][arm][name]
                assert (p.point, p.lo, p.hi) == (q.point, q.lo, q.hi)


def test_as_registered_matches_bypass_analysis_draw_for_draw():
    cfg = HELDOUT[0]
    got = bc.run_seed(LLM, "as_registered", cfg, ba.SEED0 + 3, 5)
    want = ba.run_seed(LLM, cfg, ba.SEED0 + 3, 5, names=bc.ESTIMATORS)
    _rows_equal(got, want)


def test_as_registered_reproduces_a_committed_cell_exactly():
    """fin_mid, all three arms, registered seeds 98000-98029 x 100: the committed numbers."""
    committed = json.loads((ART / "robustness" / f"bypass-analysis-{LLM}.json").read_text())
    jobs = [
        (LLM, "as_registered", "fin_mid", s, ba.REPS_PER_SEED)
        for s in range(ba.SEED0, ba.SEED0 + ba.SEEDS)
    ]
    with Pool(4) as pool:
        done = pool.map(bc._job, jobs, chunksize=1)
    rows = [r for *_, rs in done for r in rs]
    for arm in ba.ARMS:
        cell = json.loads(json.dumps(bc._cell(rows, HELDOUT[0], arm)))
        want = committed["cells"][f"fin_mid/{arm}"]
        for k in ("replicates", "mean_n_true", "mean_union", "mean_hidden", "config", "arm"):
            assert cell[k] == want[k], (arm, k)
        for name in bc.ESTIMATORS:
            assert cell["estimators"][name] == want["estimators"][name], (arm, name)


def test_parsed_scripts_adds_one_trace_and_one_gateway_record_per_bypassed_action():
    for run in (LLM, L1):
        for base, par in zip(
            bc.worlds(run, "as_registered"), bc.worlds(run, "parsed_scripts"), strict=True
        ):
            b = base.bypassed
            led = {x.id: x for x in base.ep.ledger}
            extra = par.ep.records[len(base.ep.records) :]
            assert par.ep.records[: len(base.ep.records)] == base.ep.records
            assert len(extra) == 2 * len(b)
            for w in (TRACE, GATEWAY):
                got = {r.oracle: r for r in extra if r.witness == w}
                assert set(got) == b
                for i, r in got.items():
                    assert (r.op, r.resource, r.value) == (led[i].op, led[i].resource, led[i].value)
            assert ba.bypassed_ids(par.ep.ledger, par.ep.records) == set()


def test_direct_records_equal_their_ledger_entry_so_parsed_records_look_like_direct_ones():
    for run in (LLM, L1):
        for w in bc.worlds(run, "as_registered"):
            led = {x.id: x for x in w.ep.ledger}
            for r in w.ep.records:
                if r.witness in (TRACE, GATEWAY):
                    x = led[r.oracle]
                    assert (r.op, r.resource, r.value) == (x.op, x.resource, x.value)


def test_a_run_with_no_bypass_is_unchanged_by_parsing_scripts():
    assert all(not w.bypassed for w in bc.worlds("confirm", "as_registered"))
    for a, b in zip(
        bc.worlds("confirm", "as_registered"), bc.worlds("confirm", "parsed_scripts"), strict=True
    ):
        assert a.ep.records == b.ep.records
    cfg = HELDOUT[0]
    _rows_equal(
        bc.run_seed("confirm", "as_registered", cfg, ba.SEED0, 3),
        bc.run_seed("confirm", "parsed_scripts", cfg, ba.SEED0, 3),
    )
    assert bc.pool_description("confirm")["n_extra_records_parsed_scripts"] == 0


def test_w3_world_keeps_the_registered_draws_and_only_adds_w3_records():
    cfg = HELDOUT[0]
    view = ENV_VIEWS[cfg.env_view]
    ws = bc.worlds(LLM, "w3_full_view_on_bypassed")
    assert any(w.restore for w in ws)
    for w in ws:
        assert all(r.witness == ENV and r.oracle in w.bypassed for r in w.restore)
    assert any(r.op not in view for w in ws for r in w.restore)  # update_customer
    a = bc.run_seed(LLM, "as_registered", cfg, ba.SEED0, 4)
    b = bc.run_seed(LLM, "w3_full_view_on_bypassed", cfg, ba.SEED0, 4)
    for x, y in zip(a, b, strict=True):
        assert x["n"] == y["n"]
        assert y["union"]["oracle"] >= x["union"]["oracle"]
    # at a full view there is nothing to put back
    full = ba.CONFIGS[1]
    _rows_equal(
        bc.run_seed(LLM, "as_registered", full, ba.SEED0, 3),
        bc.run_seed(LLM, "w3_full_view_on_bypassed", full, ba.SEED0, 3),
    )


@pytest.mark.parametrize("world", ["as_registered", "parsed_scripts", "w3_full_view_on_bypassed"])
def test_hidden_classes_add_up_to_the_hidden_actions(world):
    cfg = HELDOUT[0]
    rows = bc.run_seed(LLM, world, cfg, ba.SEED0 + 1, 6)
    for r in rows:
        assert sum(r["hidden"].values()) == r["n"] - r["union"]["oracle"]
        assert {c for c, _ in r["hidden"]} <= set(bc.CLASSES)
    h = bc._hidden(rows)
    assert abs(sum(h["mean_hidden_by_class"].values()) - h["mean_hidden"]) < 0.02 * len(bc.CLASSES)


def test_zero_capture_class_is_always_hidden_while_the_logs_are_blind():
    """Bypassed and invisible to W3: no witness can keep it, so it is hidden in every replicate."""
    cfg = HELDOUT[0]
    rows = bc.run_seed(LLM, "as_registered", cfg, ba.SEED0, 5)
    pool = bc.pool_description(LLM)["fin_mid"]["by_class"]
    assert pool["bypassed_w3_invisible"] > 0
    for r in rows:
        zero = sum(k for (c, _), k in r["hidden"].items() if c == "bypassed_w3_invisible")
        assert zero > 0
    # in the recorded pool they are update_customer calls (and nothing W3 sees)
    ops = bc.pool_description(LLM)["fin_mid"]["by_class_op"]["bypassed_w3_invisible"]
    assert max(ops, key=ops.get) == "update_customer"


def test_pool_description_matches_bypass_analysis():
    for run in (LLM, L1):
        d, p = ba.describe(run), bc.pool_description(run)
        assert p["n_true"] == d["n_true"]
        assert (
            p["full_mid"]["by_class"]["bypassed_w3_invisible"]
            + p["full_mid"]["by_class"]["bypassed_w3_visible"]
            == d["n_bypassed"]
        )
        assert p["n_extra_records_parsed_scripts"] == 2 * d["n_bypassed"]


def test_reject_kind():
    ok = json.dumps({"results": [{"line": 1, "op": "send_email", "status": "done"}]})
    assert bc.reject_kind(ok) is None
    assert bc.reject_kind(json.dumps({"error": "script does not parse: SyntaxError"})) == "syntax"
    assert bc.reject_kind(json.dumps({"error": "more than 50 statements"})) == "too_long"
    err = json.dumps({"error": "line 2: only string and number literals are allowed as arguments"})
    assert bc.reject_kind(err) == "invalid"
    assert bc.reject_kind("not json") == "invalid"


@pytest.mark.parametrize("run", [LLM, L1])
def test_episode_stats_agree_with_bypass_analysis_and_the_transcript(run):
    from lacuna.envs import bypass

    stats = bc.episode_stats(run)
    eps = ba.load_episodes(run)
    assert len(stats) == len(eps)
    d = ba.describe(run)
    assert sum(s["n_bypassed"] for s in stats) == d["n_bypassed"]
    assert sum(s["n_script_calls"] for s in stats) == d["n_script_calls"]
    for s, e in zip(stats, eps, strict=True):
        assert s["n_script_calls"] == bypass.script_calls(e.transcript)
        assert s["n_bypassed"] == bypass.bypassed_actions(e.transcript)
        assert s["n_scripts_ran"] + s["n_scripts_rejected"] == s["n_script_calls"]
        assert sum(s["rejected_by_kind"].values()) == s["n_scripts_rejected"]
        assert s["bypass_share"] == pytest.approx(s["n_bypassed"] / s["n_ledger"], abs=1e-4)


def test_episode_9503_of_the_neutral_run_has_every_script_rejected():
    s = {e["seed"]: e for e in bc.episode_stats(LLM)}[9503]
    assert s["n_script_calls"] == 8 and s["n_scripts_rejected"] == 8 and s["n_bypassed"] == 0
    assert s["all_scripts_rejected"] is True
    assert s["rejected_by_kind"]["syntax"] == 4


def test_paired_summary_on_synthetic_episodes():
    def ep(seed, share, rej=0, calls=2):
        return {
            "seed": seed,
            "paired": 9500 <= seed < 9512,
            "n_ledger": 100,
            "n_bypassed": round(100 * share),
            "bypass_share": share,
            "all_scripts_rejected": bool(calls) and rej == calls,
        }

    neutral = [ep(9500, 0.8), ep(9503, 0.0, rej=2), ep(9509, 0.6), ep(9520, 0.9)]
    l1 = [ep(9500, 0.9), ep(9503, 0.8), ep(9509, 0.9)]
    p = bc.paired_summary(neutral, l1)
    assert p["seeds"] == [9500, 9503, 9509]
    assert p["l1_minus_neutral"] == {9500: 0.1, 9503: 0.8, 9509: 0.3}
    assert p["neutral_all_scripts_rejected"] == [9503]
    assert p["without_9503_9509"]["mean_l1_minus_neutral"] == 0.1
    assert p["without_9503_9509"]["seeds"] == [9500]


def test_seed_base_is_unused_elsewhere():
    taken = bc.SEED0
    key = f"{taken // 1000}_{taken % 1000:03d}"
    for p in (ROOT / "scripts").glob("*.py"):
        if p.name == "bypass_counterfactuals.py":
            continue
        assert key not in p.read_text(), p.name
    ba_seeds = set(range(ba.SEED0, ba.SEED0 + ba.SEEDS)) | set(
        range(ba.LOO_SEED0, ba.LOO_SEED0 + ba.LOO_SEEDS)
    )
    assert not ba_seeds & set(range(bc.SEED0, bc.SEED0 + bc.SEEDS))


def test_main_writes_a_labelled_artifact_with_argv_and_parameters(tmp_path):
    out = tmp_path / "x.json"
    argv = ["--seeds", "2", "--reps", "2", "--sanity-seeds", "1", "--processes", "1"]
    bc.main([*argv, "--out", str(out)])
    art = json.loads(out.read_text())
    assert art["argv"] == [*argv, "--out", str(out)]
    assert art["post_hoc"] is True and "not pre-registered" in art["status"]
    p = art["params"]
    assert p["seed0"] == 98_500 and p["seeds"] == 2 and p["reps_per_seed"] == 2
    assert p["replicates"] == 4 and p["seed_range"] == [98_500, 98_501]
    for run in (LLM, L1):
        r = art["runs"][run]
        assert set(r["worlds"]) == set(bc.WORLDS)
        assert set(r["worlds"]["as_registered"]) == {"fin_mid", "full_mid"}
        assert set(r["worlds"]["w3_full_view_on_bypassed"]) == {"fin_mid"}
        assert set(r["worlds"]["parsed_scripts"]["fin_mid"]) == set(ba.ARMS)
        assert set(r["worlds"]["parsed_scripts"]["fin_mid"]["callid"]["estimators"]) == set(
            bc.ESTIMATORS
        )
        assert set(r["hidden_mass"]) == set(bc.WORLDS)
    assert art["no_bypass_sanity"]["run"] == "confirm"
    assert set(art["episodes"]) == {LLM, L1}
    assert art["paired"]["seeds"] == list(bc.PAIRED_SEEDS)


def test_docstring_and_help_name_the_artifact_the_code_writes():
    assert "artifacts/robustness/bypass-counterfactuals.json" in bc.__doc__
    assert "POST HOC" in bc.__doc__


def test_committed_artifact():
    art = json.loads((ART / "robustness" / "bypass-counterfactuals.json").read_text())
    assert art["post_hoc"] is True
    p = art["params"]
    assert p["seed_range"] == [98_500, 98_529] and p["reps_per_seed"] == 100
    assert p["replicates"] == 3000
    for run in (LLM, L1):
        cells = art["runs"][run]["worlds"]["as_registered"]["fin_mid"]
        assert cells["callid"]["replicates"] == 3000
    assert len(art["episodes"][LLM]) == 24 and len(art["episodes"][L1]) == 12
