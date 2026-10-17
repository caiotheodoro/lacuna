from __future__ import annotations

import math

import numpy as np
import pytest

from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
from lacuna.episode import run_episode
from lacuna.thinning import CONFIGS, Config, estimate_all, stratified, tables, thin
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record


def _episode(seed: int = 11):
    return run_episode(
        seed,
        ScriptedAgent(np.random.default_rng(seed)),
        ScriptedAuditor(np.random.default_rng(1), p_miss=0.0),
        p_gateway_drop=0.0,
        p_trace_drop=0.0,
    )


def test_zero_loss_full_view_keeps_every_record():
    res = _episode()
    kept = thin(res.records, Config("none", 0.0, 0.0, "full"), np.random.default_rng(0))
    assert kept == res.records


def test_financial_view_hides_non_financial_env_records_only():
    res = _episode()
    kept = thin(res.records, Config("fin", 0.0, 0.0, "financial"), np.random.default_rng(0))
    env_ops = {r.op for r in kept if r.witness == ENV}
    assert env_ops <= {"issue_refund", "cancel_order", "send_email"}
    assert [r for r in kept if r.witness != ENV] == [r for r in res.records if r.witness != ENV]


def test_drop_rates_are_roughly_respected():
    res = _episode()
    recs = res.records * 50
    kept = thin(recs, Config("s", 0.9, 0.3, "full"), np.random.default_rng(0))
    gw = sum(r.witness == GATEWAY for r in kept) / sum(r.witness == GATEWAY for r in recs)
    tr = sum(r.witness == TRACE for r in kept) / sum(r.witness == TRACE for r in recs)
    # ~750 records per witness: SE ~0.017 at p=0.7, so allow ~3.5 SE.
    assert abs(gw - 0.1) < 0.06 and abs(tr - 0.7) < 0.06


def test_per_op_tables_sum_to_the_pooled_table():
    res = _episode()
    for linkage in ("oracle", "signature"):
        total, per_op = tables(res.ledger, res.records, linkage)
        assert total[0] == 0
        np.testing.assert_array_equal(total, np.sum(list(per_op.values()), axis=0))


def test_stratum_with_one_live_witness_reports_the_floor():
    # Only W0 and nothing else saw these actions: no overlap, nothing to estimate.
    t = np.zeros(16, dtype=np.int64)
    t[1] = 5
    est = stratified({"write_file": t})
    assert est.lo == 5 and est.hi == math.inf
    assert est.detail["unidentified"] == 1


def test_perfectly_overlapping_witnesses_are_unidentified_not_confident():
    # Two witnesses that saw exactly the same actions carry no information
    # about what they jointly missed; lc_gated alone would report 20 [20, 22].
    t = np.zeros(16, dtype=np.int64)
    t[0b011] = 20
    est = stratified({"issue_refund": t})
    assert est.lo == 20 and est.hi == math.inf and est.detail["unidentified"] == 1


def test_bonferroni_widens_each_stratum_as_strata_multiply():
    def table(seed):
        rng = np.random.default_rng(seed)
        t = np.zeros(16, dtype=np.int64)
        for h in (1, 2, 3, 4, 5, 6, 7):
            t[h] = int(rng.integers(5, 40))
        return t

    one = stratified({"a": table(0)})
    five = stratified({k: table(i) for i, k in enumerate("abcde")})
    assert one.detail["level_per_stratum"] == 0.95
    assert five.detail["level_per_stratum"] == 0.99
    # Five strata of comparable size must not give a relatively tighter answer.
    assert five.width / five.n_observed > one.width / one.n_observed


def test_estimate_all_runs_on_every_config():
    res = _episode()
    rng = np.random.default_rng(0)
    for cfg in CONFIGS:
        total, per_op = tables(res.ledger, thin(res.records, cfg, rng), "oracle")
        ests = estimate_all(total, per_op)
        assert ests["union_bound"].covers(len(res.ledger))
        assert "auditor_count" not in ests  # W4 dropped (D20)


def test_rho_couples_gateway_and_trace_losses():
    # One shared outage per action. The gateway drops more often (0.9 vs
    # 0.3), so under rho=1 every action the gateway kept was also kept by the
    # trace: the gateway's survivors are a subset of the trace's, and no
    # action is gateway-only. That missing discordance is what breaks an
    # independence model. Needs many distinct actions: the shared draw is
    # cached per ledger id, by design.
    recs = []
    for i in range(400):
        recs.append(Record(TRACE, "issue_refund", f"O{i}", "1.00", oracle=i))
        recs.append(Record(GATEWAY, "issue_refund", f"O{i}", "1.00", oracle=i))

    coupled = thin(recs, Config("c", 0.9, 0.3, "full", rho=1.0), np.random.default_rng(0))
    tr = {r.oracle for r in coupled if r.witness == TRACE}
    gw = {r.oracle for r in coupled if r.witness == GATEWAY}
    assert tr and gw and gw <= tr

    indep = thin(recs, Config("c", 0.9, 0.3, "full", rho=0.0), np.random.default_rng(0))
    tr_i = {r.oracle for r in indep if r.witness == TRACE}
    gw_i = {r.oracle for r in indep if r.witness == GATEWAY}
    assert not gw_i <= tr_i  # independent drops still leave gateway-only actions


def test_heldout_configs_are_not_in_the_selection_pool():
    # The confirmatory hypotheses are tested on configurations no exploratory
    # run has touched; keeping them out of CONFIGS is what enforces that.
    from lacuna.thinning import HELDOUT

    assert {c.name for c in HELDOUT}.isdisjoint({c.name for c in CONFIGS})
    assert all(c.env_view == "financial" for c in HELDOUT)


def test_op_bias_raises_drop_only_for_named_ops():
    from lacuna.thinning import MNAR_BIAS

    cfg = Config("m", 0.5, 0.5, "financial", op_bias=MNAR_BIAS)
    assert cfg.drop(0.5, "update_customer") == pytest.approx(0.9)
    assert cfg.drop(0.5, "issue_refund") == 0.5
    assert cfg.drop(0.8, "write_file") == 1.0  # clamped, never above 1
