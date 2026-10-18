from __future__ import annotations

import math

import numpy as np
import pytest

from lacuna.estimators import loglinear, stratified
from lacuna.estimators.stratified import convolve_densities, quantile, strat_bic
from lacuna.histories import keep_witnesses
from lacuna.thinning import MNAR_HIDDEN, Config, thin
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record


def _expected_table(n_true: int, p: tuple[float, ...]) -> np.ndarray:
    """Expected counts under independent capture, rounded; cell 0 zeroed."""
    k = len(p)
    t = np.zeros(2**k, dtype=np.int64)
    for h in range(1, 2**k):
        prob = math.prod(p[w] if h >> w & 1 else 1 - p[w] for w in range(k))
        t[h] = round(n_true * prob)
    return t


def _k3(t2: np.ndarray) -> np.ndarray:
    """Embed a two-witness table as W1, W2 of a three-witness one, W3 dead."""
    t = np.zeros(8, dtype=np.int64)
    t[: len(t2)] = t2
    return t


def test_convolution_of_two_known_densities():
    # A fair coin on {1, 2} plus a binomial(2, 1/2) on {2, 3, 4}.
    offset, pmf = convolve_densities([(1, np.array([0.5, 0.5])), (2, np.array([0.25, 0.5, 0.25]))])
    assert offset == 3
    np.testing.assert_allclose(pmf, [0.125, 0.375, 0.375, 0.125])
    assert quantile(offset, pmf, 0.1) == 3
    assert quantile(offset, pmf, 0.5) == 4
    assert quantile(offset, pmf, 0.9) == 6


def test_fft_path_matches_direct_convolution(monkeypatch):
    rng = np.random.default_rng(0)
    a, b = rng.random(400), rng.random(300)
    a, b = a / a.sum(), b / b.sum()
    direct = convolve_densities([(5, a), (7, b)])[1]
    monkeypatch.setattr(stratified, "DIRECT_CONV_MAX", 0)
    off, fft = convolve_densities([(5, a), (7, b)])
    assert off == 12
    np.testing.assert_allclose(fft, direct, atol=1e-12)


def test_missing_mass_puts_the_quantile_at_infinity():
    pmf = np.array([0.3, 0.6])  # 0.1 of the mass is past the cap
    assert quantile(10, pmf, 0.5) == 11
    assert quantile(10, pmf, 0.95) == math.inf


@pytest.mark.parametrize(
    "table",
    [
        _expected_table(1000, (0.5, 0.6, 0.4)),
        _k3(_expected_table(400, (0.5, 0.5))),  # W3 dead: a two-witness stratum
    ],
)
def test_one_identified_stratum_matches_ll_bic(table):
    # Flat-prior central 95% against the profile-likelihood interval: the same
    # likelihood read two ways, so they agree to a few actions when it is
    # well identified.
    live = stratified.live_witnesses(table, 3)
    ref = loglinear.best_bic(keep_witnesses(table, live))
    est = strat_bic({"a": table})
    assert est.detail["identified"] == 1
    tol = 0.05 * ref.width + 2
    assert est.lo == pytest.approx(ref.lo, abs=tol)
    assert est.hi == pytest.approx(ref.hi, abs=tol)
    assert est.point == pytest.approx(ref.point, abs=tol)


def test_unidentified_stratum_opens_the_top_and_adds_its_count_to_the_bottom():
    good = _expected_table(600, (0.5, 0.6, 0.4))
    one_live = np.zeros(8, dtype=np.int64)
    one_live[0b001] = 7  # only the trace saw these
    overlap = np.zeros(8, dtype=np.int64)
    overlap[0b011] = 20  # trace and gateway saw exactly the same actions

    alone = strat_bic({"a": good})
    est = strat_bic({"a": good, "b": one_live, "c": overlap})
    assert est.hi == math.inf and est.point == math.inf
    assert est.lo == alone.lo + 7 + 20
    assert est.n_observed == alone.n_observed + 27
    assert est.detail["unidentified"] == 2


def test_unbounded_stratum_is_a_floor_not_a_number():
    # Two witnesses that never saw the same action: the independence model
    # puts the missed count past any cap.
    disjoint = _k3(np.array([0, 20, 20, 0]))
    good = _expected_table(600, (0.5, 0.6, 0.4))
    est = strat_bic({"a": good, "b": disjoint})
    assert est.detail["unbounded"] == 1
    assert est.hi == math.inf
    assert est.lo == strat_bic({"a": good}).lo + 40


def test_empty_strata_are_skipped():
    good = _expected_table(600, (0.5, 0.6, 0.4))
    with_empty = strat_bic({"a": good, "z": np.zeros(8, dtype=np.int64)})
    alone = strat_bic({"a": good})
    assert (with_empty.lo, with_empty.point, with_empty.hi) == (alone.lo, alone.point, alone.hi)
    assert with_empty.detail["empty"] == 1
    nothing = strat_bic({"z": np.zeros(8, dtype=np.int64)})
    assert nothing.lo == 0 and nothing.hi == math.inf


def test_two_strata_total_is_wider_than_either_and_centred_on_the_sum():
    a = _expected_table(600, (0.5, 0.6, 0.4))
    b = _k3(_expected_table(300, (0.5, 0.5)))
    ea, eb, both = strat_bic({"a": a}), strat_bic({"b": b}), strat_bic({"a": a, "b": b})
    assert both.point == pytest.approx(ea.point + eb.point, abs=3)
    assert max(ea.width, eb.width) < both.width < ea.width + eb.width


def _three_witness_records(n_actions: int) -> list[Record]:
    recs = []
    for i in range(n_actions):
        for w in (TRACE, GATEWAY, ENV):
            recs.append(Record(w, "issue_refund", f"O{i}", "1.00", oracle=i))
    return recs


def test_hidden_rule_drops_the_hidden_class_from_every_witness():
    cfg = MNAR_HIDDEN[0]
    assert cfg.env_view == "full" and cfg.rho == 0.0
    recs = _three_witness_records(3000)
    kept = thin(recs, cfg, np.random.default_rng(0))

    def rate(witness: int, hidden: bool) -> float:
        pool = [r for r in recs if r.witness == witness and (r.oracle % 3 == 0) == hidden]
        got = [r for r in kept if r.witness == witness and (r.oracle % 3 == 0) == hidden]
        return len(got) / len(pool)

    assert rate(ENV, True) == 0.0 and rate(ENV, False) == 1.0
    for w in (TRACE, GATEWAY):
        # ~1000 hidden records per pipe: keep rate 0.1 (drop 0.5 * 1.8), SE ~0.01.
        assert abs(rate(w, True) - 0.1) < 0.035
        assert abs(rate(w, False) - 0.5) < 0.04


def test_hidden_rho100_couples_the_pipes_for_the_hidden_class_too():
    kept = thin(_three_witness_records(900), MNAR_HIDDEN[1], np.random.default_rng(0))
    tr = {r.oracle for r in kept if r.witness == TRACE}
    gw = {r.oracle for r in kept if r.witness == GATEWAY}
    assert tr == gw  # equal drop rates and one shared outage: no discordance


def test_configs_without_the_hidden_rule_hide_nothing():
    r = Record(ENV, "issue_refund", "O3", "1.00", oracle=3)
    assert not Config("plain", 0.5, 0.5, "full").is_hidden(r)
    assert Config("h", 0.5, 0.5, "full", hidden_env_blind=True).is_hidden(r)
    assert not Config("h", 0.5, 0.5, "full", hidden_env_blind=True).is_hidden(
        Record(ENV, "issue_refund", "O3", "1.00", oracle=None)
    )
    cfg = Config("h", 0.5, 0.5, "financial", hidden_mult=1.8)
    assert cfg.drop(0.5, "issue_refund", hidden=True) == pytest.approx(0.9)
    assert cfg.drop(0.5, "issue_refund") == 0.5


def test_hidden_rule_on_a_recorded_episode_keys_every_witness_on_the_same_action():
    # Env records carry the kernel's ledger id and pipe records the router's;
    # the hidden class is only meaningful if those are the same id space.
    from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
    from lacuna.episode import run_episode

    res = run_episode(
        11,
        ScriptedAgent(np.random.default_rng(11)),
        ScriptedAuditor(np.random.default_rng(1), p_miss=0.0),
        p_gateway_drop=0.0,
        p_trace_drop=0.0,
    )
    ids = {w: {r.oracle for r in res.records if r.witness == w} for w in (TRACE, GATEWAY, ENV)}
    assert ids[ENV] <= ids[TRACE] == ids[GATEWAY]
    kept = thin(res.records, MNAR_HIDDEN[0], np.random.default_rng(0))
    env_kept = {r.oracle for r in kept if r.witness == ENV}
    assert env_kept == {i for i in ids[ENV] if i % 3 != 0}
