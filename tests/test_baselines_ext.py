"""The cheaper checks the paper said it did not compare: Chapman, sequence gaps, canaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lacuna.envs.kernel import LedgerEntry
from lacuna.estimators import baselines
from lacuna.estimators.baselines import SeqObs
from lacuna.thinning import Config, thin
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record

# --- Chapman -----------------------------------------------------------------


def _table(n_both=20, only1=30, only2=20, only3=0, w13=0, w23=0):
    t = np.zeros(8, dtype=np.int64)
    t[0b011] = n_both
    t[0b001] = only1
    t[0b010] = only2
    t[0b100] = only3
    t[0b101] = w13
    t[0b110] = w23
    return t


def test_chapman_point_and_variance_match_the_textbook_formulas():
    e = baselines.chapman(_table(), 0, 1)
    n1, n2, m = 50, 40, 20
    assert e.name == "chapman_w1w2"
    assert e.point == pytest.approx((n1 + 1) * (n2 + 1) / (m + 1) - 1)
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    assert e.hi == pytest.approx(e.point + 1.96 * math.sqrt(var))
    assert e.lo == pytest.approx(e.point - 1.96 * math.sqrt(var))
    assert e.n_observed == 70


def test_chapman_lower_end_never_below_what_the_pair_saw():
    # Almost full overlap: the Wald lower end would fall under the union.
    e = baselines.chapman(_table(n_both=100, only1=1, only2=1), 0, 1)
    assert e.lo >= 102


def test_chapman_uses_only_its_two_witnesses():
    base = baselines.chapman(_table(), 0, 1)
    with_w3 = baselines.chapman(_table(only3=500), 0, 1)
    assert (with_w3.point, with_w3.lo, with_w3.hi) == (base.point, base.lo, base.hi)
    assert with_w3.n_observed == base.n_observed  # the pair's union, not the three-way union


def test_chapman_with_no_overlap_is_finite():
    e = baselines.chapman(_table(n_both=0, only1=10, only2=10), 0, 1)
    assert math.isfinite(e.point) and e.bounded
    assert e.point == pytest.approx(11 * 11 - 1)


def test_chapman_pairs_are_named_and_pick_the_right_columns():
    t = _table(n_both=5, only1=0, only2=0, only3=0, w13=40, w23=3)
    assert baselines.chapman_w1w3(t).name == "chapman_w1w3"
    assert baselines.chapman_w2w3(t).name == "chapman_w2w3"
    # W1 and W3 overlap heavily, W2 and W3 hardly: very different estimates.
    assert baselines.chapman_w1w3(t).point < baselines.chapman_w2w3(t).point


def test_chapman_is_nearly_unbiased_for_independent_witnesses():
    rng = np.random.default_rng(0)
    N, ests = 400, []
    for _ in range(400):
        a, b = rng.random(N) < 0.4, rng.random(N) < 0.5
        t = np.zeros(8, dtype=np.int64)
        t[1], t[2], t[3] = (a & ~b).sum(), (~a & b).sum(), (a & b).sum()
        ests.append(baselines.chapman_w1w2(t))
    assert abs(np.mean([e.point for e in ests]) - N) < 3
    assert np.mean([e.covers(N) for e in ests]) > 0.9


# --- sequence gaps -----------------------------------------------------------


def _ledger(n, ops=("issue_refund", "update_customer", "send_email")):
    return [LedgerEntry(i, ops[i % len(ops)], f"R{i}", f"v{i}", i) for i in range(n)]


def _records(ledger):
    out = []
    for w in (TRACE, GATEWAY, ENV):
        out += [Record(w, e.op, e.resource, e.value, e.id) for e in ledger]
    return out


def test_gateway_seq_follows_log_order_and_skips_bypassed_actions():
    recs = _records(_ledger(6))
    assert baselines.gateway_seq(recs) == {i: i for i in range(6)}
    seq = baselines.gateway_seq(recs, bypass=frozenset({1, 4}))
    assert seq == {0: 0, 2: 1, 3: 2, 5: 3}


def test_seq_obs_counts_the_last_surviving_number_not_the_survivors():
    recs = _records(_ledger(10))
    seq = baselines.gateway_seq(recs)
    kept = [r for r in recs if not (r.witness == GATEWAY and r.oracle in (2, 3, 9))]
    obs = baselines.seq_obs(kept, seq)
    assert obs == SeqObs(last=9, seen=7)  # numbers 0..8 with 2 and 3 missing; last survivor is 8
    assert baselines.seq_obs([], seq) == SeqObs(0, 0)


def test_seq_gap_recovers_n_exactly_when_the_last_record_survives():
    obs = [SeqObs(last=10, seen=7), SeqObs(last=5, seen=5)]
    e = baselines.seq_gap(obs, n_observed=12)
    assert e.name == "seq_gap" and e.point == e.lo == e.hi == 15
    assert e.covers(15) and not e.covers(16)


def test_seq_gap_lower_bound_variant_is_never_wrong_and_never_tight():
    e = baselines.seq_gap_lb([SeqObs(10, 7), SeqObs(5, 5)], n_observed=12)
    assert e.lo == 15 and e.hi == math.inf and e.covers(15) and e.covers(40)


def test_seq_gap_tail_adds_the_expected_trailing_loss():
    # 10 of the 80 positions before the last survivors are gaps: tail ~ p/(1-p) per episode.
    obs = [SeqObs(last=41, seen=31), SeqObs(last=41, seen=31)]
    e = baselines.seq_gap_tail(obs, n_observed=62)
    p = 20 / 80
    assert e.point == pytest.approx(82 + 2 * p / (1 - p))
    assert e.lo <= e.point <= e.hi and e.lo >= 82
    no_gaps = baselines.seq_gap_tail([SeqObs(5, 5)], n_observed=5)
    assert no_gaps.point == 5 and no_gaps.lo == no_gaps.hi == 5


def test_sequence_check_is_exact_without_loss_and_blind_to_bypass_with_it():
    led = _ledger(30)
    recs = _records(led)
    none = Config("none", 0.0, 0.0, "full")
    kept = thin(recs, none, np.random.default_rng(0))
    obs = baselines.seq_obs(kept, baselines.gateway_seq(recs))
    assert baselines.seq_gap([obs], 30).point == 30
    bypass = frozenset(range(0, 30, 5))  # 6 of 30 never touch the gateway
    routed = [r for r in recs if not (r.witness == GATEWAY and r.oracle in bypass)]
    seq = baselines.gateway_seq(recs, bypass)
    assert baselines.seq_gap([baselines.seq_obs(routed, seq)], 30).point == 24


# --- canaries ----------------------------------------------------------------


def test_plant_canaries_adds_k_actions_seen_by_the_three_witnesses():
    led = _ledger(20)
    recs = _records(led)
    led2, recs2, ids = baselines.plant_canaries(led, recs, 5, np.random.default_rng(1))
    assert len(led2) == 25 and len(ids) == 5
    assert sorted(e.id for e in led2) == list(range(25))
    for w in (TRACE, GATEWAY, ENV):
        assert sum(r.witness == w and r.oracle in ids for r in recs2) == 5
    # Real actions keep their records, only renumbered (order preserved).
    real = [e for e in led2 if e.id not in ids]
    assert [e.resource for e in real] == [e.resource for e in led]
    real_recs = [r for r in recs2 if r.oracle not in ids]
    assert len(real_recs) == len(recs)
    by_id = {e.id: e for e in led2}
    assert all(by_id[r.oracle].resource == r.resource for r in real_recs)


def test_canaries_default_to_the_episodes_op_mix_and_can_be_forced_elsewhere():
    led = _ledger(30, ops=("issue_refund",))
    recs = _records(led)
    _, recs2, ids = baselines.plant_canaries(led, recs, 6, np.random.default_rng(1))
    assert {r.op for r in recs2 if r.oracle in ids} == {"issue_refund"}
    _, recs3, ids3 = baselines.plant_canaries(
        led, recs, 6, np.random.default_rng(1), ops=("send_email", "write_file")
    )
    assert {r.op for r in recs3 if r.oracle in ids3} == {"send_email", "write_file"}


def test_canaries_sit_outside_the_hidden_key_when_ids_are_not_interleaved():
    led = _ledger(12)
    _, _, ids = baselines.plant_canaries(
        led, _records(led), 6, np.random.default_rng(1), interleave=False
    )
    assert all(i % 3 != 0 for i in ids)
    assert not ids & {e.id for e in led}


def test_canaries_share_the_loss_process_with_real_actions():
    led = _ledger(40)
    _, recs2, ids = baselines.plant_canaries(led, _records(led), 40, np.random.default_rng(2))
    kept = thin(recs2, Config("gw", 0.5, 0.0, "full"), np.random.default_rng(3))
    seen = {r.oracle for r in kept if r.witness == GATEWAY}
    rate = len(seen & ids) / len(ids)
    assert 0.2 < rate < 0.8  # 40 canaries at p=0.5


def test_canary_histories_read_which_witnesses_saw_each_canary():
    led = _ledger(5)
    _, recs2, ids = baselines.plant_canaries(led, _records(led), 3, np.random.default_rng(1))
    kept = [r for r in recs2 if not (r.witness == TRACE and r.oracle in ids)]
    assert baselines.canary_histories(kept, ids) == [0b110] * 3
    assert baselines.strip_canaries(kept, ids) == [r for r in kept if r.oracle not in ids]


def test_canary_estimate_scales_the_union_by_the_canary_miss_rate():
    e = baselines.canary_estimate(100, [1] * 10 + [0] * 10)  # half missed by all
    assert e.name == "canary_joint" and e.point == pytest.approx(200)
    assert e.lo >= 100 and e.hi > 200
    none_missed = baselines.canary_estimate(100, [1] * 20)
    assert none_missed.point == 100
    assert none_missed.lo <= 100 <= none_missed.hi
    assert baselines.canary_estimate(100, [0] * 20).point == math.inf


@pytest.mark.parametrize("joint", [True, False])
def test_canary_estimate_with_no_canaries_is_unbounded_not_a_crash(joint):
    e = baselines.canary_estimate(100, [], joint=joint)
    assert e.point == math.inf and e.hi == math.inf and e.lo == 100 and not e.bounded


def test_planting_canaries_in_an_empty_ledger_is_a_clear_error():
    with pytest.raises(ValueError, match="empty ledger"):
        baselines.plant_canaries([], [], 3, np.random.default_rng(0))
    # k = 0 is fine, and so is an explicit op list.
    assert baselines.plant_canaries([], [], 0, np.random.default_rng(0))[2] == frozenset()
    _, recs, ids = baselines.plant_canaries([], [], 2, np.random.default_rng(0), ops=["send_email"])
    assert len(ids) == 2 and len(recs) == 6


def test_canary_independent_variant_multiplies_marginal_miss_rates():
    # W1 and W2 always seen together or not at all (coupled); W3 never sees them.
    hist = [0b011] * 10 + [0] * 10
    joint = baselines.canary_estimate(100, hist, joint=True)
    indep = baselines.canary_estimate(100, hist, joint=False)
    assert joint.name == "canary_joint" and indep.name == "canary_indep"
    assert joint.point == pytest.approx(200)
    # marginals: W1, W2 seen in half of canaries, W3 in none: q = 0.5 * 0.5 * 1
    assert indep.point == pytest.approx(100 / 0.75)


def test_canary_estimate_is_nearly_unbiased_under_exchangeable_independent_loss():
    rng = np.random.default_rng(0)
    N, K, q = 400, 60, 0.25
    pts, cov = [], []
    for _ in range(300):
        seen = rng.random(N) >= q
        canaries = [int(rng.random() >= q) for _ in range(K)]
        e = baselines.canary_estimate(int(seen.sum()), canaries)
        if math.isfinite(e.point):
            pts.append(e.point)
        cov.append(e.covers(N))
    assert abs(np.mean(pts) - N) < 0.05 * N
    assert np.mean(cov) > 0.85
