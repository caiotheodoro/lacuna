from __future__ import annotations

import numpy as np
import pytest

from lacuna.estimators import baselines
from lacuna.histories import keep_witnesses, observed_total, seen_by
from lacuna.sim import SCENARIOS, ActionType, Scenario


def test_independent_history_probs_are_products():
    s = Scenario("t", "L0", 10, (np.log(0.8 / 0.2), np.log(0.5 / 0.5), np.log(0.25 / 0.75)))
    p = s.history_probs(ActionType(1.0, (0.0, 0.0, 0.0)))
    caps = (0.8, 0.5, 0.25)
    for h in range(8):
        want = np.prod([c if (h >> j) & 1 else 1 - c for j, c in enumerate(caps)])
        assert p[h] == pytest.approx(want)


def test_hidden_type_never_reaches_masked_witnesses():
    s = SCENARIOS["l2_bypasses_proxy"]
    p = s.history_probs(s.types[1])
    for h in range(8):
        if h & 0b011:
            assert p[h] == 0.0


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_sample_is_reproducible_and_sums_to_n(name):
    s = SCENARIOS[name]
    a = s.sample(np.random.default_rng(7))
    b = s.sample(np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert a.sum() == s.n_actions


def test_keep_witnesses_moves_dropped_only_actions_to_cell_zero():
    counts = np.array([0, 5, 7, 11, 13, 17, 19, 23])
    two = keep_witnesses(counts, (0, 1))
    assert two[0] == 0
    # W2-only actions (h=4) vanish; everything else is accounted for.
    assert observed_total(two) == observed_total(counts) - 13
    assert seen_by(two, 0) == seen_by(counts, 0)


def test_floor_baselines_behave_as_documented():
    counts = SCENARIOS["l1_hides_from_trace"].sample(np.random.default_rng(3))
    n_true = int(counts.sum())
    obs = counts.copy()
    obs[0] = 0
    assert baselines.trust_trace(obs).point <= baselines.union_count(obs).point <= n_true
    assert baselines.union_bound(obs).covers(n_true)
    assert not baselines.union_bound(obs).bounded
