"""Duplicate and phantom records, reordering, and drifting loss (thinning.py additions).

All four are opt-in: defaults must leave every existing result unchanged.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pytest

from lacuna.envs.kernel import LedgerEntry
from lacuna.linkage import oracle_table, signature_table
from lacuna.thinning import Config, Corruption, DriftConfig, corrupt, thin
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record


def _pool(n: int = 60) -> list[Record]:
    """n distinct actions seen by all three witnesses, keyed by a ledger-like id."""
    return [
        Record(w, "issue_refund", f"O{i}", "1.00", oracle=i)
        for i in range(n)
        for w in (TRACE, GATEWAY, ENV)
    ]


def _ledger(n: int) -> list[LedgerEntry]:
    extra = {f.name for f in dataclasses.fields(LedgerEntry)} - {"id", "op", "resource", "value"}
    return [
        LedgerEntry(
            id=i, op="issue_refund", resource=f"O{i}", value="1.00", **dict.fromkeys(extra, 0)
        )
        for i in range(n)
    ]


def _count(recs: list[Record], w: int) -> int:
    return sum(r.witness == w for r in recs)


def test_default_corruption_is_a_noop_and_draws_nothing():
    recs = _pool()
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert corrupt(recs, Corruption(), rng) == recs
    assert rng.bit_generator.state == state


def test_config_fields_are_unchanged_so_recorded_artifacts_keep_their_keys():
    assert list(Config("x", 0.5, 0.5, "full").__dict__) == [
        "name", "p_gateway_drop", "p_trace_drop", "env_view", "rho", "anti",
        "op_bias", "hidden_mult", "hidden_env_blind",
    ]  # fmt: skip


def test_dup_rate_one_writes_every_line_of_that_witness_twice_and_adjacent():
    out = corrupt(_pool(10), Corruption(dup=(1.0, 0.0, 0.0)), np.random.default_rng(0))
    assert (_count(out, TRACE), _count(out, GATEWAY), _count(out, ENV)) == (20, 10, 10)
    for i, r in enumerate(out):
        if r.witness == TRACE and (i == 0 or out[i - 1] != r):
            assert out[i + 1] == r  # the retry follows the original


def test_dups_inflate_signature_counts_but_not_oracle_identity():
    recs = _pool(10)
    out = corrupt(recs, Corruption(dup=(1.0, 0.0, 0.0)), np.random.default_rng(0))
    # 10 refunds on distinct orders; two trace lines each makes max count 2.
    assert signature_table(recs, 3).sum() == 10
    # The trace count is 2 in every group, so each gains a trace-only action.
    assert signature_table(out, 3).sum() == 20
    assert signature_table(out, 3)[0b001] == 10
    led = _ledger(10)
    np.testing.assert_array_equal(oracle_table(led, recs, 3)[0], oracle_table(led, out, 3)[0])


def test_dups_on_a_repeated_signature_create_extra_actions():
    # Two real refunds to the same order and amount are one signature with count 2.
    recs = [Record(w, "issue_refund", "O1", "1.00", oracle=o) for o in (1, 2) for w in (0, 1, 2)]
    out = corrupt(recs, Corruption(dup=(1.0, 0.0, 0.0)), np.random.default_rng(0))
    assert signature_table(recs, 3).sum() == 2
    assert signature_table(out, 3).sum() == 4


def test_phantoms_are_new_records_with_unseen_ids_and_dont_touch_real_ones():
    recs = _pool(40)
    out = corrupt(recs, Corruption(phantom=(0.0, 0.5, 0.0)), np.random.default_rng(1))
    real = [r for r in out if r.oracle >= 0]
    assert sorted(real, key=lambda r: (r.oracle, r.witness)) == sorted(
        recs, key=lambda r: (r.oracle, r.witness)
    )
    ghosts = [r for r in out if r.oracle < 0]
    assert 5 < len(ghosts) < 40 and all(r.witness == GATEWAY for r in ghosts)
    assert len({r.oracle for r in ghosts}) == len(ghosts)  # each phantom is its own call
    led = _ledger(40)
    np.testing.assert_array_equal(oracle_table(led, recs, 3)[0], oracle_table(led, out, 3)[0])


def test_forged_phantoms_have_signatures_nothing_else_shares():
    recs = _pool(40)
    out = corrupt(
        recs, Corruption(phantom=(0.0, 0.5, 0.0), phantom_kind="forged"), np.random.default_rng(1)
    )
    ghosts = [r for r in out if r.oracle < 0]
    real = {(r.op, r.resource) for r in recs}
    assert ghosts and all((g.op, g.resource) not in real for g in ghosts)
    assert signature_table(out, 3).sum() == 40 + len(ghosts)


def test_cloned_phantoms_share_a_real_signature():
    recs = _pool(40)
    out = corrupt(recs, Corruption(phantom=(0.0, 0.5, 0.0)), np.random.default_rng(1))
    ghosts = [r for r in out if r.oracle < 0]
    real = {(r.op, r.resource) for r in recs}
    assert ghosts and all((g.op, g.resource) in real for g in ghosts)


def test_env_view_restricts_which_ops_a_w3_phantom_can_copy():
    recs = []
    for i in range(30):
        recs.append(Record(ENV, "update_customer", f"C{i}.email", "x", oracle=i))
        recs.append(Record(TRACE, "update_customer", f"C{i}.email", "x", oracle=i))
    out = corrupt(
        recs, Corruption(phantom=(0.0, 0.0, 1.0)), np.random.default_rng(0), env_view="financial"
    )
    assert out == recs  # W3 cannot see update_customer, so it has nothing to forge from


def test_reorder_keeps_each_witness_multiset_and_changes_order():
    recs = _pool(60)
    out = corrupt(recs, Corruption(swap_p=0.5), np.random.default_rng(3))
    for w in (TRACE, GATEWAY, ENV):
        assert Counter(r for r in recs if r.witness == w) == Counter(
            r for r in out if r.witness == w
        )
    assert out != recs
    jit = corrupt(recs, Corruption(jitter=5.0), np.random.default_rng(3))
    assert Counter(jit) == Counter(recs) and jit != recs


def test_reorder_moves_records_only_within_their_own_witness_slots():
    recs = _pool(60)
    out = corrupt(recs, Corruption(swap_p=1.0, jitter=4.0), np.random.default_rng(5))
    assert [r.witness for r in out] == [r.witness for r in recs]


def test_signature_and_oracle_tables_are_blind_to_record_order():
    recs = _pool(60)
    out = corrupt(recs, Corruption(swap_p=0.7, jitter=8.0), np.random.default_rng(2))
    np.testing.assert_array_equal(signature_table(recs, 3), signature_table(out, 3))
    led = _ledger(60)
    np.testing.assert_array_equal(oracle_table(led, recs, 3)[0], oracle_table(led, out, 3)[0])


def test_drift_zero_is_identical_to_the_old_thin():
    recs = _pool(200)
    a = thin(recs, Config("x", 0.5, 0.5, "full"), np.random.default_rng(9))
    b = thin(recs, DriftConfig("x", 0.5, 0.5, "full", drift=0.0), np.random.default_rng(9))
    assert a == b


def test_drift_raises_the_drop_rate_along_the_episode_and_keeps_the_mean():
    recs = []
    for i in range(4000):
        recs.append(Record(TRACE, "issue_refund", f"O{i}", "1.00", oracle=i))
        recs.append(Record(GATEWAY, "issue_refund", f"O{i}", "1.00", oracle=i))
    kept = thin(recs, DriftConfig("d", 0.5, 0.5, "full", drift=0.8), np.random.default_rng(0))
    ids = {r.oracle for r in kept if r.witness == TRACE}
    first = sum(i in ids for i in range(2000)) / 2000
    last = sum(i in ids for i in range(2000, 4000)) / 2000
    # Drop rate runs 0.1 to 0.9 along the episode: the halves average 0.3 and 0.7.
    assert first > last + 0.3
    assert abs((first + last) / 2 - 0.5) < 0.03


def test_drift_scale_is_linear_with_mean_one_and_the_drop_is_clamped():
    cfg = DriftConfig("d", 0.9, 0.9, "full", drift=1.0)
    assert cfg.drift_scale(0.0) == pytest.approx(0.0)
    assert cfg.drift_scale(0.5) == pytest.approx(1.0)
    assert cfg.drift_scale(1.0) == pytest.approx(2.0)
    assert cfg.drop(0.9 * cfg.drift_scale(0.9), "issue_refund") == 1.0
