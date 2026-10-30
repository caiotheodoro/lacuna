"""Duplicate and phantom records, reordering, and drifting loss (thinning.py additions).

All four are opt-in: defaults must leave every existing result unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter

import numpy as np
import pytest

from lacuna import thinning
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


GOLDEN_OPS = (
    "issue_refund",
    "cancel_order",
    "send_email",
    "update_customer",
    "write_file",
    "delete_file",
)
GOLDEN_SEED = 20261029


def golden_pool(n: int = 150) -> list[Record]:
    """Every witness sees every action, over six ops; the last few lines carry no ledger id."""
    recs = [
        Record(w, GOLDEN_OPS[i % len(GOLDEN_OPS)], f"R{i}", f"{i}.00", oracle=i)
        for i in range(n)
        for w in (TRACE, GATEWAY, ENV)
    ]
    return recs + [
        Record(w, "send_email", f"X{w}", "0", oracle=None) for w in (TRACE, GATEWAY, ENV)
    ]


def golden_configs() -> dict[str, Config]:
    """Every named config the artifacts were produced with, by name."""
    pools = [thinning.CONFIGS, thinning.HELDOUT, thinning.MNAR, thinning.MNAR_HIDDEN]
    pools += [thinning.FALSEPOS, *thinning.POOLS.values()]
    return {c.name: c for pool in pools for c in pool}


def digest(recs: list[Record]) -> str:
    text = "\n".join(repr(dataclasses.astuple(r)) for r in recs)
    return hashlib.sha256(text.encode()).hexdigest()


# sha256 of thin(golden_pool(), cfg, default_rng(GOLDEN_SEED)) per config, computed from
# src/lacuna/thinning.py as of commit 4399a4a (before the drift/corruption additions), so a
# change to the loss draws of any existing config shows up here and not only against itself.
GOLDEN = {
    "anti_fin": "7fa762c5e92917e44adf654fdf71b5394f333845f1348efceb52fbfe114ef1b2",
    "anti_full": "e4bc62839c39ae0ad86677c3d22fd165e49f4235a45b9631d8bb63a11d8c6839",
    "fin_light": "fc55bc347fad2c1aa6919c5ea557f4aa71f89d6c316dfcf9d6c25a59d05c23ad",
    "fin_mid": "b3aea9649649b5426cc1d4a7ccbbd4d18a11dcd745851b00eaed720a488ac416",
    "fin_mid_mnar": "8b3c3386e9b6e6c680019819808cbc1e60cf4d82259bd7e32964dafd2b93c470",
    "fin_mid_mnar_rho100": "0f0673e63f08ba55b0d855ea0f37f09d73ef99e68070378bd0af5b3b5e5e26f9",
    "fin_mid_rho100": "b7fadc6cd4669d844f1c88f27c302d0d080599807506fdbfddae31541e18ae30",
    "fin_sampled": "255a1b149aa1693497beb4421f4d9c7dd44daf96b1181513e0c89d3f111c55bc",
    "fin_sampled_rho100": "736996ea08df7297b62d1824b89ecb86e774debeba53de7387d30d27c9d6a7c4",
    "fin_sampled_rho50": "faccc4ba5137d77d33fafb2c02a7ba6ecc457b5f1ecab730c36edcaf62b9c190",
    "full_mid_hidden": "ec08598111e43630f507f8f410826b523a00a3dc6f967f3179631ca69f155621",
    "full_mid_hidden_rho100": "e49ec241a218cbf19095d8827a91062e8f9b7efa69e6e2c3fb0c896cd9fb22c9",
    "light": "6a49c0092214fb2ee91cca17d6da9b728d8db000508ab49bd444f54d2776f45d",
    "sampled": "9384fb864a540a4ed7c0b244df44aed981843d6f359f7dd7bc508dfc36fc3801",
    "transfer_light": "6a49c0092214fb2ee91cca17d6da9b728d8db000508ab49bd444f54d2776f45d",
    "transfer_mid": "2d14a1ad9b091ae5fccf7995cae21726436f2fddd25619b9c4d4a24ec68499a8",
    "transfer_mid_rho100": "726443cb6a1c7435fc522f2a7819f88a8602f65530aa825fc15f34315a7316df",
}


def test_thin_reproduces_the_pre_round_draws_for_every_existing_config():
    configs = golden_configs()
    assert set(GOLDEN) == set(configs)
    for name, cfg in configs.items():
        got = digest(thin(golden_pool(), cfg, np.random.default_rng(GOLDEN_SEED)))
        assert got == GOLDEN[name], f"thin() output changed for config {name}"


def test_drift_zero_is_identical_to_the_pre_round_thin():
    for name in ("fin_mid", "fin_mid_rho100", "full_mid_hidden", "anti_fin"):
        base = golden_configs()[name]
        cfg = DriftConfig(**{**dataclasses.asdict(base), "drift": 0.0})
        got = digest(thin(golden_pool(), cfg, np.random.default_rng(GOLDEN_SEED)))
        assert got == GOLDEN[name]


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
