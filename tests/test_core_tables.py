"""`lacuna.core.tables`: capture-history tables for any number of named witnesses."""

from __future__ import annotations

import numpy as np
import pytest

from lacuna import linkage
from lacuna.core.records import Obs, WitnessSpec
from lacuna.core.tables import build, keyed_table
from lacuna.witnesses import AUDITOR, ENV, GATEWAY, TRACE, Record

NAMES = ("trace", "gateway", "env")


def _sig(o):
    return (o.op, o.resource, o.value)


def _to_obs(r: Record) -> Obs:
    # ("oracle", None) is one shared id: callid_table merges id-less lines the same way.
    return Obs(NAMES[r.witness], r.op, r.resource, r.value, link_id=("oracle", r.oracle))


def _random_records(rng: np.random.Generator) -> list[Record]:
    """Few resources, so signatures collide; some lines without an id; and some records
    that disagree with the rest of their call about the resource (one id, two keys)."""
    recs = []
    for lid in range(int(rng.integers(5, 40))):
        op = ("cancel_order", "issue_refund")[int(rng.integers(2))]
        res = f"O{int(rng.integers(6))}"
        val = "cancelled" if op == "cancel_order" else f"{int(rng.integers(3))}.00"
        for w in (TRACE, GATEWAY, ENV, AUDITOR):
            for _ in range(int(rng.integers(0, 3))):  # 0, 1 or 2 records per witness
                oracle = None if rng.random() < 0.15 else lid
                r = f"O{int(rng.integers(6))}" if rng.random() < 0.1 else res
                recs.append(Record(w, op, r, val, oracle=oracle))
    order = rng.permutation(len(recs))
    return [recs[i] for i in order]


def _legacy_sig(r: Record):
    return linkage.signature(r.op, r.resource, r.value)


def test_keyed_table_equals_callid_table_on_random_records():
    rng = np.random.default_rng(20261115)
    for _ in range(500):
        recs = _random_records(rng)
        old = linkage.callid_table(recs, _legacy_sig)
        obs = [_to_obs(r) for r in recs if r.witness in (TRACE, GATEWAY, ENV)]
        new = keyed_table(obs, NAMES, ("trace", "gateway"), _sig)
        assert new.tolist() == old.tolist()


def test_a_missing_link_id_is_what_a_unique_id_would_be():
    # The one intended difference from callid_table, which merges id-less lines into one
    # call: here each is its own action, as if the harness had given it an id nobody shares.
    rng = np.random.default_rng(7)
    for _ in range(300):
        recs = [r for r in _random_records(rng) if r.witness in (TRACE, GATEWAY, ENV)]
        fresh = iter(range(-1, -10_000, -1))
        unique = [
            Record(r.witness, r.op, r.resource, r.value, oracle=next(fresh))
            if r.oracle is None and r.witness != ENV
            else r
            for r in recs
        ]
        old = linkage.callid_table(unique, _legacy_sig)
        obs = [Obs(NAMES[r.witness], r.op, r.resource, r.value, link_id=r.oracle) for r in recs]
        assert keyed_table(obs, NAMES, ("trace", "gateway"), _sig).tolist() == old.tolist()


def test_ties_are_broken_by_input_order_so_pass_records_in_time_order():
    # Two actions each seen by one id witness, one attached record: it goes to the first.
    a = Obs("trace", "op", "r", "v", link_id=1)
    b = Obs("gateway", "op", "r", "v", link_id=2)
    e = Obs("env", "op", "r", "v")
    assert keyed_table([a, b, e], NAMES, ("trace", "gateway"), _sig).tolist() == [
        0, 0, 1, 0, 0, 1, 0, 0,
    ]  # fmt: skip
    assert keyed_table([b, a, e], NAMES, ("trace", "gateway"), _sig).tolist() == [
        0, 1, 0, 0, 0, 0, 1, 0,
    ]  # fmt: skip


def test_duplicate_witness_names_are_refused():
    with pytest.raises(ValueError, match="more than once"):
        keyed_table([], ("a", "a", "c"), ("a",), _sig)
    specs = (WitnessSpec("a", frozenset({"p"})), WitnessSpec("a", frozenset({"p"})))
    with pytest.raises(ValueError, match="more than once"):
        build([], specs, _sig)


def test_bypassed_action_is_an_env_only_cell():
    obs = [
        Obs("trace", "cancel_order", "O1", "cancelled", link_id="c1"),
        Obs("gateway", "cancel_order", "O1", "cancelled", link_id="c1"),
        Obs("env", "cancel_order", "O1", "cancelled"),
        Obs("env", "cancel_order", "O2", "cancelled"),
    ]
    t = keyed_table(obs, NAMES, ("trace", "gateway"), _sig)
    assert t.tolist() == [0, 0, 0, 0, 1, 0, 0, 1]


def test_records_without_a_link_id_are_separate_actions():
    obs = [
        Obs("trace", "cancel_order", "O1", "cancelled"),
        Obs("trace", "cancel_order", "O1", "cancelled"),
    ]
    t = keyed_table(obs, ("trace", "gateway"), ("trace", "gateway"), _sig)
    assert t.tolist() == [0, 2, 0, 0]


def test_two_attached_witnesses_pair_in_order():
    # No id witness saw anything: two attached logs with 2 and 1 records of one signature
    # are at least two actions, one seen by both and one by the first only.
    obs = [
        Obs("env", "cancel_order", "O1", "cancelled"),
        Obs("env", "cancel_order", "O1", "cancelled"),
        Obs("audit", "cancel_order", "O1", "cancelled"),
    ]
    t = keyed_table(obs, ("spans", "env", "audit"), ("spans",), _sig)
    assert t.tolist() == [0, 0, 1, 0, 0, 0, 1, 0]


def test_four_witnesses_give_sixteen_cells():
    obs = [
        Obs("a", "op", "r", "v", link_id=1),
        Obs("b", "op", "r", "v", link_id=1),
        Obs("c", "op", "r", "v", link_id=1),
        Obs("d", "op", "r", "v"),
    ]
    t = keyed_table(obs, ("a", "b", "c", "d"), ("a", "b", "c"), _sig)
    assert t.tolist() == [0] * 15 + [1]


def test_records_of_unlisted_witnesses_are_ignored():
    obs = [Obs("trace", "op", "r", "v", link_id=1), Obs("other", "op", "r", "v", link_id=1)]
    t = keyed_table(obs, ("trace", "gateway"), ("trace", "gateway"), _sig)
    assert t.tolist() == [0, 1, 0, 0]


def test_id_witness_must_be_a_named_witness():
    with pytest.raises(ValueError, match="not among the witnesses"):
        keyed_table([], ("trace", "gateway"), ("spans",), _sig)


SPECS = (
    WitnessSpec("spans_a", frozenset({"a"})),
    WitnessSpec("gateway", frozenset({"a"})),
    WitnessSpec("spans_b", frozenset({"b"})),
    WitnessSpec("mcp_log", frozenset({"b"})),
    WitnessSpec("env", frozenset({"a", "b"}), by_id=False),
)


def test_build_gives_one_table_per_plane_with_only_the_witnesses_that_see_it():
    obs = [
        Obs("spans_a", "op", "r1", "v", stratum="a", link_id="t1"),
        Obs("gateway", "op", "r1", "v", stratum="a", link_id="t1"),
        Obs("env", "op", "r1", "v", stratum="a"),
        Obs("mcp_log", "op", "r2", "v", stratum="b", link_id="t2"),
        Obs("env", "op", "r3", "v", stratum="b"),
    ]
    out = build(obs, SPECS, _sig)
    assert list(out) == ["a", "b"]
    names_a, table_a = out["a"]
    names_b, table_b = out["b"]
    assert names_a == ("spans_a", "gateway", "env")
    assert names_b == ("spans_b", "mcp_log", "env")
    assert table_a.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert table_b.tolist() == [0, 0, 1, 0, 1, 0, 0, 0]


def test_build_keeps_a_plane_nobody_recorded_as_an_empty_table():
    out = build([], SPECS, _sig)
    assert out["a"][1].tolist() == [0] * 8 and out["b"][1].tolist() == [0] * 8


@pytest.mark.parametrize(
    "bad, message",
    [
        (Obs("nobody", "op", "r", "v", stratum="a"), "no spec"),
        (Obs("env", "op", "r", "v"), "no stratum"),
        (Obs("gateway", "op", "r", "v", stratum="b"), "cannot see"),
    ],
)
def test_build_refuses_records_that_contradict_the_specs(bad, message):
    with pytest.raises(ValueError, match=message):
        build([bad], SPECS, _sig)


def test_build_refuses_a_plane_with_one_witness():
    specs = (WitnessSpec("only", frozenset({"a"})),)
    with pytest.raises(ValueError, match="fewer than two"):
        build([], specs, _sig)
