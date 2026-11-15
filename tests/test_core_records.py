"""`lacuna.core.records`: records that do not know which environment made them."""

from __future__ import annotations

import dataclasses

import pytest

from lacuna.core.records import Obs, WitnessSpec


def test_obs_defaults_and_is_hashable():
    o = Obs("spans", "issue_refund", "O1", "5.00")
    assert (o.stratum, o.link_id, o.ts, o.seq, o.oracle) == (None, None, None, None, None)
    assert o.session is None
    assert {o: 1}[Obs("spans", "issue_refund", "O1", "5.00")] == 1


def test_the_session_is_the_last_field_so_earlier_positional_callers_still_mean_the_same():
    names = [f.name for f in dataclasses.fields(Obs)]
    assert names[:9] == [
        "witness", "op", "resource", "value", "stratum", "link_id", "ts", "seq", "oracle",
    ]  # fmt: skip
    assert names[9:] == ["session"]
    assert Obs("env", "x", "r", "v", session="s1") != Obs("env", "x", "r", "v", session="s2")


def test_obs_is_frozen():
    o = Obs("spans", "issue_refund", "O1", "5.00")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.op = "x"


def test_witness_spec_links_by_id_unless_told_otherwise():
    assert WitnessSpec("spans", frozenset({"a"})).by_id is True
    assert WitnessSpec("env", frozenset({"a", "b"}), by_id=False).by_id is False
