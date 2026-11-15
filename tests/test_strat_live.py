"""`strat_bic(live=...)`: a log that recorded nothing is not a log that could not look."""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.estimators.stratified import strat_bic

# Three witnesses, all active: cells indexed by history bits.
FULL = np.array([0, 6, 7, 20, 5, 9, 8, 30], dtype=np.int64)
# The third witness recorded nothing (every cell with bit 2 is zero).
QUIET = np.array([0, 6, 7, 20, 0, 0, 0, 0], dtype=np.int64)
EMPTY = np.zeros(8, dtype=np.int64)
ALL = (0, 1, 2)


def test_default_is_what_it_was_before_the_argument_existed():
    # Printed by the function as it stood at 65758fa^ (before `live`), on these tables.
    est = strat_bic({"a": FULL, "b": QUIET})
    assert (est.n_observed, est.point, est.lo, est.hi) == (118, 123.0, 119.0, 130.0)
    assert est.detail == {
        "strata": 2,
        "identified": 2,
        "unidentified": 0,
        "unbounded": 0,
        "empty": 0,
        "unbounded_mass": 0.0,
    }


def test_declared_witness_with_no_records_is_counted_and_changes_no_number():
    plain = strat_bic({"a": FULL, "b": QUIET})
    est = strat_bic({"a": FULL, "b": QUIET}, live={"a": ALL, "b": ALL})
    assert est.detail["silent"] == 1
    assert (est.n_observed, est.point, est.lo, est.hi) == (
        plain.n_observed,
        plain.point,
        plain.lo,
        plain.hi,
    )
    assert {k: v for k, v in est.detail.items() if k != "silent"} == plain.detail


def test_a_witness_that_cannot_see_the_stratum_is_not_silent():
    est = strat_bic({"a": FULL, "b": QUIET}, live={"a": ALL, "b": (0, 1)})
    assert est.detail["silent"] == 0


def test_a_declared_stratum_where_nobody_recorded_anything_is_silent_too():
    est = strat_bic({"a": FULL, "b": EMPTY}, live={"a": ALL, "b": ALL})
    assert est.detail["silent"] == 1 and est.detail["empty"] == 1


def test_counts_from_an_undeclared_witness_are_an_error():
    with pytest.raises(ValueError, match="not declared"):
        strat_bic({"a": FULL}, live={"a": (0, 1)})


def test_live_must_name_every_stratum_with_data():
    with pytest.raises(ValueError, match="no declared witnesses"):
        strat_bic({"a": FULL, "b": QUIET}, live={"a": ALL})


def test_a_declared_witness_must_exist_in_the_table():
    with pytest.raises(ValueError, match="no such witness"):
        strat_bic({"a": FULL}, live={"a": (0, 1, 2, 3)})
