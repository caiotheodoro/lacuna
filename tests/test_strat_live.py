"""`strat_bic(live=...)`: a log that saw nothing is not a log that could not look."""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.estimators.stratified import strat_bic

# Three witnesses, all active: cells indexed by history bits.
FULL = np.array([0, 6, 7, 20, 5, 9, 8, 30], dtype=np.int64)
# The third witness recorded nothing (every cell with bit 2 is zero).
WIPED = np.array([0, 6, 7, 20, 0, 0, 0, 0], dtype=np.int64)


def test_default_is_unchanged_and_adds_no_key():
    est = strat_bic({"a": FULL, "b": WIPED})
    assert "wiped" not in est.detail


def test_declared_witness_with_no_records_is_counted_as_wiped():
    plain = strat_bic({"a": FULL, "b": WIPED})
    est = strat_bic({"a": FULL, "b": WIPED}, live={"a": (0, 1, 2), "b": (0, 1, 2)})
    assert est.detail["wiped"] == 1
    assert (est.point, est.lo, est.hi) == (plain.point, plain.lo, plain.hi)


def test_structurally_blind_witness_is_not_wiped():
    est = strat_bic({"a": FULL, "b": WIPED}, live={"a": (0, 1, 2), "b": (0, 1)})
    assert est.detail["wiped"] == 0


def test_counts_from_an_undeclared_witness_are_an_error():
    with pytest.raises(ValueError, match="not declared"):
        strat_bic({"a": FULL}, live={"a": (0, 1)})


def test_live_must_name_every_stratum_with_data():
    with pytest.raises(ValueError, match="no declared witnesses"):
        strat_bic({"a": FULL, "b": WIPED}, live={"a": (0, 1, 2)})
