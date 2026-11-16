"""`lacuna.estimators.sa_hom` against the registered `scripts/study5_lib.sa_hom`.

The registered code is frozen with study 5. The library's is the one the tool runs.
They must give the same estimate wherever the registered one gives one. No raw run is
read here: tables are seeded draws and literals.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402

from lacuna.estimators import loglinear  # noqa: E402
from lacuna.estimators import sa_hom as sh  # noqa: E402

# `s5-s0-021`, an unstressed run of study 5: per session and plane, the actions in both
# logs and not in the end state, and the actions in all three. No other cell holds any.
# Read once from the raw run with `lacuna.rig.witnesses.load` and
# `study5_lib.session_tables`; its planes' tables are in `artifacts/study5/confirm.json`.
S0_021 = [
    (4, 3), (2, 5), (3, 2), (2, 4), (3, 4), (3, 4), (3, 2), (2, 4), (1, 6), (1, 5), (1, 8), (1, 5),
    (3, 4), (2, 6), (2, 7), (3, 4), (2, 5), (1, 6), (3, 5), (3, 3), (2, 7), (2, 5), (5, 3), (4, 1),
    (3, 1), (0, 9), (3, 4), (2, 3), (3, 5), (1, 7), (5, 1), (2, 6), (2, 6), (3, 4), (4, 4), (2, 3),
    (0, 4), (2, 5), (3, 5), (1, 6), (4, 3), (3, 3), (3, 3), (3, 3), (4, 2), (4, 2), (4, 3), (4, 2),
]  # fmt: skip


def no_loss(logs_only: int, all_three: int) -> np.ndarray:
    return np.array([0, 0, 0, logs_only, 0, 0, 0, all_three], dtype=np.int64)


def pool(tables: list[np.ndarray]) -> np.ndarray:
    out = np.sum(tables, axis=0).astype(np.int64)
    out[0] = 0
    return out


def drawn(rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """A run's worth of per-session tables and their sum, from one of four regimes."""
    regime = rng.integers(4)
    sessions = int(rng.integers(1, 30))
    if regime == 0:  # every cell in use
        rates = rng.uniform(0.2, 6.0, size=8)
    elif regime == 1:  # sparse: most cells empty in most sessions
        rates = rng.uniform(0.0, 0.6, size=8)
    elif regime == 2:  # a few cells only
        rates = rng.uniform(0.5, 8.0, size=8) * (rng.random(8) < 0.4)
    else:  # no loss: both logs hold everything, the state holds a part
        rates = np.array([0, 0, 0, rng.uniform(0.5, 6), 0, 0, 0, rng.uniform(0.5, 9)])
    tables = [rng.poisson(rates).astype(np.int64) for _ in range(sessions)]
    for t in tables:
        t[0] = 0
    return pool(tables), tables


def cases(n: int = 400) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    rng = np.random.default_rng(20261116)
    return [drawn(rng) for _ in range(n)]


def same_or_repaired(got, want) -> bool:
    """The two agree, or the registered one gave no answer and the library's says how it
    made one. Returns whether it was the second."""
    if "nonconverged" not in want.detail:
        assert got == want, (got, want)
        return False
    # The registered code's "no answer": an infinite point from the count seen up.
    assert (want.point, want.lo, want.hi) == (math.inf, want.n_observed, math.inf)
    assert got.detail == {k: v for k, v in want.detail.items() if k != "nonconverged"} | {
        "closed_form": 1
    }
    assert (got.name, got.n_observed) == (want.name, want.n_observed)
    assert got.point == got.n_observed + got.detail["z0"] and got.lo <= got.point <= got.hi
    return True


def test_the_library_gives_the_registered_estimate_on_seeded_tables():
    seen = {"bounded": 0, "unbounded": 0, "dispersed": 0}
    for pooled, per_session in cases():
        want = lib.sa_hom(pooled, per_session)
        same_or_repaired(sh.sa_hom(pooled, per_session), want)
        seen["bounded" if want.bounded else "unbounded"] += 1
        seen["dispersed"] += want.detail["phi"] > 1
    assert min(seen.values()) >= 20, seen  # the draws reach every branch


def test_on_tables_with_no_loss_the_two_differ_only_where_the_registered_has_no_answer():
    # The shape of every unstressed run of study 5. Which of these the registered code
    # fails on is decided by rounding, so no count of them is asserted.
    for a in range(1, 200, 13):
        for b in range(1, 400, 17):
            pooled = no_loss(a, b)
            got = sh.sa_hom(pooled, [])
            same_or_repaired(got, lib.sa_hom(pooled, []))
            assert (got.point, got.lo) == (a + b, a + b) and a + b <= got.hi < math.inf


@pytest.mark.parametrize(
    "pooled, why",
    [
        ([0, 0, 0, 0, 0, 0, 0, 0], "no actions"),
        ([0, 5, 3, 9, 4, 0, 0, 0], "no action in both lists"),
        ([0, 0, 0, 0, 7, 0, 0, 0], "the state alone"),
        ([0, 30, 30, 30, 30, 1, 0, 0], "the point is past the cap"),
    ],
)
def test_the_registered_guards_are_kept(pooled, why):
    pooled = np.array(pooled, dtype=np.int64)
    for per_session in ([], [pooled], [pooled // 2, pooled - pooled // 2]):
        want = lib.sa_hom(pooled, per_session)
        got = sh.sa_hom(pooled, per_session)
        assert got == want, why  # a guard returns before any fit, so always the same
        assert got.detail["unbounded"] == 1 and "z0" not in got.detail, why
        assert (got.point, got.lo, got.hi) == (math.inf, pooled.sum(), math.inf), why


def test_a_profile_with_no_upper_end_under_the_cap_keeps_its_point():
    pooled = np.array([0, 20, 0, 0, 20, 1, 0, 0])  # 41 seen, 400 more by the point
    got = sh.sa_hom(pooled, [])
    same_or_repaired(got, lib.sa_hom(pooled, []))
    assert got.point == 441.0 <= sh.loglinear.CAP_RATIO * 41 and got.hi == math.inf
    assert 41 < got.lo < 441 and "unbounded" not in got.detail


def test_the_estimate_is_named_as_asked_and_needs_three_witnesses():
    pooled = np.array([0, 4, 6, 20, 9, 3, 5, 40])
    assert sh.sa_hom(pooled, []).name == "ll_sa_hom" == sh.NAME
    assert sh.sa_hom(pooled, [], "other").name == "other"
    assert sh.TELEMETRY_PAIR == lib.TELEMETRY_PAIR
    with pytest.raises(ValueError, match="three witnesses"):
        sh.sa_hom(np.array([0, 4, 6, 20]), [])


def test_the_point_is_lincoln_petersen_between_either_log_and_the_state():
    pooled = np.array([0, 4, 6, 20, 9, 3, 5, 40])
    e = sh.sa_hom(pooled, [])
    assert e.point == pytest.approx(87 + 9 * 30 / 48) and e.detail["z0"] == pytest.approx(5.625)
    assert e.lo <= e.point <= e.hi and math.isfinite(e.hi)
    assert e.detail["phi"] == 1.0 == loglinear.homogeneity_dispersion([pooled])


# The defect: no answer on a complete log ---------------------------------------------


def test_the_run_where_no_log_lost_anything_gets_the_count():
    per_session = [no_loss(a, b) for a, b in S0_021]
    pooled = pool(per_session)
    assert list(pooled) == [0, 0, 0, 123, 0, 0, 0, 202]
    e = sh.sa_hom(pooled, per_session)
    assert e.detail["phi"] == pytest.approx(1.1164593296938115, rel=1e-12)  # as in confirm.json
    assert "nonconverged" not in e.detail and "unbounded" not in e.detail
    # Nothing is in the state alone, so the missing cell is 0 x 123 / 202. One action
    # no witness saw costs 1.968 of log-likelihood and the cutoff here is 2.144 (1.921
    # times this run's dispersion), so the interval holds 326 and not 327.
    assert (e.n_observed, e.point, e.lo, e.hi) == (325, 325.0, 325.0, 326.0)
    assert sh.sa_hom(pooled, []).hi == 325.0  # with no dispersion the cutoff is under 1.968
    same_or_repaired(e, lib.sa_hom(pooled, per_session))


def test_the_closed_form_is_the_likelihood_the_iteration_reaches():
    for pooled, _ in cases(40):
        n = int(pooled.sum())
        if n == 0:
            continue
        exact = sh.ClosedFormProfile(pooled, sh.TELEMETRY_PAIR)
        iterated = loglinear.Profile(pooled, sh.TELEMETRY_PAIR)
        for big_n in (n + 1, n + 3, 2 * n, 5 * n + 2):  # at n itself the iteration can fail
            try:
                want = iterated(big_n)
            except FloatingPointError:
                continue
            assert exact(big_n) == pytest.approx(want, abs=1e-6), (list(pooled), big_n)
    # By hand on the smallest table: both lists hold one action and N = 2, so the
    # fitted cells are 1/2 each and l = log 2! - log 1! + 1 log(1/4) + 1 log(1/4).
    one = sh.ClosedFormProfile(np.array([0, 0, 0, 0, 0, 1, 0, 0]), sh.TELEMETRY_PAIR)
    assert one(2) == pytest.approx(math.log(2) + 2 * math.log(1 / 4))


def test_when_the_iteration_fails_everywhere_the_estimate_is_still_the_registered_one(monkeypatch):
    # The tables on which the registered code fits a profile and has an answer.
    fitted = [(p, s, lib.sa_hom(p, s)) for p, s in cases(150)]
    fitted = [x for x in fitted if not {"unbounded", "nonconverged"} & set(x[2].detail)]
    drawn_cases, want = [x[:2] for x in fitted], [x[2] for x in fitted]
    assert len(want) >= 60

    def fails(self, big_n):
        raise loglinear.ConvergenceError(f"IRLS did not converge at N={big_n}")

    monkeypatch.setattr(loglinear.Profile, "__call__", fails)
    for (pooled, per_session), w in zip(drawn_cases, want, strict=True):
        assert "nonconverged" in lib.sa_hom(pooled, per_session).detail  # the registered path
        got = sh.sa_hom(pooled, per_session)
        assert got.detail == w.detail | {"closed_form": 1}
        assert (got.point, got.lo, got.hi) == (w.point, w.lo, w.hi), list(pooled)
