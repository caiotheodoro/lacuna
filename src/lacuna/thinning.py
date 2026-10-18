"""Post-hoc logging loss and witness views over recorded live episodes.

Live episodes are recorded with complete gateway and trace logs. Loss is
applied here, per record, as many times as needed: the faults are
independent of the agent by construction (docs/DECISIONS.md D11, D12), so a
thinned replay has the same distribution as a run with faults switched on.
Paid episodes then give hundreds of fault realisations instead of one.

Env views restrict what W3 can see, by op. `financial` stands in for
change-data-capture on refunds, orders and outbound mail only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lacuna.envs.kernel import LedgerEntry
from lacuna.estimators import Estimate, baselines, bounds, diagnostics, loglinear
from lacuna.estimators.stratified import has_discordance, live_witnesses, strat_bic
from lacuna.histories import keep_witnesses, observed_total
from lacuna.linkage import oracle_table, signature_table
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record

ENV_VIEWS = {
    "full": None,
    "financial": ("issue_refund", "cancel_order", "send_email"),
}

HIDDEN_MOD = 3


@dataclass(frozen=True)
class Config:
    name: str
    p_gateway_drop: float
    p_trace_drop: float
    env_view: str
    # Share of actions whose gateway and trace drops are perfectly coupled,
    # as one logging outage takes both pipes down. rho=0 is the independent
    # fault model of D11; rho>0 violates the independence that every
    # log-linear fit without a W1:W2 term assumes.
    rho: float = 0.0
    # Multiplier on both pipes' drop probability, per op. Missingness then
    # depends on the action itself, not only on a coin: the MNAR case. With
    # the financial env view, giving the ops W3 cannot see a higher drop rate
    # concentrates every witness's blindness on the same actions.
    op_bias: tuple[tuple[str, float], ...] = ()
    # MNAR on a key no witness record shows: every HIDDEN_MOD-th ledger id.
    # Those actions lose both log lines at `hidden_mult` times the rate and,
    # with `hidden_env_blind`, never reach W3. The op of a hidden action is
    # like any other's, so stratifying by op cannot separate it out.
    hidden_mult: float = 1.0
    hidden_env_blind: bool = False

    def is_hidden(self, r: Record) -> bool:
        if self.hidden_mult == 1.0 and not self.hidden_env_blind:
            return False
        return r.oracle is not None and r.oracle % HIDDEN_MOD == 0

    def drop(self, witness_p: float, op: str, hidden: bool = False) -> float:
        for name, mult in self.op_bias:
            if name == op:
                witness_p *= mult
                break
        if hidden:
            witness_p *= self.hidden_mult
        return min(1.0, witness_p)


CONFIGS = (
    Config("light", 0.05, 0.10, "full"),
    Config("sampled", 0.90, 0.30, "full"),
    Config("fin_light", 0.05, 0.10, "financial"),
    Config("fin_sampled", 0.90, 0.30, "financial"),
    Config("fin_sampled_rho50", 0.90, 0.30, "financial", rho=0.5),
    Config("fin_sampled_rho100", 0.90, 0.30, "financial", rho=1.0),
)

# Held out of estimator selection (D27) and of every exploratory run, so the
# confirmatory hypotheses in docs/PRE-REGISTRATION.md are tested on a
# configuration whose results nobody has seen. Not in CONFIGS: running these
# before the confirmatory run would void the holdout and needs a decision
# entry saying so.
HELDOUT = (
    Config("fin_mid", 0.50, 0.50, "financial"),
    Config("fin_mid_rho100", 0.50, 0.50, "financial", rho=1.0),
)

# Added after the confirmatory run, so exploratory (docs/DECISIONS.md D31).
# The ops the financial env view cannot see are also the ops both logs drop
# most, so all three witnesses are blind to the same actions.
MNAR_BIAS = (("update_customer", 1.8), ("write_file", 1.8))
MNAR = (
    Config("fin_mid_mnar", 0.50, 0.50, "financial", op_bias=MNAR_BIAS),
    Config("fin_mid_mnar_rho100", 0.50, 0.50, "financial", rho=1.0, op_bias=MNAR_BIAS),
)

# The boundary of op stratification (docs/MNAR-ROBUSTNESS.md), exploratory:
# the same 1.8x drop and W3 blindness as MNAR, keyed on the ledger id instead
# of the op, with the full env view so W3 is live in every stratum.
_HIDDEN = {"hidden_mult": 1.8, "hidden_env_blind": True}
MNAR_HIDDEN = (
    Config("full_mid_hidden", 0.50, 0.50, "full", **_HIDDEN),
    Config("full_mid_hidden_rho100", 0.50, 0.50, "full", rho=1.0, **_HIDDEN),
)

# Every named pool `scripts/thin_episodes.py --pool` can run. A new pool is one
# entry here, so parallel work adds a line rather than editing the script.
POOLS: dict[str, tuple[Config, ...]] = {
    "exploratory": CONFIGS,
    "heldout": HELDOUT,
    "mnar": MNAR,
    "mnar_hidden": MNAR_HIDDEN,
}


def thin(records: list[Record], cfg: Config, rng: np.random.Generator) -> list[Record]:
    """Drop witness records under `cfg`.

    With rho > 0 a share of actions have their gateway and trace drops driven
    by one shared uniform, so an action whose trace line is lost has
    necessarily lost its gateway line too (p_gateway_drop >= p_trace_drop
    here). Records are keyed by `oracle`, the ledger id, which is how two
    witnesses' records for the same action are recognised; a record without
    one falls back to independent drops.
    """
    view = ENV_VIEWS[cfg.env_view]
    shared: dict[int, float] = {}
    coupled: dict[int, bool] = {}
    out = []
    for r in records:
        if r.witness in (GATEWAY, TRACE) and r.oracle is not None and cfg.rho > 0:
            if r.oracle not in shared:
                shared[r.oracle] = rng.random()
                coupled[r.oracle] = rng.random() < cfg.rho
            u = shared[r.oracle] if coupled[r.oracle] else rng.random()
        else:
            u = rng.random()
        hidden = cfg.is_hidden(r)
        if r.witness == GATEWAY and u < cfg.drop(cfg.p_gateway_drop, r.op, hidden):
            continue
        if r.witness == TRACE and u < cfg.drop(cfg.p_trace_drop, r.op, hidden):
            continue
        if r.witness == ENV and (
            (view is not None and r.op not in view) or (hidden and cfg.hidden_env_blind)
        ):
            continue
        out.append(r)
    return out


def stratified(per_op_tables: dict[str, np.ndarray], alpha: float = 0.05) -> Estimate:
    """Sum of per-op estimates, each at a Bonferroni-corrected level.

    The summed interval covers the total only if every stratum's interval
    covers its own count, because the sum is monotone in each term and one
    miss cannot be cancelled by another. At nominal 0.95 per stratum that is
    0.95**m jointly, which is 0.77 at m=5. Each stratum is therefore run at
    1 - alpha/m, so the joint level is at least 1 - alpha (Bonferroni).

    A stratum with fewer than two live witnesses, or with no discordance
    among them, is unidentified: it contributes `union_bound`, so the total's
    point and upper end go to infinity while the lower end stays the observed
    count, which remains a valid floor.
    """
    strata = list(per_op_tables.values())
    usable = []
    for table in strata:
        k = int(np.log2(len(table)))
        live = live_witnesses(table, k)
        usable.append(len(live) >= 2 and has_discordance(table, live))
    m = max(sum(usable), 1)
    level = 1 - alpha / m

    n = point = lo = hi = 0.0
    n_unidentified = 0
    for table, ok in zip(strata, usable, strict=True):
        obs = observed_total(table)
        n += obs
        if not ok:
            n_unidentified += 1
            point, hi = math.inf, math.inf
            lo += obs
            continue
        k = int(np.log2(len(table)))
        e = diagnostics.lc_gated(keep_witnesses(table, live_witnesses(table, k)), level=level)
        point, lo, hi = point + e.point, lo + e.lo, hi + e.hi
    return Estimate(
        "strat_lc_gated",
        int(n),
        point,
        lo,
        hi,
        {
            "strata": len(strata),
            "unidentified": n_unidentified,
            "level_per_stratum": round(level, 4),
        },
    )


def estimate_all(table4: np.ndarray, per_op: dict[str, np.ndarray]) -> dict[str, Estimate]:
    """Every estimator, over the three hard witnesses only.

    W4 is excluded throughout (docs/DECISIONS.md D20): with capture
    probability 1 it makes `union_count` equal N by construction, which would
    flatter the floor and the method alike.
    """
    t3 = keep_witnesses(table4, (0, 1, 2))
    per_op3 = {op: keep_witnesses(t, (0, 1, 2)) for op, t in per_op.items()}
    return {
        "trust_trace": baselines.trust_trace(t3),
        "union_count": baselines.union_count(t3),
        "union_bound": baselines.union_bound(t3),
        "ll_independence": loglinear.independence(t3),
        "ll_bic": loglinear.best_bic(t3),
        "ll_average": loglinear.averaged(t3),
        "anchor_g1": bounds.anchor_bounds(t3, gamma=1.0),
        "anchor_g2": bounds.anchor_bounds(t3, gamma=2.0),
        "anchor_g4": bounds.anchor_bounds(t3, gamma=4.0),
        "lc_gated": diagnostics.lc_gated(t3),
        "strat_lc_gated": stratified(per_op3),
        "strat_bic": strat_bic(per_op3),
    }


def tables(
    ledger: list[LedgerEntry], records: list[Record], linkage: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pooled 4-witness table and per-op tables, observed (cell 0 zeroed)."""
    ops = sorted({e.op for e in ledger} | {r.op for r in records})
    per_op = {}
    for op in ops:
        recs = [r for r in records if r.op == op]
        if linkage == "oracle":
            t, _ = oracle_table([e for e in ledger if e.op == op], recs, 4)
        else:
            t = signature_table(recs, 4)
        t = t.copy()
        t[0] = 0
        per_op[op] = t
    total = np.sum(list(per_op.values()), axis=0) if per_op else np.zeros(16, dtype=np.int64)
    return total, per_op
