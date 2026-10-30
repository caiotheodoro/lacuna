"""The floor. Anything lacuna publishes has to beat these, or say that it did not.

- `trust_trace`: believe the agent's own log. What most audits do today.
- `auditor_count`: believe an LLM that rebuilt the actions from the transcript.
- `union_count`: every distinct action any witness saw. Always <= N.
- `union_bound`: the same union read as [n, inf). It always covers and says
  nothing. This is the flag-everything of counting: a method only earns its
  keep by giving a finite upper end while keeping coverage.

Cheaper checks an auditor could run instead of fitting a log-linear model,
added after the confirmatory run so exploratory (scripts/baselines_battery.py):

- `chapman_w1w2`, `chapman_w1w3`, `chapman_w2w3`: two-witness
  Lincoln-Petersen (Chapman) with its standard variance interval.
- `seq_gap`, `seq_gap_lb`, `seq_gap_tail`: the gateway numbers every action it
  routes, so the highest number seen counts the routed actions.
- `canary_joint`, `canary_indep`: the auditor plants known actions and reads
  the loss rate off how many the witnesses missed.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

import numpy as np

from lacuna.envs.kernel import LedgerEntry
from lacuna.estimators import Estimate
from lacuna.histories import observed_total, seen_by
from lacuna.witnesses import ENV, GATEWAY, Record

TRACE, AUDITOR = 0, 3  # witness indices; GATEWAY (1) and ENV (2) come from lacuna.witnesses


def trust_trace(counts: np.ndarray) -> Estimate:
    n = seen_by(counts, TRACE)
    return Estimate("trust_trace", observed_total(counts), n, n, n)


def auditor_count(counts: np.ndarray) -> Estimate:
    n = seen_by(counts, AUDITOR)
    return Estimate("auditor_count", observed_total(counts), n, n, n)


def union_count(counts: np.ndarray) -> Estimate:
    n = observed_total(counts)
    return Estimate("union_count", n, n, n, n)


def union_bound(counts: np.ndarray) -> Estimate:
    n = observed_total(counts)
    return Estimate("union_bound", n, n, n, float("inf"))


Z = 1.96  # two-sided 95%


# --- Chapman -----------------------------------------------------------------


def chapman(counts: np.ndarray, a: int, b: int, *, name: str | None = None) -> Estimate:
    """Two-witness Lincoln-Petersen in Chapman's bias-corrected form.

    Uses witnesses `a` and `b` only and ignores every other column of the
    table. N = (n_a+1)(n_b+1)/(m+1) - 1, Seber's variance
    (n_a+1)(n_b+1)(n_a-m)(n_b-m) / ((m+1)^2 (m+2)), Wald interval clipped below
    at what the pair saw. Needs the two witnesses to capture independently of
    each other and every action to be capturable by both: a witness that
    cannot see some actions (W3 under a financial view) makes this an
    estimate of the population both can see, not of N.
    """
    n_a, n_b = seen_by(counts, a), seen_by(counts, b)
    m = int(sum(c for h, c in enumerate(counts) if h and (h >> a) & 1 and (h >> b) & 1))
    seen = n_a + n_b - m
    point = (n_a + 1) * (n_b + 1) / (m + 1) - 1
    var = (n_a + 1) * (n_b + 1) * (n_a - m) * (n_b - m) / ((m + 1) ** 2 * (m + 2))
    half = Z * math.sqrt(var)
    nm = name or f"chapman_w{a + 1}w{b + 1}"
    return Estimate(nm, seen, point, max(seen, point - half), max(seen, point + half))


def chapman_w1w2(counts: np.ndarray) -> Estimate:
    return chapman(counts, 0, 1)


def chapman_w1w3(counts: np.ndarray) -> Estimate:
    return chapman(counts, 0, 2)


def chapman_w2w3(counts: np.ndarray) -> Estimate:
    return chapman(counts, 1, 2)


# --- Sequence-number gaps ------------------------------------------------------


@dataclass(frozen=True)
class SeqObs:
    """What the gateway's counter shows for one episode.

    `last` is the highest sequence number among surviving gateway records,
    plus one (the count of actions it numbered up to there); `seen` is how
    many records survived. last - seen is the number of visible gaps.
    """

    last: int
    seen: int


def gateway_seq(records: Sequence[Record], bypass: Collection[int] = ()) -> dict[int, int]:
    """Gateway sequence number of each action, from the complete log.

    The recorded gateway log lists one record per routed action in routing
    order and carries no counter field, so the number is the record's position
    in that log. This assumes the gateway's own counter is never lost, only
    the records it numbered: the generous reading of a sequence check.
    `bypass` lists ledger ids that never touched the gateway; they take no
    number, so the counter counts routed actions only.
    """
    seq: dict[int, int] = {}
    for r in records:
        if r.witness == GATEWAY and r.oracle is not None and r.oracle not in bypass:
            seq[r.oracle] = len(seq)
    return seq


def seq_obs(kept: Sequence[Record], seq: dict[int, int]) -> SeqObs:
    """The counter as an auditor reads it from the gateway records that survived."""
    nums = [seq[r.oracle] for r in kept if r.witness == GATEWAY and r.oracle in seq]
    return SeqObs(max(nums) + 1 if nums else 0, len(set(nums)))


def seq_gap(obs: Sequence[SeqObs], n_observed: int) -> Estimate:
    """Sum over episodes of the highest gateway number seen, as a point.

    Exact when every action routes through the gateway and each episode's last
    record survives; short by the lost tail otherwise, and by every bypassed
    action always. The interval is the point, the way `trust_trace` reports
    one, so coverage scores it as an exact claim.
    """
    n = sum(o.last for o in obs)
    return Estimate("seq_gap", n_observed, n, n, n)


def seq_gap_lb(obs: Sequence[SeqObs], n_observed: int) -> Estimate:
    """The same count read as a lower bound [n, inf): valid, and as silent as union_bound."""
    n = sum(o.last for o in obs)
    return Estimate("seq_gap_lb", n_observed, n, n, float("inf"))


def seq_gap_tail(obs: Sequence[SeqObs], n_observed: int) -> Estimate:
    """`seq_gap` plus the expected lost tail, for the one thing a counter cannot see.

    Records lost after an episode's last surviving number leave no gap. If each
    record is lost independently with rate p, estimated from the gaps before
    the last survivor, the tail per episode is geometric with mean p/(1-p).
    The interval is Wald on the summed tails, with p's own uncertainty by the
    delta method. Wrong, by construction, when loss comes in runs or differs
    by episode: the tail is then longer than a geometric.
    """
    base = sum(o.last for o in obs)
    positions = sum(max(o.last - 1, 0) for o in obs)  # slots before each last survivor
    gaps = sum(o.last - o.seen for o in obs)
    p = min(gaps / positions, 0.95) if positions else 0.0
    e = len(obs)
    tail = e * p / (1 - p)
    var = e * p / (1 - p) ** 2
    if positions:
        var += (e / (1 - p) ** 2) ** 2 * p * (1 - p) / positions
    half = Z * math.sqrt(var)
    point = base + tail
    return Estimate(
        "seq_gap_tail", n_observed, point, max(base, point - half), point + half, {"loss_rate": p}
    )


# --- Canaries -------------------------------------------------------------------

CANARY_WITNESSES = (TRACE, GATEWAY, ENV)
_FAR = 10**6  # canary ids when not interleaved: far above any ledger id


def plant_canaries(
    ledger: Sequence[LedgerEntry],
    records: Sequence[Record],
    k: int,
    rng: np.random.Generator,
    *,
    ops: Sequence[str] | None = None,
    interleave: bool = True,
) -> tuple[list[LedgerEntry], list[Record], frozenset[int]]:
    """Add `k` known synthetic actions to one episode, as an auditor planting canaries would.

    Each canary gets a unique resource and value (so it never merges with a
    real action under signature linkage) and a record for every witness that
    could see it; the loss process then decides which survive. Ops default to
    draws from the episode's own ledger, so the canary op mix is the real one;
    `ops` instead cycles through the given list, which is how a
    miscalibrated canary set is built. With `interleave` the canaries take
    random positions in the episode and real ledger ids are renumbered, so
    anything keyed on order (bursts) treats them like the rest. Without it
    they get far-off ids that are never a multiple of 3, which keeps them
    outside an id-keyed hidden class, as a key the auditor cannot see would.
    """
    led = sorted(ledger, key=lambda e: e.id)
    n = len(led)
    if ops is None:
        pool = [e.op for e in led]
        if k > 0 and not pool:
            raise ValueError(
                "cannot draw canary ops from an empty ledger; pass `ops` to plant canaries anyway"
            )
        canary_ops = [pool[i] for i in rng.integers(0, len(pool), k)] if pool else []
    else:
        canary_ops = [ops[j % len(ops)] for j in range(k)]
    if interleave:
        slots = set(rng.choice(n + k, size=k, replace=False).tolist())
        real_slots = [i for i in range(n + k) if i not in slots]
        remap = {e.id: real_slots[i] for i, e in enumerate(led)}
        cids = sorted(slots)
    else:
        remap = {e.id: e.id for e in led}
        cids = [_FAR + 3 * j + 1 for j in range(k)]
    new_ledger = [replace(e, id=remap[e.id]) for e in led]
    canary_entries = [
        LedgerEntry(cid, op, f"CANARY-{j}", f"canary-{j}", -1)
        for j, (cid, op) in enumerate(zip(cids, canary_ops, strict=True))
    ]
    new_records = [
        replace(r, oracle=remap.get(r.oracle, r.oracle)) if r.oracle is not None else r
        for r in records
    ]
    for w in CANARY_WITNESSES:
        new_records += [Record(w, c.op, c.resource, c.value, c.id) for c in canary_entries]
    return sorted(new_ledger + canary_entries, key=lambda e: e.id), new_records, frozenset(cids)


def strip_canaries(records: Sequence[Record], canary_ids: Collection[int]) -> list[Record]:
    """What the estimator under test sees: the real actions' records only."""
    return [r for r in records if r.oracle not in canary_ids]


def canary_histories(records: Sequence[Record], canary_ids: Collection[int]) -> list[int]:
    """Capture history of each canary (witness bits 0..2), in canary-id order."""
    hist = {c: 0 for c in sorted(canary_ids)}
    for r in records:
        if r.oracle in hist and r.witness in CANARY_WITNESSES:
            hist[r.oracle] |= 1 << r.witness
    return list(hist.values())


def canary_estimate(n_observed: int, histories: Sequence[int], *, joint: bool = True) -> Estimate:
    """N = (distinct real actions any witness saw) / (1 - P(missed by all)), P from canaries.

    `joint` reads P as the share of canaries no witness captured, which keeps
    any dependence between witnesses; otherwise P is the product of the
    witnesses' marginal miss rates, which assumes the witnesses independent.
    Both assume canaries are exchangeable with real actions: same op mix, same
    chance of being lost. Loss that depends on something the canary does not
    carry (an op mix the canaries do not match, a hidden key) breaks that and
    the estimate inherits the bias with a confident interval.

    With no canaries (`histories` empty) the miss rate is unknown and the estimate is
    unbounded above: point and upper end infinite, lower end the observed count.

    Interval: delta method on U/(1-q), with U binomial and q from K canaries,
    q's variance taken at the Agresti-Coull adjusted rate so a lucky run of
    all-captured canaries does not give a zero-width interval.
    """
    k = len(histories)
    if k == 0:
        # No canaries, no estimate of the miss rate: nothing bounds N above.
        name = "canary_joint" if joint else "canary_indep"
        return Estimate(name, n_observed, math.inf, n_observed, math.inf, {"q_miss": None, "k": 0})
    hs = [int(h) for h in histories]
    if joint:
        miss = sum(h == 0 for h in hs)
        q, q_adj = miss / k, (miss + 2) / (k + 4)
        var_q = q_adj * (1 - q_adj) / k
    else:
        miss_w = [sum(not (h >> w) & 1 for h in hs) for w in CANARY_WITNESSES]
        q = math.prod(m / k for m in miss_w)
        adj = [(m + 2) / (k + 4) for m in miss_w]
        q_adj = math.prod(adj)
        var_q = sum((q_adj / a) ** 2 * a * (1 - a) / k for a in adj)
    name = "canary_joint" if joint else "canary_indep"
    if q >= 1.0:
        return Estimate(name, n_observed, math.inf, n_observed, math.inf, {"q_miss": q})
    point = n_observed / (1 - q)
    n_adj = n_observed / (1 - q_adj)
    var = n_adj * q_adj / (1 - q_adj) + n_adj**2 * var_q / (1 - q_adj) ** 2
    half = Z * math.sqrt(var)
    return Estimate(
        name, n_observed, point, max(n_observed, point - half), point + half, {"q_miss": q, "k": k}
    )
