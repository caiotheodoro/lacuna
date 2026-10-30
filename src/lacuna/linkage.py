"""From witness records to a capture-history count table, two ways.

Oracle linkage knows which ledger entry each record belongs to. It is the
upper bound: the table an auditor would get with perfect record matching.

Effect-signature linkage is what an auditor can actually do. Records are
grouped by signature (op, resource, and the value where every witness can
reproduce it). Within a signature, if witness w has c_w records, there are
at least max_w c_w distinct actions, and the records are paired in order.
In-order pairing maximises overlap, which biases N slightly low; the gap
between the two linkages measures that bias (docs/PLAN.md).

Content digests (file bodies, email text) are left out of signatures, because
an auditor reading a transcript cannot reproduce a hash of text it
paraphrased. Two different emails to one address in one episode therefore
look like one action seen twice. That is a real linkage error, and it is
measured, not hidden.

Agent-supplied identifiers are never used (AGENTS.md rule 4).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np

from lacuna.envs.infra_tools import VALUE_OPS as INFRA_VALUE_OPS
from lacuna.envs.kernel import LedgerEntry
from lacuna.witnesses import AUDITOR, ENV, GATEWAY, TRACE, Record

BACKOFFICE_VALUE_OPS = ("issue_refund", "update_customer", "cancel_order", "delete_file")
# One union over every env. Op names are disjoint across envs
# (tests/test_infra.py), so adding an env leaves the others' signatures as
# they were, and the tables need not know which env produced the records.
VALUE_OPS = BACKOFFICE_VALUE_OPS + INFRA_VALUE_OPS


def signature(op: str, resource: str, value: str) -> tuple[str, str, str]:
    return op, resource, value if op in VALUE_OPS else ""


def oracle_table(
    ledger: list[LedgerEntry], records: list[Record], k: int
) -> tuple[np.ndarray, dict]:
    """Full 2**k table including the true cell 0, plus auditor match stats.

    Auditor records carry no ledger id, so they are matched to ledger entries
    by signature, each entry at most once. That is the best case for W4;
    unmatched auditor records are its false positives.
    """
    hist = {e.id: 0 for e in ledger}
    for r in records:
        if r.witness != AUDITOR and r.oracle is not None and r.oracle in hist:
            hist[r.oracle] |= 1 << r.witness

    free: dict[tuple, list[int]] = defaultdict(list)
    for e in ledger:
        free[signature(e.op, e.resource, e.value)].append(e.id)
    false_pos = 0
    for r in records:
        if r.witness != AUDITOR:
            continue
        pool = free.get(signature(r.op, r.resource, r.value))
        if pool:
            hist[pool.pop(0)] |= 1 << AUDITOR
        else:
            false_pos += 1

    table = np.zeros(2**k, dtype=np.int64)
    for h in hist.values():
        table[h & (2**k - 1)] += 1
    return table, {"auditor_false_positives": false_pos}


def signature_table(records: list[Record], k: int) -> np.ndarray:
    """Observed 2**k table (cell 0 is 0) from signature clusters."""
    per_sig: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        if r.witness < k:
            per_sig[signature(r.op, r.resource, r.value)][r.witness] += 1
    table = np.zeros(2**k, dtype=np.int64)
    for counts in per_sig.values():
        for j in range(max(counts.values())):
            h = sum(1 << w for w, c in counts.items() if j < c)
            table[h] += 1
    return table


def callid_table(records: list[Record], key: Callable[[Record], tuple]) -> np.ndarray:
    """Trace and gateway linked by call id; W3 attached by `key`, in order.

    `Record.oracle` on a trace or gateway record is the ledger id the harness
    looked up from that line's call id (`witnesses.py`), so grouping by it is
    grouping by call id. W3 records are never grouped by it.
    """
    calls: dict = defaultdict(int)
    key_of: dict = {}
    for r in records:
        if r.witness in (TRACE, GATEWAY):
            calls[r.oracle] |= 1 << r.witness
            key_of[r.oracle] = key(r)
    by_key: dict = defaultdict(list)
    for cid, bits in calls.items():
        by_key[key_of[cid]].append(bits)
    w3: dict = defaultdict(int)
    for r in records:
        if r.witness == ENV:
            w3[key(r)] += 1
    t = np.zeros(8, dtype=np.int64)
    for k in set(by_key) | set(w3):
        seen = sorted(by_key.get(k, []), key=lambda b: -bin(b).count("1"))
        c3 = w3.get(k, 0)
        for j, bits in enumerate(seen):
            t[bits | (4 if j < c3 else 0)] += 1
        t[4] += max(0, c3 - len(seen))
    return t
