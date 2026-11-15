"""What one witness claims happened, with the witness named, not numbered.

`lacuna.witnesses.Record` fixes four witnesses by index. `Obs` names its
witness and says which plane (`stratum`) the action went through, so the same
code serves any number of logs.

`link_id` is an id that the logs of one plane share for one action (a trace
id, a call id). A witness that has no such id, such as a state diff, is
attached by signature instead (`WitnessSpec.by_id = False`). `oracle` is the
ground-truth ledger id where a test harness knows it; no estimator reads it.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class Obs:
    witness: str
    op: str
    resource: str
    value: str
    stratum: str | None = None
    link_id: Hashable | None = None
    ts: float | None = None
    seq: int | None = None
    oracle: int | None = None


@dataclass(frozen=True)
class WitnessSpec:
    """A witness, the planes it can see, and whether its records carry `link_id`."""

    name: str
    sees: frozenset[str]
    by_id: bool = True
