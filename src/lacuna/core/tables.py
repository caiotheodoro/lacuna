"""From named-witness records to capture-history count tables.

`keyed_table` is `lacuna.linkage.callid_table` for any number of witnesses
(`tests/test_core_tables.py` pins the two equal on three). Witnesses that
share an id per action are linked by it; the rest are attached by a caller's
key, in order, to the actions the id witnesses agree on most. In-order
attachment maximises overlap, the same bias `lacuna.linkage` documents.

A table is a length-2**K integer array indexed by history bits, witness
`names[0]` at bit 0 (`lacuna.histories`). Cell 0 is always 0 here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence

import numpy as np

from lacuna.core.records import Obs, WitnessSpec


def keyed_table(
    obs: Iterable[Obs],
    names: Sequence[str],
    id_witnesses: Sequence[str],
    key: Callable[[Obs], Hashable],
) -> np.ndarray:
    """One table over `names`.

    A record with no `link_id` is its own action. Three things the caller owns:
    pass records in time order, because among actions the id witnesses saw
    equally well an attached record goes to the earliest; give one action one
    key, because when the id witnesses of one link id disagree on the key the
    last record read decides; and keep link ids unique within the records given.
    """
    if len(set(names)) != len(names):
        raise ValueError(f"a witness is named more than once in {list(names)}")
    bit = {n: i for i, n in enumerate(names)}
    stray = [w for w in id_witnesses if w not in bit]
    if stray:
        raise ValueError(f"id witnesses {stray} are not among the witnesses {list(names)}")
    ids = set(id_witnesses)

    groups: dict[Hashable, int] = defaultdict(int)
    key_of: dict[Hashable, Hashable] = {}
    attached: dict[Hashable, Counter] = defaultdict(Counter)
    for o in obs:
        if o.witness not in bit:
            continue
        if o.witness in ids:
            gid = o.link_id if o.link_id is not None else object()
            groups[gid] |= 1 << bit[o.witness]
            key_of[gid] = key(o)
        else:
            attached[key(o)][bit[o.witness]] += 1

    by_key: dict[Hashable, list[int]] = defaultdict(list)
    for gid, bits in groups.items():
        by_key[key_of[gid]].append(bits)

    table = np.zeros(2 ** len(names), dtype=np.int64)
    for k in set(by_key) | set(attached):
        seen = sorted(by_key.get(k, []), key=lambda b: -bin(b).count("1"))
        counts = attached.get(k, {})
        for j in range(max(len(seen), max(counts.values(), default=0))):
            bits = seen[j] if j < len(seen) else 0
            table[bits | sum(1 << w for w, c in counts.items() if j < c)] += 1
    return table


def build(
    obs: Iterable[Obs],
    specs: Sequence[WitnessSpec],
    key: Callable[[Obs], Hashable],
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """One table per plane, over the witnesses whose spec says they see it.

    A witness that cannot see a plane is left out of that plane's table; a zero
    there would read as "saw nothing" when the truth is "could not look".
    A record that contradicts the specs is an error, not a count. Link ids are
    scoped to a plane: the same id in two planes is two actions.
    """
    by_name = {s.name: s for s in specs}
    if len(by_name) != len(specs):
        raise ValueError(f"a witness is named more than once in {[s.name for s in specs]}")
    strata = sorted(set().union(*(s.sees for s in specs)))
    per: dict[str, list[Obs]] = {s: [] for s in strata}
    for o in obs:
        spec = by_name.get(o.witness)
        if spec is None:
            raise ValueError(f"record from witness {o.witness!r}, which has no spec")
        if o.stratum is None:
            raise ValueError(f"record from {o.witness!r} has no stratum")
        if o.stratum not in spec.sees:
            raise ValueError(f"{o.witness!r} cannot see {o.stratum!r}, yet recorded there")
        per[o.stratum].append(o)

    out: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for s in strata:
        seeing = [sp for sp in specs if s in sp.sees]
        if len(seeing) < 2:
            raise ValueError(f"plane {s!r} has fewer than two witnesses")
        names = tuple(sp.name for sp in seeing)
        ids = tuple(sp.name for sp in seeing if sp.by_id)
        out[s] = (names, keyed_table(per[s], names, ids, key))
    return out
