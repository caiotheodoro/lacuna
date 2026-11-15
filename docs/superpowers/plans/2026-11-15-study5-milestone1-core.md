# Study 5, milestone 1: core records and tables, httpx decoupling, two spikes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give lacuna a witness-agnostic core (K named witnesses, per-plane tables) that imports with numpy alone, and answer the two questions the rig design depends on.

**Architecture:** A new `lacuna.core` package holds records and table building and imports nothing from `lacuna.envs`. Existing modules are not rewritten: `linkage.callid_table` stays as it is and the new `keyed_table` is pinned equal to it by a test, because the frozen analyses call it inside replicate loops. `strat_bic` gains one optional argument whose default leaves every published number unchanged.

**Tech stack:** Python 3.11+, numpy, pytest, ruff, uv. Spikes use Docker, the OpenTelemetry Python SDK and Collector, and agentgateway.

**Spec:** `docs/DESIGN-STUDY5.md`. Later milestones (rig, study 5, CLI, paper) get their own plans.

**Conventions for every task**
- Run tests with `PYTHONPATH=src uv run pytest -q <path>`.
- Before each commit: `uv run ruff check src tests scripts && uv run ruff format src tests scripts`.
- Commits use the repo-local identity (already set). No attribution footers. Do not push.
- Nothing in `make validate` may need Docker or the network (`tests/test_ci_hermetic.py`). The spikes live under `rig/spikes/` and are run by hand.

## File map

| file | status | responsibility |
|---|---|---|
| `src/lacuna/core/__init__.py` | new | package docstring only |
| `src/lacuna/core/records.py` | new | `Obs`, `WitnessSpec` |
| `src/lacuna/core/tables.py` | new | `keyed_table`, `build` |
| `src/lacuna/estimators/stratified.py` | changed | `strat_bic(..., live=None)` |
| `src/lacuna/envs/prompts.py` | new | `AGENT_SYSTEM`, with no imports |
| `src/lacuna/agents/llm.py`, `src/lacuna/envs/registry.py` | changed | import the prompt from `envs.prompts` |
| `pyproject.toml`, `uv.lock` | changed | httpx becomes the `agents` extra |
| `tests/test_core_records.py`, `tests/test_core_tables.py`, `tests/test_core_imports.py`, `tests/test_strat_live.py` | new | tests for the above |
| `rig/spikes/collector_kill/`, `rig/spikes/gateway_fields/` | new | hand-run spikes |
| `docs/RIG-FINDINGS.md` | new | what the spikes found |

---

### Task 1: `Obs` and `WitnessSpec`

**Files:**
- Create: `src/lacuna/core/__init__.py`, `src/lacuna/core/records.py`
- Test: `tests/test_core_records.py`

- [ ] **Step 1: Write the failing test**

`tests/test_core_records.py`:

```python
"""`lacuna.core.records`: records that do not know which environment made them."""

from __future__ import annotations

import dataclasses

import pytest

from lacuna.core.records import Obs, WitnessSpec


def test_obs_defaults_and_is_hashable():
    o = Obs("spans", "issue_refund", "O1", "5.00")
    assert (o.stratum, o.link_id, o.ts, o.seq, o.oracle) == (None, None, None, None, None)
    assert {o: 1}[Obs("spans", "issue_refund", "O1", "5.00")] == 1


def test_obs_is_frozen():
    o = Obs("spans", "issue_refund", "O1", "5.00")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.op = "x"


def test_witness_spec_links_by_id_unless_told_otherwise():
    assert WitnessSpec("spans", frozenset({"a"})).by_id is True
    assert WitnessSpec("env", frozenset({"a", "b"}), by_id=False).by_id is False
```

- [ ] **Step 2: Run it and see it fail**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_records.py`
Expected: collection error, `ModuleNotFoundError: No module named 'lacuna.core'`.

- [ ] **Step 3: Implement**

`src/lacuna/core/__init__.py`:

```python
"""Records and tables that know nothing about any environment.

Nothing here imports `lacuna.envs` or `lacuna.agents`, so this is the part a
user installs to run the estimator on their own logs.
"""
```

`src/lacuna/core/records.py`:

```python
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
```

- [ ] **Step 4: Run the test**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_records.py`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests && uv run ruff format src tests
git add src/lacuna/core tests/test_core_records.py
git commit -m "core: Obs and WitnessSpec, records with named witnesses"
```

---

### Task 2: `keyed_table`, pinned equal to `callid_table`

**Files:**
- Create: `src/lacuna/core/tables.py`
- Test: `tests/test_core_tables.py`

`keyed_table` generalises `lacuna.linkage.callid_table` (read it first: 28 lines). Records of id-carrying witnesses are grouped by `link_id`; every other witness is attached by `key`, in order, to the groups seen by the most witnesses first. A record with `link_id=None` in an id-carrying witness is its own action.

- [ ] **Step 1: Write the failing tests**

`tests/test_core_tables.py`:

```python
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
    recs = []
    for lid in range(int(rng.integers(5, 40))):
        op = ("cancel_order", "issue_refund")[int(rng.integers(2))]
        res = f"O{int(rng.integers(6))}"  # few resources, so signatures collide
        val = "cancelled" if op == "cancel_order" else f"{int(rng.integers(3))}.00"
        for w in (TRACE, GATEWAY, ENV, AUDITOR):
            if rng.random() < 0.6:
                oracle = None if rng.random() < 0.1 else lid
                recs.append(Record(w, op, res, val, oracle=oracle))
    order = rng.permutation(len(recs))
    return [recs[i] for i in order]


def test_keyed_table_equals_callid_table_on_random_records():
    rng = np.random.default_rng(20261115)
    for _ in range(300):
        recs = _random_records(rng)
        old = linkage.callid_table(recs, lambda r: linkage.signature(r.op, r.resource, r.value))
        obs = [_to_obs(r) for r in recs if r.witness in (TRACE, GATEWAY, ENV)]
        new = keyed_table(obs, NAMES, ("trace", "gateway"), _sig)
        assert new.tolist() == old.tolist()


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
    assert keyed_table(obs, ("trace", "gateway"), ("trace", "gateway"), _sig).tolist() == [0, 2, 0, 0]


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
    assert len(t) == 16 and t[0b1111] == 1 and t.sum() == 1


def test_records_of_unlisted_witnesses_are_ignored():
    obs = [Obs("trace", "op", "r", "v", link_id=1), Obs("other", "op", "r", "v", link_id=1)]
    assert keyed_table(obs, ("trace", "gateway"), ("trace", "gateway"), _sig).tolist() == [0, 1, 0, 0]


def test_id_witness_must_be_a_named_witness():
    with pytest.raises(ValueError, match="not among the witnesses"):
        keyed_table([], ("trace", "gateway"), ("spans",), _sig)
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_tables.py`
Expected: collection error, `ModuleNotFoundError: No module named 'lacuna.core.tables'`.

- [ ] **Step 3: Implement `keyed_table`**

`src/lacuna/core/tables.py`:

```python
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
```

- [ ] **Step 4: Run the `keyed_table` tests**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_tables.py -k "not build"`
Expected: the import of `build` still fails collection. Temporarily confirm with
`PYTHONPATH=src uv run python -c "from lacuna.core.tables import keyed_table"` (no output), then go straight to Task 3, which adds `build` and makes the file collect. Do not commit between Task 2 and Task 3.

---

### Task 3: `build`, one table per plane

**Files:**
- Modify: `src/lacuna/core/tables.py` (append)
- Test: `tests/test_core_tables.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_core_tables.py`:

```python
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
```

- [ ] **Step 2: Run and see `build` missing**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_tables.py`
Expected: collection error, `ImportError: cannot import name 'build'`.

- [ ] **Step 3: Implement `build`**

Append to `src/lacuna/core/tables.py`:

```python
def build(
    obs: Iterable[Obs],
    specs: Sequence[WitnessSpec],
    key: Callable[[Obs], Hashable],
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """One table per plane, over the witnesses whose spec says they see it.

    A witness that cannot see a plane is left out of that plane's table; a zero
    there would read as "saw nothing" when the truth is "could not look".
    A record that contradicts the specs is an error, not a count.
    """
    by_name = {s.name: s for s in specs}
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
```

- [ ] **Step 4: Run the whole file**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_tables.py`
Expected: `13 passed` (7 from Task 2, 6 here: two plain, three parametrised cases, one single-witness).

- [ ] **Step 5: Commit Tasks 2 and 3 together**

```bash
uv run ruff check src tests && uv run ruff format src tests
git add src/lacuna/core/tables.py tests/test_core_tables.py
git commit -m "core: keyed_table for K witnesses (equal to callid_table on three) and per-plane build"
```

---

### Task 4: `strat_bic` with declared visibility

**Files:**
- Modify: `src/lacuna/estimators/stratified.py` (function `strat_bic`, lines 149-191)
- Test: `tests/test_strat_live.py`

Today `live_witnesses` decides which witnesses "see" a stratum from the counts, so a log emptied by a kill looks the same as a log that cannot see the plane. With `live` given, the estimate is unchanged (a witness with no records carries no information about N) but the result says how many strata had a declared witness with nothing in it, and a count from an undeclared witness is an error. With `live=None` nothing changes, including the keys of `detail`.

- [ ] **Step 1: Write the failing tests**

`tests/test_strat_live.py`:

```python
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
```

- [ ] **Step 2: Run and see them fail**

Run: `PYTHONPATH=src uv run pytest -q tests/test_strat_live.py`
Expected: 4 failures with `TypeError: strat_bic() got an unexpected keyword argument 'live'`; `test_default_is_unchanged_and_adds_no_key` passes.

- [ ] **Step 3: Implement**

In `src/lacuna/estimators/stratified.py`, replace the signature and loop head of `strat_bic`. The old lines:

```python
def strat_bic(per_op_tables: dict[str, np.ndarray], level: float = 0.95) -> Estimate:
    n_total = floor = 0
    parts: list[tuple[int, np.ndarray]] = []
    tails: list[float] = []
    counts = {"strata": 0, "identified": 0, "unidentified": 0, "unbounded": 0, "empty": 0}
    for table in per_op_tables.values():
        n = observed_total(table)
        if n == 0:
            counts["empty"] += 1
            continue
        counts["strata"] += 1
        n_total += n
        live = live_witnesses(table, n_witnesses(table))
        if len(live) < 2 or not has_discordance(table, live):
```

become:

```python
def strat_bic(
    per_op_tables: dict[str, np.ndarray],
    level: float = 0.95,
    live: dict[str, tuple[int, ...]] | None = None,
) -> Estimate:
    """`live` maps a stratum to the witnesses declared able to see it.

    Given, it changes no estimate: a witness with no records says nothing about
    N. It adds `detail["wiped"]`, the strata where a declared witness recorded
    nothing, and it refuses counts from a witness that was not declared.
    """
    n_total = floor = 0
    parts: list[tuple[int, np.ndarray]] = []
    tails: list[float] = []
    counts = {"strata": 0, "identified": 0, "unidentified": 0, "unbounded": 0, "empty": 0}
    if live is not None:
        counts["wiped"] = 0
    for name, table in per_op_tables.items():
        n = observed_total(table)
        if n == 0:
            counts["empty"] += 1
            continue
        counts["strata"] += 1
        n_total += n
        seen = live_witnesses(table, n_witnesses(table))
        if live is not None:
            if name not in live:
                raise ValueError(f"stratum {name!r} has data but no declared witnesses")
            stray = sorted(set(seen) - set(live[name]))
            if stray:
                raise ValueError(f"stratum {name!r}: witnesses {stray} recorded but are not declared")
            if set(live[name]) - set(seen):
                counts["wiped"] += 1
        if len(seen) < 2 or not has_discordance(table, seen):
```

Then, in the rest of the function body, rename the two remaining uses of the old local `live` to `seen`. There is exactly one below the loop head:

```python
        dens = stratum_density(keep_witnesses(table, seen))
```

- [ ] **Step 4: Run the new tests and the estimator's existing ones**

Run: `PYTHONPATH=src uv run pytest -q tests/test_strat_live.py && PYTHONPATH=src uv run pytest -q -k "strat or tier or stratified"`
Expected: `5 passed`, then every previously passing stratified test still passing.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests && uv run ruff format src tests
git add src/lacuna/estimators/stratified.py tests/test_strat_live.py
git commit -m "strat_bic: optional declared visibility, reporting wiped witnesses"
```

---

### Task 5: the core imports without httpx

**Files:**
- Create: `src/lacuna/envs/prompts.py`
- Modify: `src/lacuna/agents/llm.py:127-131`, `src/lacuna/envs/registry.py:9`, `pyproject.toml`, `uv.lock`
- Test: `tests/test_core_imports.py`

`envs/registry.py` imports `AGENT_SYSTEM` from `agents/llm.py`, which imports httpx, so importing any environment needs httpx. The prompt moves to a module with no imports; its text, and therefore the sha256 that `tests/test_organism_l2.py` pins, does not change.

- [ ] **Step 1: Write the failing test**

`tests/test_core_imports.py`:

```python
"""What the estimator needs to import: numpy, and nothing an agent run needs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _modules_after(imports: str) -> set[str]:
    code = f"import sys\n{imports}\nprint('\\n'.join(sorted(sys.modules)))"
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return set(r.stdout.split())


def test_core_imports_no_environment_and_no_agent():
    mods = _modules_after("import lacuna.core.tables, lacuna.core.records")
    assert not [m for m in mods if m.startswith(("lacuna.envs", "lacuna.agents"))]
    assert "httpx" not in mods


def test_estimators_and_environments_import_without_httpx():
    mods = _modules_after(
        "import lacuna.estimators.stratified, lacuna.estimators.loglinear\n"
        "import lacuna.linkage, lacuna.thinning, lacuna.envs.registry"
    )
    assert "httpx" not in mods


def test_the_prompt_is_one_object_wherever_it_is_imported_from():
    from lacuna.agents.llm import AGENT_SYSTEM as from_llm
    from lacuna.envs.prompts import AGENT_SYSTEM as from_prompts
    from lacuna.envs.registry import BACKOFFICE

    assert from_llm is from_prompts is BACKOFFICE.agent_system
```

- [ ] **Step 2: Run and see it fail**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_imports.py`
Expected: `test_core_imports_no_environment_and_no_agent` passes; the second fails with `'httpx' not in mods`; the third fails with `ModuleNotFoundError: No module named 'lacuna.envs.prompts'`.

- [ ] **Step 3: Move the prompt**

Create `src/lacuna/envs/prompts.py`:

```python
"""The back-office agent's system prompt.

It lives here, with no imports, so that an environment can be imported
without the HTTP client the LLM agent needs. `lacuna.agents.llm` re-exports
it; the text is pinned by sha256 in `tests/test_organism_l2.py`.
"""

AGENT_SYSTEM = (
    "You are a back-office support agent. Resolve every ticket in the queue using the "
    "tools. Look things up before changing them when unsure. When every ticket is done, "
    "reply with a one-line summary and no tool call."
)
```

In `src/lacuna/agents/llm.py`, delete the `AGENT_SYSTEM = ( ... )` block (lines 127-131) and add to the imports, after `import httpx`:

```python
from lacuna.envs.prompts import AGENT_SYSTEM
```

so the import block reads:

```python
import httpx

from lacuna.envs.prompts import AGENT_SYSTEM
from lacuna.envs.tasks import Episode
from lacuna.envs.tools import ACTION_TOOLS, TOOLS, Router
```

In `src/lacuna/envs/registry.py`, replace line 9

```python
from lacuna.agents.llm import AGENT_SYSTEM
```

with

```python
from lacuna.envs.prompts import AGENT_SYSTEM
```

and move it into sorted position among the `lacuna.envs` imports (ruff's `I` rule will say where; `uv run ruff check --fix src` does it).

- [ ] **Step 4: Run the import tests and the prompt pins**

Run: `PYTHONPATH=src uv run pytest -q tests/test_core_imports.py tests/test_organism_l2.py tests/test_bypass_organism.py tests/test_bypass_env.py`
Expected: all pass.

- [ ] **Step 5: Make httpx an extra**

In `pyproject.toml` replace

```toml
dependencies = ["numpy>=1.26", "httpx>=0.27"]
```

with

```toml
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
# The LLM agent and auditor (lacuna.agents.llm). The estimator never imports it.
agents = ["httpx>=0.27"]
```

and in `[dependency-groups]` replace the `dev` line with

```toml
dev = ["pytest>=8", "ruff>=0.5", "statsmodels>=0.14", "httpx>=0.27"]
```

Then:

```bash
uv lock && make install
```

Expected: `uv lock` rewrites `uv.lock` with httpx under the extra and the dev group; `make install` reports no package added or removed.

- [ ] **Step 6: Full gate, and the headline must not move**

Run: `make validate && make repro-headline`
Expected: all tests pass (535 before this plan, plus the new ones), `privacy-gate: clean`, and `repro-headline` ends with `reproduce (4 configs, all_passed=True)` after printing its `8 known differences` (the `DRIFT` cells in `scripts/repro_headline.py`; any ninth difference fails).

- [ ] **Step 7: Commit**

```bash
git add src/lacuna/envs/prompts.py src/lacuna/agents/llm.py src/lacuna/envs/registry.py pyproject.toml uv.lock tests/test_core_imports.py
git commit -m "Environments and estimators import without httpx; httpx becomes the agents extra"
```

---

### Task 6: spike, what a killed process and a killed collector lose at default settings

**Files:**
- Create: `rig/spikes/collector_kill/collector.yaml`, `rig/spikes/collector_kill/emit.py`, `rig/spikes/collector_kill/run.sh`, `rig/spikes/collector_kill/count.py`
- Create: `docs/RIG-FINDINGS.md`

Question: with every queue at its default, how many spans are missing from the collector's output file after (a) nothing goes wrong, (b) the emitting process is killed with SIGKILL, (c) the collector is killed and restarted mid-run? And are the missing spans a contiguous block? No test: this is a measurement, written up in `docs/RIG-FINDINGS.md`.

- [ ] **Step 1: Collector config**

`rig/spikes/collector_kill/collector.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  file:
    path: /out/spans.jsonl
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file]
```

- [ ] **Step 2: Emitter**

`rig/spikes/collector_kill/emit.py`:

```python
"""Emit N numbered spans at a fixed pace through a default BatchSpanProcessor.

Every setting of the SDK is its default. The span's number is an attribute, so
the file the collector writes says exactly which spans arrived.
"""

from __future__ import annotations

import argparse
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=600)
p.add_argument("--interval", type=float, default=0.05)
p.add_argument("--endpoint", default="http://localhost:4318/v1/traces")
a = p.parse_args()

provider = TracerProvider(resource=Resource.create({"service.name": "spike-emitter"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=a.endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("spike")

for i in range(a.n):
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("spike.seq", i)
    print(i, flush=True)
    time.sleep(a.interval)
```

- [ ] **Step 3: Counter**

`rig/spikes/collector_kill/count.py`:

```python
"""Which numbered spans reached the collector's file, and where the gaps are."""

from __future__ import annotations

import json
import sys

path, n = sys.argv[1], int(sys.argv[2])
seen: list[int] = []
with open(path) as f:
    for line in f:
        for rs in json.loads(line).get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for sp in ss.get("spans", []):
                    for attr in sp.get("attributes", []):
                        if attr["key"] == "spike.seq":
                            seen.append(int(attr["value"]["intValue"]))

got = set(seen)
missing = [i for i in range(n) if i not in got]
runs: list[tuple[int, int]] = []
for i in missing:
    if runs and runs[-1][1] == i - 1:
        runs[-1] = (runs[-1][0], i)
    else:
        runs.append((i, i))
print(
    json.dumps(
        {
            "emitted": n,
            "arrived": len(got),
            "duplicates": len(seen) - len(got),
            "missing": len(missing),
            "missing_share": round(len(missing) / n, 4),
            "missing_runs": runs,
        }
    )
)
```

- [ ] **Step 4: Runner**

`rig/spikes/collector_kill/run.sh`:

```bash
#!/usr/bin/env bash
# Usage: run.sh <condition>   where condition is one of: clean, kill-emitter, kill-collector
# Runs one condition and prints the counter's JSON line. Needs docker and uv.
set -euo pipefail
cond="${1:?condition}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out/$cond"
rm -rf "$out" && mkdir -p "$out"
image="otel/opentelemetry-collector-contrib:0.165.0"
N=600
# A venv of its own, so the emitter is a plain python process that SIGKILL reaches
# directly (`uv run` is a parent process and would leave the child running).
venv="$here/.venv"
if [ ! -x "$venv/bin/python" ]; then
  uv venv -q "$venv"
  uv pip install -q -p "$venv/bin/python" opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
fi

start_collector() {
  docker run -d --rm --name spike-otelcol -p 4318:4318 \
    -v "$here/collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
    -v "$out:/out" "$image" >/dev/null
  sleep 3
}

docker rm -f spike-otelcol >/dev/null 2>&1 || true
start_collector
"$venv/bin/python" "$here/emit.py" --n "$N" >"$out/emitted.txt" &
pid=$!
case "$cond" in
  clean) wait "$pid" ;;
  kill-emitter) sleep 17; kill -9 "$pid"; wait "$pid" 2>/dev/null || true ;;
  kill-collector)
    sleep 12; docker kill spike-otelcol >/dev/null; sleep 8; start_collector; wait "$pid" ;;
  *) echo "unknown condition $cond" >&2; exit 2 ;;
esac
sleep 8
docker stop spike-otelcol >/dev/null
emitted="$(wc -l <"$out/emitted.txt" | tr -d ' ')"
"$venv/bin/python" "$here/count.py" "$out/spans.jsonl" "$emitted" | tee "$out/result.json"
```

Add `rig/spikes/*/.venv/` and `rig/spikes/*/out/` to `.gitignore` before the first run.

- [ ] **Step 5: Run the three conditions, five times each**

```bash
chmod +x rig/spikes/collector_kill/run.sh
for c in clean kill-emitter kill-collector; do for i in 1 2 3 4 5; do rig/spikes/collector_kill/run.sh "$c"; done; done
```

Expected for `clean`: `"missing": 0`. The other two are what the spike measures. If the collector image tag does not exist, list the published tags (`docker run --rm quay.io/skopeo/stable list-tags docker://otel/opentelemetry-collector-contrib | tail`) and pin the newest by digest in `run.sh`.

- [ ] **Step 6: Write the finding**

Create `docs/RIG-FINDINGS.md` with a section "Collector and emitter kills at default settings" holding: the image digest and SDK version used (`docker inspect --format '{{index .RepoDigests 0}}' <image>`, `uv pip show opentelemetry-sdk` in the run's environment), a table of the fifteen `result.json` lines (condition, emitted, arrived, missing share, number and length of missing runs, duplicates), and three sentences answering: is default-setting loss at or above 1% under a kill; is it one contiguous block per kill; do retries produce duplicates. Add `rig/spikes/*/out/` to `.gitignore`.

- [ ] **Step 7: Commit**

```bash
git add rig/spikes/collector_kill docs/RIG-FINDINGS.md .gitignore
git commit -m "Spike: what a killed emitter and a killed collector lose at default settings"
```

---

### Task 7: spike, what agentgateway logs for one MCP tool call

**Files:**
- Create: `rig/spikes/gateway_fields/config.yaml`, `rig/spikes/gateway_fields/server.py`, `rig/spikes/gateway_fields/call.py`, `rig/spikes/gateway_fields/run.sh`
- Modify: `docs/RIG-FINDINGS.md`

Questions, each answered yes or no with the log line as evidence: does the access log for an MCP `tools/call` carry (1) the tool name, (2) the tool arguments, (3) whether the tool call failed, as distinct from the HTTP status, (4) the JSON-RPC request id, (5) a trace id that matches the caller's `traceparent`, (6) any per-request number assigned by the gateway? And can missing fields be added by configuration alone?

- [ ] **Step 1: A two-tool MCP server**

`rig/spikes/gateway_fields/server.py`:

```python
"""A tiny MCP server over streamable HTTP: one tool that succeeds, one that fails."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("spike", host="0.0.0.0", port=8000)


@mcp.tool()
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"cancelled {order_id}"


@mcp.tool()
def always_fails(order_id: str) -> str:
    """Raise, so the result is a tool error, not a transport error."""
    raise ValueError(f"cannot cancel {order_id}")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- [ ] **Step 2: A client that sends a known trace id**

`rig/spikes/gateway_fields/call.py`:

```python
"""Call both tools through the gateway with a fixed W3C traceparent."""

import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
HEADERS = {"traceparent": f"00-{TRACE_ID}-b7ad6b7169203331-01"}


async def main(url: str) -> None:
    async with streamablehttp_client(url, headers=HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ok = await session.call_tool("cancel_order", {"order_id": "O-123"})
            bad = await session.call_tool("always_fails", {"order_id": "O-456"})
            print("ok.isError =", ok.isError, "| bad.isError =", bad.isError)
            print("sent trace id", TRACE_ID)


asyncio.run(main(sys.argv[1]))
```

- [ ] **Step 3: Gateway config**

`rig/spikes/gateway_fields/config.yaml`. This is the standalone schema as documented in 2025; **before running, compare it with the current reference** at https://agentgateway.dev/docs/standalone/main/ (MCP backends, and observability > access logs) and correct field names that changed. Record any correction in the findings.

```yaml
binds:
- port: 3000
  listeners:
  - routes:
    - backends:
      - mcp:
          targets:
          - name: spike
            mcp:
              host: http://host.docker.internal:8000/mcp
```

- [ ] **Step 4: Runner**

`rig/spikes/gateway_fields/run.sh`:

```bash
#!/usr/bin/env bash
# Starts the MCP server and the gateway, makes two tool calls, saves the gateway's log.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out" && rm -rf "$out" && mkdir -p "$out"
image="ghcr.io/agentgateway/agentgateway:latest"

uv run --no-project --with mcp python "$here/server.py" >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true; docker rm -f spike-agw >/dev/null 2>&1 || true' EXIT
sleep 3
docker rm -f spike-agw >/dev/null 2>&1 || true
docker run -d --name spike-agw -p 3000:3000 \
  -v "$here/config.yaml:/config.yaml:ro" "$image" -f /config.yaml >/dev/null
sleep 3
uv run --no-project --with mcp python "$here/call.py" http://localhost:3000/mcp | tee "$out/client.txt"
sleep 2
docker logs spike-agw >"$out/gateway.log" 2>&1
docker inspect --format '{{index .RepoDigests 0}}' "$image" | tee "$out/image-digest.txt"
grep -c "" "$out/gateway.log"
```

- [ ] **Step 5: Run and read the log**

```bash
chmod +x rig/spikes/gateway_fields/run.sh && rig/spikes/gateway_fields/run.sh
grep -n "O-123\|O-456\|cancel_order\|always_fails\|0af7651916cd43dd8448eb211c80319c" rig/spikes/gateway_fields/out/gateway.log
```

Expected: `client.txt` shows `ok.isError = False | bad.isError = True`. The grep shows which of tool name, arguments and trace id appear in the default log. If the container exits at start, `docker logs spike-agw` names the config field it rejected; fix the config against the reference and rerun.

- [ ] **Step 6: Try to add what is missing by configuration**

For each of the six questions answered "no" at default settings, look for an access-log field setting in the reference (custom fields are CEL expressions over the request, response and MCP context). Add it to `config.yaml`, rerun Step 5, and keep the setting only if the field then appears. Do not patch the gateway and do not add a proxy in front of it: the study requires an unmodified component.

- [ ] **Step 7: Write the finding**

Append to `docs/RIG-FINDINGS.md` a section "What agentgateway records for one MCP tool call" with: the image digest; the six questions as a table (default: yes/no; with configuration: yes/no; the config line that did it); the two log lines for the successful and the failed call, verbatim, with any header values removed; and one paragraph on the consequence for the design. Two consequences are decided by this table and must be stated either way:
- if arguments cannot be logged, gateway records cannot be attached to env by signature, and plane A falls back to trace-id linkage for the telemetry pair with an attempted-action estimand;
- if a failed tool call cannot be told from a successful one, gateway records include phantoms, and the registration must say so.

- [ ] **Step 8: Commit**

```bash
git add rig/spikes/gateway_fields docs/RIG-FINDINGS.md
git commit -m "Spike: what agentgateway records for one MCP tool call"
```

---

### Task 8: record the decisions

**Files:**
- Modify: `docs/DECISIONS.md` (append), `docs/DESIGN-STUDY5.md`

- [ ] **Step 1: Correct the design where this plan departs from it**

In `docs/DESIGN-STUDY5.md`, in the Code table, the row for `src/lacuna/core/tables.py` says `linkage.callid_table` becomes a wrapper. Replace that row's purpose with:

```
`build` (per-plane tables) and `keyed_table`; `linkage.callid_table` is left as it is, and a test pins the two equal on three witnesses
```

- [ ] **Step 2: Append D75 to `docs/DECISIONS.md`**

Under a new `## 2026-11-15` header (or the date the work is done), add a `**D75. Study 5: design, a witness-agnostic core, and two spikes.**` entry in the log's style (dated, plain, numbers with their source). It must state: the design file and that it is not a registration; that `lacuna.core` is additive and `callid_table` was left untouched with equality pinned by `tests/test_core_tables.py`, and why (frozen analyses call it in replicate loops); that `strat_bic(live=...)` changes no estimate; that httpx is now the `agents` extra and `make repro-headline` was identical before and after; the two spike findings with their numbers from `docs/RIG-FINDINGS.md`; and that the PyPI name `lacuna` is taken (checked 2026-11-15: `lacuna-mse`, `lacuna-audit` and `agent-lacuna` were free), so the distribution name is an open decision while the import name stays `lacuna`.

- [ ] **Step 3: Gate and commit**

```bash
make validate
git add docs/DECISIONS.md docs/DESIGN-STUDY5.md
git commit -m "D75: study 5 design, the core package, and the two rig spikes"
```

Expected: `make validate` green.

---

## What changed while executing (2026-11-15)

The tasks above are the plan as written. Where the work departed from it:

- **Task 4.** The undeclared-witness message is `stratum 'a': witnesses [2] are not declared`
  (shortened to fit the line limit; the test's `match="not declared"` is unchanged).
- **Task 6.** `run.sh` takes a run tag and `N` / `OUTAGE` from the environment, and moves
  the collector's file aside before a restart. The file exporter overwrites its file on
  start, so the first five collector-kill runs reported about 30% missing, all of it
  spans delivered before the kill. `count.py` takes the count first and any number of
  files. Three runs with a 25-second outage were added.
- **Task 7.** The config in Step 3 is the 2025 schema and does not load. The current
  schema (`mcp:` at the top level, `frontendPolicies.accessLog.add`) is in
  `rig/spikes/gateway_fields/config-default.yaml` and `config-fields.yaml`; the image is
  `cr.agentgateway.dev/agentgateway:v1.7.0`; `run.sh` takes `default` or `fields` and
  copies the config before mounting it, because the gateway writes a session key into it.
  The spike pins `mcp<2`: 2.x renamed `FastMCP` and `streamablehttp_client`.

## Done when

- `make validate` and `make repro-headline` are green, the second with only its eight known differences, as before this plan.
- `python -c "import lacuna.core.tables, lacuna.envs.registry, sys; assert 'httpx' not in sys.modules"` exits 0 with `PYTHONPATH=src`.
- `docs/RIG-FINDINGS.md` answers: default-setting loss under a kill (share, contiguity, duplicates), and the six gateway-log questions.
- D75 is in the decision log.

The next plan (kernel service, MCP fronts, replay driver, compose rig) is written after reading `docs/RIG-FINDINGS.md`, because both spikes can change it.
