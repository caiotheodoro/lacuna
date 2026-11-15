"""A map file turns log files into witness records; the rig's own map is held to the rig's code."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lacuna import mapfile
from lacuna.core.tables import build
from lacuna.envs.registry import BACKOFFICE
from lacuna.mapfile import MapError, parse
from lacuna.rig import witnesses as rw

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "rig" / "run"
RIG_MAP = ROOT / "rig" / "map.toml"

SIMPLE = """
version = 1
planes = ["p"]
actions = ["pay", "mail"]

[[witness]]
name = "log"
format = "jsonl"
pipeline = "shipper"
plane = "p"
select = [{ path = "kind", equals = "call" }]
ok = [{ path = "ok", equals = true }]
session = "sid"
op = "op"
resource = ["target", "id"]
value = "amount"
id = ["sid", "n"]
seq = "n"

[[witness]]
name = "state"
format = "jsonl"
plane = { path = "team", map = { red = "p" } }
session = "sid"
op = "op"
resource = "res"
value = "val"
"""


def line(**kw) -> str:
    base = {"kind": "call", "ok": True, "sid": "s1", "n": 1, "op": "pay",
            "target": {"id": "o1"}, "amount": "5"}  # fmt: skip
    return json.dumps(base | kw)


def read(lines: list[str], text: str = SIMPLE, block: int = 0):
    m = parse(text)
    return mapfile.read_block(m, m.blocks[block], lines)


# Parsing ---------------------------------------------------------------------------


def test_a_map_is_planes_witnesses_and_where_each_field_is():
    m = parse(SIMPLE)
    assert m.planes == ("p",) and m.env is None and m.actions == {"pay", "mail"}
    log, state = m.blocks
    assert (log.name, log.source, log.format, log.pipeline) == ("log", "log", "jsonl", "shipper")
    assert log.resource == ("target", "id") and log.id == (("sid",), ("n",)) and log.by_id
    assert state.plane is None and state.plane_map == {"red": "p"} and not state.by_id
    specs = m.specs()
    assert [(s.name, sorted(s.sees), s.by_id) for s in specs] == [
        ("log", ["p"], True),
        ("state", ["p"], False),
    ]
    assert m.sources() == ("log", "state")


@pytest.mark.parametrize(
    "change, message",
    [
        (("version = 1", "version = 2"), "version = 1"),
        (('planes = ["p"]', "planes = []"), "non-empty list"),
        (('planes = ["p"]', 'planes = ["p", "p"]'), "named twice"),
        (('format = "jsonl"\npipeline', 'format = "csv"\npipeline'), "format must be one of"),
        (('plane = "p"', 'plane = "q"'), "['q'] are not in `planes`"),
        (('plane = "p"', "plane = 3"), "plane is a plane's name"),
        (('op = "op"\nresource = ["target", "id"]', 'resource = ["target", "id"]'), "give `tool`"),
        (('value = "amount"', ""), "`op` needs `resource` and `value`"),
        (('session = "sid"\nop = "op"\nresource = ["t', 'when = "x"\nop = "op"\nresource = ["t'),
         "unknown keys ['when']"),
        (('ok = [{ path = "ok", equals = true }]', 'ok = [{ path = "ok" }]'), "exactly `path`"),
        (('actions = ["pay", "mail"]', ""), "need the map's `actions`"),
        (('actions = ["pay", "mail"]', 'actions = ["pay"]\nextra = 1'), "unknown top-level"),
        (('id = ["sid", "n"]', "id = [[]]"), "a path is a key"),
        (('actions = ["pay", "mail"]', 'actions = ["pay"]\nenv = "nope"'), "is not one of"),
    ],
)  # fmt: skip
def test_a_map_that_cannot_be_followed_is_refused_with_where(change, message):
    old, new = change
    assert old in SIMPLE
    with pytest.raises(MapError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse(SIMPLE.replace(old, new, 1))


def test_not_toml_no_witness_and_mixed_ids_are_refused():
    with pytest.raises(MapError, match="not TOML"):
        parse("version = ")
    with pytest.raises(MapError, match="no .*witness.* block"):
        parse('version = 1\nplanes = ["p"]')
    second = SIMPLE.replace('name = "state"', 'name = "log"')
    with pytest.raises(MapError, match="some blocks have an id and some do not"):
        parse(second)


def test_tool_and_arguments_need_an_env_to_read_them():
    text = SIMPLE.replace('op = "op"\nresource = ["target", "id"]\nvalue = "amount"',
                          'tool = "op"\nargs = "target"')  # fmt: skip
    with pytest.raises(MapError, match="needs `args`, and the map needs `env`"):
        parse(text)
    m = parse(text.replace("version = 1", 'version = 1\nenv = "backoffice"'))
    assert m.env == "backoffice" and m.blocks[0].tool == ("op",)


# Reading -----------------------------------------------------------------------------


def test_a_record_becomes_an_action_with_its_plane_id_and_session():
    obs, c = read([line()])
    (o,) = obs
    assert (o.witness, o.op, o.resource, o.value) == ("log", "pay", "o1", "5")
    assert (o.stratum, o.link_id, o.seq, o.session) == ("p", ("s1", 1), 1, "s1")
    assert (c.read, c.kept) == (1, 1)
    assert mapfile.key(o) == ("s1", "pay", "o1", "5")


def test_every_record_read_is_counted_under_what_happened_to_it():
    lines = [
        line(),
        line(kind="other"),  # not this witness's record
        line(ok=False),  # the call failed
        line(ok=1),  # 1 is not true: a status code is not a success flag
        line(sid=""),  # no session
        line(op="read"),  # not an action
        "not json",
        json.dumps({"kind": "call", "ok": True, "sid": "s1", "n": 2, "op": "pay"}),  # no target
    ]
    obs, c = read(lines)
    assert len(obs) == 1
    assert vars(c) == {
        "read": 7, "not_selected": 1, "plane_not_mapped": 0, "failed": 2, "no_session": 1,
        "bad_arguments": 0, "not_an_action": 2, "kept": 1,
    }  # fmt: skip
    assert c.read == sum(v for k, v in vars(c).items() if k != "read")


def test_a_plane_read_from_the_record_goes_through_the_maps_table():
    row = {"sid": "s1", "op": "pay", "res": "o1", "val": "5"}
    lines = [json.dumps(row | {"team": "red"}), json.dumps(row | {"team": "blue"}), json.dumps(row)]
    obs, c = read(lines, block=1)
    assert [(o.stratum, o.link_id) for o in obs] == [("p", None)]
    assert (c.kept, c.plane_not_mapped) == (1, 2)


def test_has_requires_a_field_and_an_id_with_a_part_missing_links_nothing():
    text = SIMPLE.replace('select = [{ path = "kind", equals = "call" }]', 'has = ["n"]')
    obs, c = read([line(), json.dumps({"sid": "s1", "op": "pay"})], text)
    assert (c.kept, c.not_selected) == (1, 1)
    obs, _ = read([line(n=None)])
    assert obs[0].link_id is None and obs[0].seq is None  # kept, and its own action


def test_with_an_env_the_action_is_what_the_env_makes_of_tool_and_arguments():
    text = """
version = 1
planes = ["p"]
env = "backoffice"
[[witness]]
name = "log"
format = "jsonl"
plane = "p"
session = "sid"
tool = "tool"
args = "args"
id = ["n"]
"""
    refund = {"order_id": "O1", "amount": 5}
    lines = [
        json.dumps({"sid": "s", "n": 1, "tool": "issue_refund", "args": refund}),
        json.dumps({"sid": "s", "n": 2, "tool": "issue_refund", "args": json.dumps(refund)}),
        json.dumps({"sid": "s", "n": 3, "tool": "get_order", "args": {"order_id": "O1"}}),
        json.dumps({"sid": "s", "n": 4, "tool": "issue_refund", "args": [1]}),
        json.dumps({"sid": "s", "n": 5, "tool": "issue_refund", "args": "{not json"}),
        json.dumps({"sid": "s", "n": 6, "tool": "issue_refund", "args": {"order_id": "O1"}}),
    ]
    obs, c = read(lines, text)
    assert [(o.op, o.resource, o.value) for o in obs] == [("issue_refund", "O1", "5.00")] * 2
    assert (c.kept, c.not_an_action, c.bad_arguments) == (2, 1, 3)


def test_records_come_in_call_order_then_the_ones_with_no_id():
    m = parse(SIMPLE)
    log = [line(n=3, amount="3"), line(n=1, amount="1"), line(sid="s0", n=9, amount="9")]
    state = [json.dumps({"team": "red", "sid": "s1", "op": "pay", "res": "o1", "val": "1"})]
    obs, report = mapfile.records(m, {"log": log, "state": state})
    assert [(o.witness, o.session, o.seq) for o in obs] == [
        ("log", "s0", 9), ("log", "s1", 1), ("log", "s1", 3), ("state", "s1", None),
    ]  # fmt: skip
    assert [(r["witness"], r["block"], r["source"], r["kept"]) for r in report] == [
        ("log", 0, "log", 3), ("state", 1, "state", 1),
    ]  # fmt: skip
    with pytest.raises(MapError, match=r"no input for sources \['state'\]"):
        mapfile.records(m, {"log": log})


# The rig's own map -------------------------------------------------------------------


def _tables(sources: dict[str, list[str]]) -> dict:
    m = mapfile.load(RIG_MAP)
    obs, _ = mapfile.records(m, sources)
    return {p: (n, [int(x) for x in t]) for p, (n, t) in build(obs, m.specs(), mapfile.key).items()}


def _rig_tables(run: Path) -> dict:
    d = rw.load(run, BACKOFFICE)
    built = build(d["obs"], rw.SPECS, rw.key)
    return {p: (n, [int(x) for x in t]) for p, (n, t) in built.items()}


def _sources(run: Path) -> dict[str, list[str]]:
    w = run / "witnesses"
    return {n: (w / f"{n}.jsonl").read_text().splitlines() for n in ("spans", "logs", "env")}


def test_the_rigs_map_gives_the_tables_the_rigs_code_gives():
    m = mapfile.load(RIG_MAP)
    assert [(s.name, sorted(s.sees), s.by_id) for s in m.specs()] == [
        (s.name, sorted(s.sees), s.by_id) for s in rw.SPECS
    ]
    got = _tables(_sources(FIXTURE))
    assert got == _rig_tables(FIXTURE)
    assert sum(sum(t) for _, t in got.values()) > 10


def test_and_still_does_when_each_log_has_lost_something_different(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    w = run / "witnesses"
    spans = (w / "spans.jsonl").read_text().splitlines()
    logs = (w / "logs.jsonl").read_text().splitlines()
    env = (w / "env.jsonl").read_text().splitlines()
    assert len(spans) > 1 and len(logs) > 3 and len(env) > 4
    (w / "spans.jsonl").write_text("\n".join(spans[:-1]) + "\n")  # the last batch
    (w / "logs.jsonl").write_text("\n".join(logs[2:]) + "\n")  # the first lines
    (w / "env.jsonl").write_text("\n".join(env[::2]) + "\n")  # every other row
    got = _tables(_sources(run))
    assert got == _rig_tables(run)
    # The three losses differ, so more than one cell of some table is filled.
    assert max(sum(1 for x in t if x) for _, t in got.values()) >= 3


# Lint ----------------------------------------------------------------------------------


def test_the_rigs_map_can_be_audited_and_lint_says_what_it_cannot_see():
    errors, warnings = mapfile.lint(mapfile.load(RIG_MAP))
    assert errors == [] and warnings == []


def test_a_plane_with_one_recorder_or_one_pipeline_is_an_error():
    one = parse(SIMPLE.split('[[witness]]\nname = "state"')[0])
    errors, _ = mapfile.lint(one)
    assert errors == ["plane p: 1 witness can see it (['log']); needs two"]
    head = 'name = "state"\nformat = "jsonl"'
    shared = parse(SIMPLE.replace(head, head + '\npipeline = "shipper"'))
    errors, _ = mapfile.lint(shared)
    assert errors == [
        "plane p: its witnesses ['log', 'state'] all arrive by one pipeline (shipper)"
    ]
    assert mapfile.lint(parse(SIMPLE))[0] == []


def test_lint_warns_about_what_makes_an_estimate_too_low_or_too_high():
    _, warnings = mapfile.lint(parse(SIMPLE))
    text = " | ".join(warnings)
    assert "one witness has an id, so nothing is linked by it" in text
    assert "witness[1] (state): no `pipeline` named" in text
    assert "no `ok` test" not in text and "no `session`" not in text
    bare = SIMPLE.replace('ok = [{ path = "ok", equals = true }]\n', "").replace(
        'session = "sid"\nop = "op"\nresource = ["target"', 'op = "op"\nresource = ["target"'
    )
    text = " | ".join(mapfile.lint(parse(bare))[1])
    assert "witness[0] (log): no `ok` test, so a failed call counts as an action" in text
    assert "witness[0] (log): no `session`" in text
    state_op = 'op = "op"\nresource = "res"'
    two_logs = SIMPLE.replace(state_op, 'id = ["sid"]\n' + state_op)
    assert "none is a record of the" in " | ".join(mapfile.lint(parse(two_logs))[1])
