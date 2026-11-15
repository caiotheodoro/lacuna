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


def _outcomes(c) -> int:
    skip = ("read", "unreadable_lines", "duplicates")
    return sum(v for k, v in vars(c).items() if k not in skip)


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
        "unreadable_lines": 1, "duplicates": 0,
        "read": 7, "not_selected": 1, "plane_not_mapped": 0, "failed": 2, "no_session": 1,
        "bad_arguments": 0, "not_an_action": 2, "kept": 1,
    }  # fmt: skip
    assert c.read == _outcomes(c) and c.read + c.unreadable_lines == len(lines)


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
    assert errors == ["plane p: 1 of the map's witnesses can see it (['log']); needs two"]
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


# What the review of this module found ---------------------------------------------------


def test_lines_that_are_not_records_and_second_copies_of_a_span_are_counted():
    lines = [line(), "", "  ", "[1, 2]", '"text"', line(n=2)[:20], line(n=3)]
    obs, c = read(lines)
    assert (c.kept, c.read, c.unreadable_lines) == (2, 2, 3)  # blank lines are not lines
    span = {"traceId": "t1", "spanId": "s1", "attributes": []}
    bare = {"attributes": []}  # no ids: cannot be a copy of anything
    req = {"resourceSpans": [{"scopeSpans": [{"spans": [span, span, bare, bare, bare]}]}]}
    text = SIMPLE.replace('format = "jsonl"\npipeline', 'format = "otlp-spans"\npipeline')
    obs, c = read([json.dumps(req), "{cut"], text)
    assert (c.read, c.duplicates, c.unreadable_lines) == (4, 1, 1)


def test_a_file_in_another_format_reads_nothing_and_says_so():
    text = SIMPLE.replace('format = "jsonl"\npipeline', 'format = "otlp-spans"\npipeline')
    obs, c = read([line(), line(n=2)], text)
    assert (obs, c.read, c.unreadable_lines) == ([], 0, 0)  # objects, but no span in them
    with pytest.raises(MapError, match="source 'log' is not otlp-spans"):
        read([json.dumps({"resourceSpans": ["x"]})], text)


def test_a_number_equals_an_equal_number_and_true_is_not_one():
    text = SIMPLE.replace("equals = true", "equals = 200")
    assert read([line(ok=200)], text)[1].kept == 1
    assert read([line(ok=200.0)], text)[1].kept == 1  # a float status is still 200
    assert read([line(ok=True)], text)[1].failed == 1
    assert read([line(ok="200")], text)[1].failed == 1


def test_a_session_an_id_part_or_an_action_that_is_not_text_or_a_number_is_not_usable():
    for bad in (["x"], {"a": 1}, None, True, ""):
        assert read([line(sid=bad)])[1].no_session == 1, bad
    no_key = json.loads(line())
    del no_key["sid"]
    assert read([json.dumps(no_key)])[1].no_session == 1  # the path leads nowhere
    assert read([line(sid=0)])[0][0].session == 0  # zero is a session's name like any other
    for bad in ([1], {"a": 1}, None, "", True):
        (o,), c = read([line(n=bad)])
        assert o.link_id is None and c.kept == 1, bad
    assert read([line(n=2.5)])[0][0].link_id == ("s1", 2.5)
    for field in ("op", "amount"):
        assert read([line(**{field: ["pay"]})])[1].not_an_action == 1
    assert read([line(target={"id": {"deep": 1}})])[1].not_an_action == 1


def test_arguments_an_envs_own_normaliser_cannot_read_are_bad_arguments():
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
[[witness]]
name = "state"
format = "jsonl"
plane = "p"
session = "sid"
op = "op"
resource = "res"
value = "val"
"""
    # A whole number too long to be a float: the env's normaliser raises on it.
    args = '{"order_id": "O1", "amount": 1' + "0" * 400 + "}"
    obs, c = read(['{"sid": "s", "n": 2, "tool": "issue_refund", "args": ' + args + "}"], text)
    assert (obs, c.bad_arguments) == ([], 1)
    # A block that reads ops directly takes the env's own actions by default.
    row = {"sid": "s", "op": "issue_refund", "res": "O1", "val": "5.00"}
    assert (
        read([json.dumps(row), json.dumps(row | {"op": "get_order"})], text, block=1)[1].kept == 1
    )


def test_calls_are_ordered_by_number_then_by_text_and_a_call_with_no_number_comes_last():
    m = parse(SIMPLE)
    log = [line(n="b", amount="1"), line(n=10, amount="2"), line(n=None, amount="3"),
           line(n=2, amount="4"), line(n="a", amount="5"), line(n=0, amount="6")]  # fmt: skip
    obs, _ = mapfile.records(m, {"log": log, "state": []})
    assert [o.value for o in obs] == ["6", "4", "2", "3", "5", "1"]
    assert [o.seq for o in obs] == [0, 2, 10, None, "a", "b"]


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("version = 1", "version = true", "version = 1"),
        ("version = 1", "version = 1.0", "version = 1"),
        ('pipeline = "shipper"', "pipeline = 5", "pipeline: a non-empty string"),
        ('pipeline = "shipper"', "pipeline = [1]", "pipeline: a non-empty string"),
        ('pipeline = "shipper"', "source = [1]", "source: a non-empty string"),
        ('pipeline = "shipper"', 'has = "abc"', "has: a list of paths"),
        ('pipeline = "shipper"', "has = 5", "has: a list of paths"),
        ('pipeline = "shipper"', "has = { a = 1 }", "has: a list of paths"),
        ('id = ["sid", "n"]', 'id = "sid"', "id: a list of paths"),
        ('map = { red = "p" }', "map = { red = 1 }", "non-empty table of plane names"),
        ('map = { red = "p" }', "map = {}", "non-empty table of plane names"),
        ('pipeline = "shipper"', 'body = "json"', "for otlp-logs"),
        ('actions = ["pay", "mail"]', 'actions = ["pay"]\nenv = ["x"]', "is not one of"),
        ('planes = ["p"]', 'planes = [""]', "non-empty list of names"),
    ],
)  # fmt: skip
def test_a_value_of_the_wrong_kind_is_refused_by_name_and_never_a_crash(old, new, message):
    assert old in SIMPLE
    with pytest.raises(MapError, match=message):
        parse(SIMPLE.replace(old, new, 1))


def test_a_witness_entry_that_is_not_a_table_is_refused():
    for bad in ("witness = [1]", 'witness = ["x"]'):
        with pytest.raises(MapError, match=r"witness\[0\]: a table"):
            parse(f'version = 1\nplanes = ["p"]\n{bad}')


def test_a_map_no_table_can_be_built_from_is_refused_before_any_file_is_read():
    one = parse(SIMPLE.split('[[witness]]\nname = "state"')[0])
    with pytest.raises(MapError, match="plane p: 1 of the map's witnesses can see it"):
        mapfile.records(one, {"log": []})
    unseen = parse(SIMPLE.replace('planes = ["p"]', 'planes = ["p", "q"]'))
    with pytest.raises(MapError, match="plane q: 0 of the map's witnesses"):
        mapfile.records(unseen, {"log": [], "state": []})
    extra = "".join(
        f'[[witness]]\nname = "w{i}"\nformat = "jsonl"\nplane = "p"\n'
        'op = "op"\nresource = "res"\nvalue = "val"\n' for i in range(4)
    )  # fmt: skip
    crowded = parse(SIMPLE + extra)
    with pytest.raises(MapError, match="6 witnesses; more than 5 is too many"):
        mapfile.records(crowded, {})
    assert "more than 5 is too many" in " ".join(mapfile.lint(crowded)[0])


def test_lint_sees_a_row_two_blocks_would_both_read_and_witnesses_read_from_one_file():
    state = SIMPLE[SIMPLE.index('[[witness]]\nname = "state"') :]
    errors, _ = mapfile.lint(parse(SIMPLE + state.replace("{ red", "{ blue")))
    assert errors == [
        "plane p: witness state has no id and two of its blocks read state for this plane, "
        "so a row both select is counted twice"
    ]
    # Two blocks of an id witness are one witness: its id still links nothing.
    log = SIMPLE[SIMPLE.index("[[witness]]") : SIMPLE.index('[[witness]]\nname = "state"')]
    _, warnings = mapfile.lint(parse(SIMPLE + log))
    assert "plane p: one witness has an id, so nothing is linked by it" in warnings
    same_file = SIMPLE.replace('name = "state"\nformat = "jsonl"',
                               'name = "state"\nsource = "log"\nformat = "jsonl"')  # fmt: skip
    _, warnings = mapfile.lint(parse(same_file))
    assert any("['log', 'state'] are read from one file (log)" in w for w in warnings)
    assert not any("read from one file" in w for w in mapfile.lint(parse(SIMPLE))[1])


# The rig's map against the rig's code, on records that are not clean -------------------


def _attr(span: dict, key: str) -> dict | None:
    return next((a for a in span["attributes"] if a["key"] == key), None)


def _awkward(run: Path) -> None:
    """Edit a copy of the fixture so that it holds what a clean run does not: failed
    calls, records with no session, a batch delivered twice, a service that is no
    originator, and receiver lines that are not successes."""
    w = run / "witnesses"
    spans = [json.loads(x) for x in (w / "spans.jsonl").read_text().splitlines()]
    actions = [
        sp
        for req in spans
        for rs in req["resourceSpans"]
        for ss in rs["scopeSpans"]
        for sp in ss["spans"]
        if (t := _attr(sp, "gen_ai.tool.name"))
        and t["value"]["stringValue"] in BACKOFFICE.action_tools
    ]  # fmt: skip
    assert len(actions) >= 6
    _attr(actions[0], "lacuna.tool.is_error")["value"] = {"boolValue": True}
    actions[1]["attributes"].remove(_attr(actions[1], "lacuna.session"))
    actions[2]["attributes"].remove(_attr(actions[2], "lacuna.tool.is_error"))
    stranger = json.loads(json.dumps(spans[0]))
    for rs in stranger["resourceSpans"]:
        for a in rs["resource"]["attributes"]:
            if a["key"] == "service.name":
                a["value"] = {"stringValue": "telemetrygen"}
        for ss in rs["scopeSpans"]:
            for sp in ss["spans"]:
                sp["spanId"] = sp["spanId"][::-1]
    out = [json.dumps(x) for x in (*spans, spans[-1], stranger)]  # the last batch twice
    (w / "spans.jsonl").write_text("\n".join(out) + "\n")

    logs = [json.loads(x) for x in (w / "logs.jsonl").read_text().splitlines()]
    seen = {"gateway.jsonl": 0, "front-b.jsonl": 0}
    for req in logs:
        for rl in req["resourceLogs"]:
            for sl in rl["scopeLogs"]:
                for rec in sl["logRecords"]:
                    name = next(a["value"]["stringValue"] for a in rec["attributes"]
                                if a["key"] == "log.file.name")  # fmt: skip
                    body = json.loads(rec["body"]["stringValue"])
                    tool = body.get("gen_ai.tool.name") or body.get("tool")
                    if name not in seen or tool not in BACKOFFICE.action_tools:
                        continue
                    seen[name] += 1
                    if name == "gateway.jsonl" and seen[name] == 1:
                        body["http.status"] = 500
                    elif name == "gateway.jsonl" and seen[name] == 2:
                        body["session"] = None
                    elif name == "front-b.jsonl" and seen[name] == 1:
                        body["ok"] = False
                    elif name == "front-b.jsonl" and seen[name] == 2:
                        del body["session"]
                    rec["body"]["stringValue"] = json.dumps(body)
    assert min(seen.values()) >= 2
    (w / "logs.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n{cut\n")


def test_the_two_agree_on_failed_calls_missing_sessions_copies_and_strangers(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    _awkward(run)
    m = mapfile.load(RIG_MAP)
    obs, report = mapfile.records(m, _sources(run))
    got = {p: (n, [int(x) for x in t]) for p, (n, t) in build(obs, m.specs(), mapfile.key).items()}
    assert got == _rig_tables(run)
    spans, gateway, front, _ = report
    # Every kind of record the clean fixture lacks was met, and counted where it belongs.
    assert spans["failed"] == 2 and spans["no_session"] == 1 and spans["duplicates"] > 0
    assert spans["plane_not_mapped"] > 0
    assert (gateway["failed"], gateway["no_session"]) == (1, 1)
    assert (front["failed"], front["no_session"]) == (1, 1)
    assert gateway["unreadable_lines"] == front["unreadable_lines"] == 1
    # And the tables are no longer the clean ones: some actions are now seen by fewer.
    assert got != _rig_tables(FIXTURE)


def test_where_the_map_and_the_rigs_code_part_ways_is_on_rows_no_run_writes(tmp_path):
    # Pinned so the difference is a known one. The rig's code takes every row of the
    # end state as it is; the map holds a row to a session, an action and a plane.
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    env = run / "witnesses" / "env.jsonl"
    odd = {
        "session": "s0",
        "op": "not_an_op",
        "resource": "r",
        "value": "v",
        "principal": "plane-a",
    }
    env.write_text(env.read_text() + json.dumps(odd) + "\n")
    m = mapfile.load(RIG_MAP)
    obs, report = mapfile.records(m, _sources(run))
    mine = build(obs, m.specs(), mapfile.key)
    assert report[3]["not_an_action"] == 1
    assert int(_rig_tables(run)["a"][1][4]) == int(mine["a"][1][4]) + 1  # env-only cell
