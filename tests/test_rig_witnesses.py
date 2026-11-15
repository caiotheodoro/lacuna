"""Ingest and the run summary, on a trimmed real run of the rig (tests/fixtures/rig/run).

The fixture is unstressed, so every pipeline delivered everything. The tests that
need loss remove lines from a copy.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from lacuna.core.records import Obs
from lacuna.envs.registry import BACKOFFICE
from lacuna.ingest import jsonl, otlp
from lacuna.rig.witnesses import (
    ACCESS,
    ENV,
    LOG_PLANE,
    LOG_SERVICE,
    SPANS,
    front_calls,
    gateway_calls,
    key,
    load,
    span_calls,
    summarise,
)

RUN = Path(__file__).parent / "fixtures" / "rig" / "run"


def _lines(rel: str) -> list[str]:
    return (RUN / rel).read_text().splitlines()


def _copy(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(RUN, run)
    return run


# Readers --------------------------------------------------------------------


def test_spans_are_read_once_each_with_typed_attributes():
    spans = otlp.read_spans(_lines("witnesses/spans.jsonl"))
    tools = [s for s in spans if s["name"].startswith("execute_tool")]
    assert len(spans) == len(tools) + 4  # one root span per plane's share of a session
    assert {s["resource"]["service.name"] for s in spans} == {"agent-a", "agent-b"}
    a = tools[0]["attributes"]
    assert isinstance(a["jsonrpc.request.id"], int) and a["lacuna.tool.is_error"] is False
    assert len(tools[0]["trace_id"]) == 32 and tools[0]["end_ns"] >= tools[0]["start_ns"] > 0
    # A retried export delivers a batch twice; the copy is dropped.
    twice = otlp.read_spans(_lines("witnesses/spans.jsonl") * 2)
    assert twice == spans


def test_a_line_cut_short_is_skipped_by_both_readers():
    lines = _lines("witnesses/spans.jsonl")
    cut = lines[:-1] + [lines[-1][: len(lines[-1]) // 2], "", "not json"]
    assert len(otlp.read_spans(cut)) < len(otlp.read_spans(lines))
    assert otlp.read_spans(["", "[]", "garbage"]) == []
    assert jsonl.read(['{"a": 1}', "", "starting up", '{"a": 2', "[1]", ' {"b": 2} ']) == [
        {"a": 1},
        {"b": 2},
    ]


def test_delivered_log_records_name_their_file_and_carry_the_line():
    logs = otlp.read_logs(_lines("witnesses/logs.jsonl"))
    assert {r["attributes"]["log.file.name"] for r in logs} == {"gateway.jsonl", "front-b.jsonl"}
    assert all(isinstance(r["body"], str) and r["observed_ns"] > 0 for r in logs)
    front = [r["body"] for r in logs if r["attributes"]["log.file.name"] == "front-b.jsonl"]
    assert front == [ln for ln in _lines("logs/front-b.jsonl") if ln.startswith("{")]


# What counts as an action -----------------------------------------------------


def _gw(**over) -> dict:
    base = {"scope": "request", "mcp.method.name": "tools/call", "http.status": 200,
            "gen_ai.tool.name": "cancel_order", "tool_args": {"order_id": "O1"},
            "tool_is_error": False, "session": "s0", "trace.id": "t" * 32, "rpc_id": 4}  # fmt: skip
    return {**base, **over}


def test_a_gateway_line_is_ok_only_if_the_call_and_the_tool_both_succeeded():
    assert gateway_calls([_gw()])[0] == {
        "tool": "cancel_order", "args": {"order_id": "O1"}, "ok": True, "session": "s0",
        "trace_id": "t" * 32, "rpc_id": 4, "ts": None,
    }  # fmt: skip
    timed = gateway_calls([_gw(time="2026-11-15T08:52:43.573982Z")])[0]
    assert timed["ts"] == pytest.approx(1794732763.573982)
    assert gateway_calls([_gw(time="not a time")])[0]["ts"] is None
    assert gateway_calls([_gw(tool_is_error=True)])[0]["ok"] is False
    assert gateway_calls([_gw(**{"http.status": 502})])[0]["ok"] is False
    # As shipped the gateway logs no failure flag; without it a line is not counted.
    line = _gw()
    del line["tool_is_error"]
    assert gateway_calls([line])[0]["ok"] is False
    # Other requests of the session and the gateway's own messages are not tool calls.
    assert gateway_calls([_gw(**{"mcp.method.name": "tools/list"}), _gw(scope="app")]) == []


def test_front_lines_and_span_attributes_read_the_same_way():
    line = {"tool": "cancel_order", "args": {"order_id": "O1"}, "ok": True, "session": "s0",
            "trace_id": "t" * 32, "rpc_id": 4, "ts": 1.0}  # fmt: skip
    assert front_calls([line, {**line, "ok": False}, {"message": "started"}]) == [
        line,
        {**line, "ok": False},
    ]
    span = {
        "trace_id": "t" * 32, "span_id": "s" * 16, "start_ns": 2_000_000_000,
        "resource": {"service.name": "agent-b"},
        "attributes": {"mcp.method.name": "tools/call", "gen_ai.tool.name": "cancel_order",
                       "gen_ai.tool.call.arguments": '{"order_id": "O1"}',
                       "lacuna.session": "s0", "jsonrpc.request.id": 4,
                       "lacuna.tool.is_error": False},
    }  # fmt: skip
    ((plane, call),) = span_calls([span])
    assert plane == "b" and call == {**line, "ts": 2.0}
    # A span whose call never returned has no flag, and is not an action.
    unfinished = {**span, "attributes": {k: v for k, v in span["attributes"].items()
                                         if k != "lacuna.tool.is_error"}}  # fmt: skip
    assert span_calls([unfinished])[0][1]["ok"] is False
    # A span from a service that is not an originator of this rig is ignored.
    assert span_calls([{**span, "resource": {"service.name": "telemetrygen"}}]) == []


def test_the_key_is_a_signature_inside_a_session():
    a = Obs(ENV, "update_customer", "C1.tier", "gold", session="s0")
    assert key(a) == ("s0", "update_customer", "C1.tier", "gold")
    assert key(a) != key(Obs(ENV, "update_customer", "C1.tier", "gold", session="s1"))
    # A file's content is not part of what identifies a write (lacuna.linkage.VALUE_OPS).
    assert key(Obs(ENV, "write_file", "/x", "abc", session="s0")) == ("s0", "write_file", "/x", "")


# The run --------------------------------------------------------------------


def test_records_are_actions_only_and_every_one_has_plane_session_and_link():
    d = load(RUN)
    assert {o.witness for o in d["obs"]} == {SPANS, ACCESS, ENV}
    for o in d["obs"]:
        assert o.op in BACKOFFICE.action_tools and o.stratum in ("a", "b") and o.session
        assert (o.link_id is None) == (o.witness == ENV)
    # Reads are in the logs and are not records.
    reads = [c for _, c in d["span_calls"] if c["tool"] not in BACKOFFICE.action_tools]
    assert reads and len([o for o in d["obs"] if o.witness == SPANS]) < len(d["span_calls"])


def test_the_unstressed_run_links_fully_and_matches_the_ledger():
    s = summarise(RUN)
    ledger = [json.loads(ln) for ln in _lines("truth/ledger.jsonl")]
    per_plane = Counter(e["principal"] for e in ledger)
    assert s["headline"]["actions_in_ledger"] == len(ledger) == 16
    for plane, p in s["planes"].items():
        assert p["actions_in_ledger"] == per_plane[f"plane-{plane}"] > 0
        assert p["witness_names"] == [SPANS, ACCESS, ENV]
        link = p["link"]
        assert link["agreement"] == link["coverage"] == 1.0
        assert link["pairs"] == link["pairs_same_key"] == p["actions_in_ledger"]
        assert p["records_without_session"] == 0
        assert p["spans_delivered"] == p["access_lines_delivered"] == p["calls_written_by_receiver"]
        for w in (SPANS, ACCESS):
            assert p["witnesses"][w]["actions_seen"] == p["actions_in_ledger"]
            assert p["witnesses"][w]["matching_no_action"] == 0
        assert p["witnesses"][ENV]["matching_no_action"] == 0
        assert sum(p["table"]) == p["union_count"] == p["actions_in_ledger"]
        assert p["table"][0] == 0
        # The floors come from the table, never from the ledger (AGENTS.md rules 3 and 5).
        spans_bit = 1 << p["witness_names"].index(SPANS)
        assert p["trust_trace"] == sum(n for i, n in enumerate(p["table"]) if i & spans_bit)
        assert p["union_bound"] == {"at_least": p["union_count"], "at_most": None}


def test_lost_spans_move_actions_out_of_the_spans_column(tmp_path):
    run = _copy(tmp_path)
    path = run / "witnesses" / "spans.jsonl"
    kept = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        for rs in d["resourceSpans"]:
            if any(a["value"].get("stringValue") == "agent-b"
                   for a in rs["resource"]["attributes"]):  # fmt: skip
                for ss in rs["scopeSpans"]:
                    ss["spans"] = ss["spans"][: len(ss["spans"]) // 2]
        kept.append(json.dumps(d))
    path.write_text("\n".join(kept) + "\n")
    before, after = summarise(RUN)["planes"], summarise(run)["planes"]
    assert after["a"] == before["a"]
    b = after["b"]
    assert 0 < b["witnesses"][SPANS]["actions_seen"] < b["actions_in_ledger"]
    assert b["witnesses"][SPANS]["missed_share"] > 0
    # Loss lowers how many actions both logs have. It is not a linking fault: every
    # pair that is left still agrees.
    assert b["link"]["coverage"] < 1.0 and b["link"]["agreement"] == 1.0
    assert b["link"]["pairs"] == b["witnesses"][SPANS]["actions_seen"]
    assert b["spans_delivered"] < b["calls_written_by_receiver"] == b["access_lines_delivered"]
    # Access still saw them, so the union is whole and the table moved, not shrank.
    assert b["union_count"] == b["actions_in_ledger"]
    spans_bit = 1 << b["witness_names"].index(SPANS)
    assert sum(n for i, n in enumerate(b["table"]) if not i & spans_bit) > 0


def test_lost_access_lines_and_a_missing_file_are_counted_not_fatal(tmp_path):
    run = _copy(tmp_path)
    (run / "witnesses" / "logs.jsonl").unlink()
    s = summarise(run)
    for p in s["planes"].values():
        assert p["witnesses"][ACCESS]["actions_seen"] == 0
        assert p["access_lines_delivered"] == 0 < p["calls_written_by_receiver"]
        assert p["link"] == {"pairs": 0, "pairs_same_key": 0, "agreement": None, "coverage": 0.0}
        assert p["union_count"] == p["actions_in_ledger"]  # spans alone still cover it here


def test_a_record_matching_no_ledger_entry_is_reported(tmp_path):
    run = _copy(tmp_path)
    path = run / "logs" / "front-b.jsonl"
    ghost = {"args": {"order_id": "O999"}, "ok": True, "rpc_id": 99, "session": "s0",
             "tool": "cancel_order", "trace_id": "f" * 32, "ts": 1.0}  # fmt: skip
    delivered = run / "witnesses" / "logs.jsonl"
    rec = {"resourceLogs": [{"scopeLogs": [{"logRecords": [{
        "observedTimeUnixNano": "1", "body": {"stringValue": json.dumps(ghost)},
        "attributes": [{"key": "log.file.name", "value": {"stringValue": "front-b.jsonl"}}],
    }]}]}]}  # fmt: skip
    path.write_text(path.read_text() + json.dumps(ghost) + "\n")
    delivered.write_text(delivered.read_text() + json.dumps(rec) + "\n")
    b = summarise(run)["planes"]["b"]
    assert b["witnesses"][ACCESS]["matching_no_action"] == 1
    assert b["union_count"] == b["actions_in_ledger"] + 1


def test_an_env_row_with_no_writer_is_refused(tmp_path):
    run = _copy(tmp_path)
    path = run / "witnesses" / "env.jsonl"
    row = {"session": "s0", "op": "cancel_order", "resource": "O1", "value": "cancelled",
           "principal": None}  # fmt: skip
    path.write_text(path.read_text() + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="no stratum"):
        summarise(run)


def _edit_spans(run: Path, edit) -> None:
    """Apply `edit(span)` to every span of a copied run. Return True from it to stop."""
    path = run / "witnesses" / "spans.jsonl"
    out, done = [], False
    for line in path.read_text().splitlines():
        d = json.loads(line)
        for rs in d["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                for sp in ss["spans"]:
                    if not done:
                        done = bool(edit(sp))
        out.append(json.dumps(d))
    path.write_text("\n".join(out) + "\n")


def test_two_logs_that_share_an_id_and_disagree_are_reported_not_hidden(tmp_path):
    run = _copy(tmp_path)

    def wrong_amount(sp) -> bool:
        attrs = {a["key"]: a["value"] for a in sp["attributes"]}
        if attrs.get("gen_ai.tool.name", {}).get("stringValue") != "issue_refund":
            return False
        args = json.loads(attrs["gen_ai.tool.call.arguments"]["stringValue"])
        args["amount"] = 0.01
        attrs["gen_ai.tool.call.arguments"]["stringValue"] = json.dumps(args)
        return True

    _edit_spans(run, wrong_amount)
    before, after = summarise(RUN)["planes"], summarise(run)["planes"]
    (plane,) = [p for p in after if after[p] != before[p]]
    p, was = after[plane], before[plane]
    assert p["link"]["pairs"] == was["link"]["pairs"]
    assert p["link"]["pairs_same_key"] == was["link"]["pairs_same_key"] - 1
    assert p["link"]["agreement"] < 1.0 and p["link"]["coverage"] == 1.0
    # Scoring sees the bad record; the floor is still the table's own count.
    assert p["witnesses"][SPANS]["actions_seen"] == was["witnesses"][SPANS]["actions_seen"] - 1
    assert p["witnesses"][SPANS]["matching_no_action"] == 1
    assert p["trust_trace"] == was["trust_trace"]


def test_a_record_with_no_session_is_set_aside_and_counted(tmp_path):
    run = _copy(tmp_path)
    path = run / "witnesses" / "logs.jsonl"
    path.write_text(path.read_text().replace('\\"session\\":\\"s0\\",', ""))
    after = summarise(run)["planes"]["a"]
    was = summarise(RUN)["planes"]["a"]
    assert after["records_without_session"] > 0
    # Without the session a signature could be any session's; the record makes no
    # action of its own, so the union does not grow past what the other logs hold.
    assert after["union_count"] == was["union_count"] == after["actions_in_ledger"]
    assert after["witnesses"][ACCESS]["matching_no_action"] == 0
    assert (
        after["witnesses"][ACCESS]["records"]
        == was["witnesses"][ACCESS]["records"] - after["records_without_session"]
    )


def test_records_reach_the_table_in_call_order_whatever_file_they_came_from():
    obs = [o for o in load(RUN)["obs"] if o.witness != ENV]
    order = [(str(o.session), o.stratum, o.seq) for o in obs]
    assert order == sorted(order)
    # So a tie between a spans-only and an access-only action is not always won by spans.
    assert {o.witness for o in obs[:4]} == {SPANS, ACCESS}


def test_the_log_files_have_one_plane_and_one_service_each():
    assert set(LOG_PLANE) == set(LOG_SERVICE)
    assert sorted(LOG_PLANE.values()) == ["a", "b"]
