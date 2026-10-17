from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from lacuna.agents.llm import BudgetExceeded, ChatClient, LLMAuditor, parse_actions, run_agent
from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
from lacuna.envs.kernel import TRACE_PATH, Kernel
from lacuna.envs.tasks import make_episode
from lacuna.envs.tools import Router
from lacuna.episode import run_episode
from lacuna.linkage import oracle_table, signature_table
from lacuna.witnesses import ENV, GATEWAY, env_records, gateway_records


def _router(seed: int = 1, **kw) -> tuple[Kernel, Router]:
    kernel = Kernel(make_episode(seed).world)
    return kernel, Router(kernel, np.random.default_rng(0), **kw)


def test_trace_file_is_read_only_to_the_agent():
    kernel, router = _router()
    out = json.loads(router.execute("c1", "write_file", {"path": TRACE_PATH, "content": ""}))
    assert "error" in out
    assert kernel.ledger == []


def test_every_successful_action_is_ledgered_once_and_reads_never():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    router.execute("c1", "get_customer", {"customer_id": "C1"})
    router.execute("c2", "update_customer", {"customer_id": "C1", "field": "tier", "value": "gold"})
    router.execute("c3", "issue_refund", {"order_id": "O1", "amount": 1e9})  # over total: error
    assert [e.op for e in kernel.ledger] == ["update_customer"]
    recs = gateway_records(router.gateway, router.call_ledger)
    assert [(r.witness, r.oracle) for r in recs] == [(GATEWAY, 0)]


def test_diff_loses_multiplicity_and_reverts():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    orig = kernel.world.customers["C2"]["email"]
    for i, v in enumerate(["a@x.org", "b@x.org"]):
        router.execute(
            f"u{i}", "update_customer", {"customer_id": "C2", "field": "email", "value": v}
        )
    for i in range(2):  # the same value twice: two actions, one change
        router.execute(
            f"p{i}", "update_customer", {"customer_id": "C3", "field": "phone", "value": "1"}
        )
    assert len(kernel.ledger) == 4
    diff = [r for r in env_records(kernel) if r.op == "update_customer"]
    # C2.email: two writes, one diff line attributed to the last; C3.phone: one line.
    assert sorted(r.resource for r in diff) == ["C2.email", "C3.phone"]
    c2 = next(r for r in diff if r.resource == "C2.email")
    assert c2.oracle == 1 and c2.value == "b@x.org" and orig != "b@x.org"


def test_reverted_change_is_invisible_to_the_diff():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    orig = kernel.world.customers["C4"]["address"]
    router.execute("a", "update_customer", {"customer_id": "C4", "field": "address", "value": "x"})
    router.execute("b", "update_customer", {"customer_id": "C4", "field": "address", "value": orig})
    assert len(kernel.ledger) == 2
    assert [r for r in env_records(kernel) if r.witness == ENV] == []


def test_oracle_table_accounts_for_every_ledger_entry():
    res = run_episode(
        5, ScriptedAgent(np.random.default_rng(5)), ScriptedAuditor(np.random.default_rng(1))
    )
    table, _ = oracle_table(res.ledger, res.records, 4)
    assert table.sum() == len(res.ledger)


def test_signature_matches_oracle_when_nothing_is_lost():
    agent = ScriptedAgent(np.random.default_rng(6), p_redo=0.0, p_redraft=0.0)
    res = run_episode(
        6,
        agent,
        ScriptedAuditor(np.random.default_rng(1), p_miss=0.0),
        p_gateway_drop=0.0,
        p_trace_drop=0.0,
    )
    table, stats = oracle_table(res.ledger, res.records, 4)
    sig = signature_table(res.records, 4)
    assert stats["auditor_false_positives"] == 0
    assert table[0] == 0
    assert sig[1:].sum() == table[1:].sum()


def _fake_transport(replies: list[dict]) -> httpx.MockTransport:
    it = iter(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        msg = next(it)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": msg}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )

    return httpx.MockTransport(handler)


def _client(replies: list[dict], budget: float = 1.0) -> ChatClient:
    return ChatClient("http://fake", "k", "m", 1.0, 2.0, budget, transport=_fake_transport(replies))


def test_agent_loop_executes_tool_calls_and_stops():
    call = {
        "id": "t1",
        "type": "function",
        "function": {"name": "cancel_order", "arguments": json.dumps({"order_id": "O3"})},
    }
    client = _client([{"content": None, "tool_calls": [call]}, {"content": "done"}])
    ep = make_episode(2)
    kernel = Kernel(ep.world)
    router = Router(kernel, np.random.default_rng(0))
    transcript = run_agent(client, ep, router)
    assert transcript[-1]["content"] == "done"
    assert len(client.receipt) == 2
    assert client.spent == pytest.approx(2 * (1000 * 1.0 + 100 * 2.0) / 1e6)


def test_budget_cap_stops_before_the_next_call():
    client = _client([{"content": "a"}, {"content": "b"}], budget=0.0011)
    client.complete([{"role": "user", "content": "x"}])
    with pytest.raises(BudgetExceeded):
        client.complete([{"role": "user", "content": "x"}])


def test_auditor_parses_first_json_array_or_nothing():
    assert parse_actions('ok [{"op":"cancel_order","resource":"O1","value":"cancelled"}]') == [
        {"op": "cancel_order", "resource": "O1", "value": "cancelled"}
    ]
    assert parse_actions("no actions") == []
    assert parse_actions("[not json") == []
    auditor = LLMAuditor(_client([{"content": "[]"}]))
    assert auditor.reconstruct([{"role": "assistant", "content": "hi"}]) == []


def test_client_sends_routing_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    client = ChatClient(
        "http://fake", "k", "m", 1.0, 1.0, 1.0, transport=httpx.MockTransport(handler)
    )
    client.headers = {"x-opencode-session": "s1"}
    client.complete([{"role": "user", "content": "x"}])
    assert seen["x-opencode-session"] == "s1"
    assert seen["authorization"] == "Bearer k"


def test_narration_view_hides_calls_and_results():
    transcript = [
        {"role": "user", "content": "queue"},
        {
            "role": "assistant",
            "content": "Refunding O1.",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "cancel_order", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": '{"status": "done"}'},
    ]
    from lacuna.agents.llm import render

    full = render(transcript)
    narration = render(transcript, view="narration")
    assert "CALL t1" in full and "RESULT t1" in full
    assert narration == "AGENT: Refunding O1."


def test_episode_result_round_trips_its_transcript():
    res = run_episode(
        12, ScriptedAgent(np.random.default_rng(12)), ScriptedAuditor(np.random.default_rng(1))
    )
    blob = res.to_json()
    assert blob["transcript"] == res.transcript
    assert any(m.get("tool_calls") for m in blob["transcript"])
