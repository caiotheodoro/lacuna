from __future__ import annotations

import json

import numpy as np
import pytest

from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
from lacuna.envs.infra import ACTION_OPS, InfraKernel
from lacuna.envs.infra_tasks import make_episode
from lacuna.envs.registry import BACKOFFICE, ENVS
from lacuna.envs.tools import Router
from lacuna.episode import run_episode
from lacuna.linkage import VALUE_OPS, oracle_table, signature_table
from lacuna.thinning import POOLS
from lacuna.witnesses import ENV, GATEWAY, TRACE, env_records, gateway_records, trace_records

INFRA = ENVS["infra"]


def _router(seed: int = 1, **kw) -> tuple[InfraKernel, Router]:
    kernel = InfraKernel(make_episode(seed).world)
    return kernel, Router(kernel, np.random.default_rng(0), env=INFRA, **kw)


def _episode(seed: int, n_tickets: int = 30, **kw):
    return run_episode(
        seed,
        ScriptedAgent(np.random.default_rng(seed)),
        ScriptedAuditor(np.random.default_rng(1), p_miss=0.0, env=INFRA),
        n_tickets=n_tickets,
        env=INFRA,
        **kw,
    )


def test_op_names_are_disjoint_across_envs():
    # linkage.VALUE_OPS is one union over envs; that is only safe if no op
    # name means two things.
    assert not set(BACKOFFICE.action_tools) & set(INFRA.action_tools)
    assert set(VALUE_OPS) == set(BACKOFFICE.value_ops) | set(INFRA.value_ops)
    assert set(INFRA.action_tools) == set(ACTION_OPS)
    assert {t["function"]["name"] for t in INFRA.tools} == set(INFRA.handlers(None))


def test_backoffice_env_object_is_the_default_path():
    def run(env):
        return run_episode(
            4,
            ScriptedAgent(np.random.default_rng(4)),
            ScriptedAuditor(np.random.default_rng(1), env=env),
            n_tickets=30,
            env=env,
        ).to_json()

    assert run(BACKOFFICE) == run(None)


def test_every_successful_infra_action_is_ledgered_once_and_reads_never():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    router.execute("r1", "list_services", {})
    router.execute("r2", "get_service", {"service": "api"})
    router.execute("r3", "get_metrics", {"service": "api"})
    router.execute("a1", "scale_service", {"service": "api", "replicas": 9})
    router.execute("e1", "update_config", {"service": "api", "key": "nope", "value": "1"})
    router.execute("e2", "scale_service", {"service": "api", "replicas": 2.5})
    router.execute("e3", "send_page", {"team": "nobody", "message": "x"})
    router.execute("a2", "send_page", {"team": "core", "message": "x"})
    assert [e.op for e in kernel.ledger] == ["scale_service", "send_page"]
    recs = gateway_records(router.gateway, router.call_ledger, INFRA)
    assert [(r.witness, r.oracle) for r in recs] == [(GATEWAY, 0), (GATEWAY, 1)]
    assert [r.oracle for r in trace_records(kernel, router.call_ledger, INFRA)] == [0, 1]


def test_runbooks_cannot_reach_the_trace_file():
    kernel, router = _router()
    out = json.loads(
        router.execute("c1", "write_runbook", {"name": "../var/log/x", "content": "hi"})
    )
    assert "error" in out and kernel.ledger == []


def test_transient_work_is_invisible_to_the_diff():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    svc = kernel.world.services["search"]
    replicas, live = svc["replicas"], svc["history"][-1]
    calls = [
        ("deploy_version", {"service": "search", "version": "9.9.9"}),
        ("rollback", {"service": "search"}),
        ("scale_service", {"service": "search", "replicas": 30}),
        ("scale_service", {"service": "search", "replicas": replicas}),
        ("update_config", {"service": "search", "key": "log_level", "value": "debug"}),
        ("update_config", {"service": "search", "key": "log_level", "value": "info"}),
    ]
    for i, (tool, args) in enumerate(calls):
        assert "error" not in router.execute(f"c{i}", tool, args)
    assert len(kernel.ledger) == 6
    assert svc["history"][-1] == live
    assert env_records(kernel, INFRA) == []


def test_two_edits_show_as_one_change_attributed_to_the_last():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    for i, v in enumerate(["40", "80"]):
        router.execute(
            f"c{i}", "update_config", {"service": "cart", "key": "pool_size", "value": v}
        )
    (rec,) = env_records(kernel, INFRA)
    assert (rec.op, rec.resource, rec.value, rec.oracle) == (
        "update_config",
        "cart.pool_size",
        "80",
        1,
    )


def test_live_version_is_attributed_to_the_deploy_that_shipped_it():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    router.execute("a", "deploy_version", {"service": "auth", "version": "7.1.0"})  # stays
    router.execute("b", "deploy_version", {"service": "auth", "version": "7.2.0"})  # fails
    router.execute("c", "rollback", {"service": "auth"})
    router.execute("d", "rollback", {"service": "cart"})  # past the start: no deploy shipped it
    recs = {(r.op, r.resource): r for r in env_records(kernel, INFRA)}
    assert recs[("deploy_version", "auth")].value == "7.1.0"
    assert recs[("deploy_version", "auth")].oracle == 0
    assert recs[("rollback", "cart")].oracle == 3
    assert len(recs) == 2


def test_pages_reach_the_network_sink():
    kernel, router = _router(p_gateway_drop=0.0, p_trace_drop=0.0)
    router.execute("p", "send_page", {"team": "money", "message": "billing is down"})
    (rec,) = env_records(kernel, INFRA)
    assert (rec.witness, rec.op, rec.resource, rec.oracle) == (ENV, "send_page", "money", 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_table_sums_to_the_ledger(seed):
    res = _episode(seed, p_gateway_drop=0.3, p_trace_drop=0.3)
    table, _ = oracle_table(res.ledger, res.records, 4)
    assert table.sum() == len(res.ledger)


def test_normaliser_reproduces_the_ledger():
    res = _episode(3, p_gateway_drop=0.0, p_trace_drop=0.0)
    ledger = {e.id: (e.op, e.resource, e.value) for e in res.ledger}
    for r in res.records:
        if r.witness in (TRACE, GATEWAY):
            assert (r.op, r.resource, r.value) == ledger[r.oracle]


def test_scripted_infra_episode_end_to_end():
    res = _episode(5, p_gateway_drop=0.0, p_trace_drop=0.0)
    assert res.n_tool_errors == 0
    table, stats = oracle_table(res.ledger, res.records, 4)
    assert table[0] == 0 and stats["auditor_false_positives"] == 0
    # With complete logs the trace and gateway see every action, so every
    # action is in a cell with bits 0 and 1 set; W3 misses the transient ones.
    assert table[0b1011] + table[0b1111] == len(res.ledger)
    assert 0 < table[0b1011] < len(res.ledger)
    np.testing.assert_array_equal(signature_table(res.records, 4)[1:], table[1:])


def test_smoke_check_outcome_drives_the_rollback():
    ep = make_episode(6, 30)
    deploys = [t for t in ep.tickets if t.plan[0][0] == "deploy_version"]
    assert deploys
    for t in deploys:
        svc, version = t.plan[0][1]["service"], t.plan[0][1]["version"]
        rolled_back = any(tool == "rollback" for tool, _ in t.plan)
        assert rolled_back == ((svc, version) in ep.world.bad_versions)


def test_transfer_pool_uses_the_full_env_view():
    names = [c.name for c in POOLS["transfer"]]
    assert names == ["transfer_light", "transfer_mid", "transfer_mid_rho100"]
    assert {c.env_view for c in POOLS["transfer"]} == {"full"}


def test_llm_loop_offers_the_infra_tools_and_prompt():
    import httpx

    from lacuna.agents.llm import ChatClient, run_agent

    bodies: list[dict] = []
    call = {
        "id": "t1",
        "type": "function",
        "function": {"name": "rollback", "arguments": json.dumps({"service": "api"})},
    }
    replies = iter([{"content": None, "tool_calls": [call]}, {"content": "done"}])

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": next(replies)}]})

    client = ChatClient("http://fake", "k", "m", 1.0, 1.0, 1.0, httpx.MockTransport(handler))
    ep = make_episode(2)
    kernel = InfraKernel(ep.world)
    router = Router(kernel, np.random.default_rng(0), env=INFRA)
    run_agent(client, ep, router)
    assert {t["function"]["name"] for t in bodies[0]["tools"]} == set(INFRA.handlers(None))
    assert bodies[0]["messages"][0]["content"] == INFRA.agent_system
    assert [e.op for e in kernel.ledger] == ["rollback"]
