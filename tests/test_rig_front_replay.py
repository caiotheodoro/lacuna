"""The MCP front and the replay client, joined in-process: no socket, no collector.

The client talks streamable HTTP to the front's real ASGI app through httpx's
in-process transport, so request headers, JSON-RPC ids and the access log are
the ones a run produces. Spans go to an in-memory exporter.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import httpx
import numpy as np
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lacuna.envs.kernel import Kernel
from lacuna.envs.registry import BACKOFFICE
from lacuna.envs.service import KernelService
from lacuna.envs.tasks import make_episode
from lacuna.envs.tools import TOOLS, Router
from lacuna.rig.front import build_server, make_app, trace_id_of
from lacuna.rig.replay import replay_all, replay_session
from lacuna.rig.sessions import Session, Step, scripted_sessions


class Rig:
    """A kernel service, one front per plane, and a tracer, all in this process."""

    def __init__(self):
        self.service = KernelService(BACKOFFICE, {"ka": "plane-a", "kb": "plane-b"})
        self.logs: dict[str, list[dict]] = {"a": [], "b": []}
        self.spans = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.spans))
        self.tracer = provider.get_tracer("test")
        self.apps = {}
        self.stack = contextlib.AsyncExitStack()

    async def __aenter__(self):
        for plane, key in (("a", "ka"), ("b", "kb")):

            async def forward(session, tool, args, key=key):
                return self.service.execute(session, key, tool, args)

            app, manager = make_app(build_server(BACKOFFICE, forward, self.logs[plane].append))
            await self.stack.enter_async_context(manager.run())
            self.apps[plane] = app
        return self

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    def connect(self, plane: str):
        transport = httpx.ASGITransport(app=self.apps[plane])

        @contextlib.asynccontextmanager
        async def connect(headers: dict[str, str]):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://front", headers=headers
            ) as http:
                async with streamable_http_client("http://front/mcp", http_client=http) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        yield client

        return connect

    def tool_spans(self) -> list:
        return [s for s in self.spans.get_finished_spans() if s.name.startswith("execute_tool")]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def test_the_front_lists_the_env_tools_with_their_schemas():
    async def go():
        async with Rig() as rig:
            async with rig.connect("a")({"x-lacuna-session": "s0"}) as client:
                return (await client.list_tools()).tools

    tools = run(go())
    assert [t.name for t in tools] == [t["function"]["name"] for t in TOOLS]
    refund = next(t for t in tools if t.name == "issue_refund")
    assert refund.inputSchema == TOOLS[5]["function"]["parameters"]
    assert refund.description == "Refund part or all of an order."


def test_one_plane_replayed_alone_leaves_the_ledger_an_in_process_run_leaves():
    seed = 4
    (session,) = scripted_sessions(seed, 7, ("a",))

    async def go():
        async with Rig() as rig:
            rig.service.open(session.session, seed, 7)
            texts = await replay_session(session, rig.connect("a"), rig.tracer)
            return rig, texts

    rig, texts = run(go())
    kernel = Kernel(make_episode(seed, 7).world)
    router = Router(kernel, np.random.default_rng(0), 0.0, 0.0)
    want = [router.execute(f"c{i}", st.tool, st.args) for i, st in enumerate(session.steps)]
    assert texts == want  # the agent would have seen the same replies
    got = [(e["op"], e["resource"], e["value"]) for e in rig.service.ledger(session.session)]
    assert got == [(e.op, e.resource, e.value) for e in kernel.ledger]
    assert {e["principal"] for e in rig.service.ledger(session.session)} == {"plane-a"}


def _two_planes(seed: int = 2):
    sessions = scripted_sessions(seed, 7, ("a", "b"))

    async def go():
        async with Rig() as rig:
            rig.service.open(sessions[0].session, seed, 7)
            await asyncio.gather(
                *(replay_session(s, rig.connect(s.plane), rig.tracer) for s in sessions)
            )
            return rig

    return sessions, run(go())


def test_each_call_is_one_span_and_one_access_log_line_with_the_same_ids():
    sessions, rig = _two_planes()
    spans = rig.tool_spans()
    for s in sessions:
        mine = [sp for sp in spans if sp.attributes["lacuna.plane"] == s.plane]
        lines = rig.logs[s.plane]
        assert len(mine) == len(lines) == len(s.steps)
        for step, span, line in zip(s.steps, mine, lines, strict=True):
            a = span.attributes
            assert a["gen_ai.tool.name"] == step.tool == line["tool"]
            assert json.loads(a["gen_ai.tool.call.arguments"]) == step.args == line["args"]
            assert a["lacuna.session"] == s.session == line["session"]
            # AGENTS.md rule 4: the link id is what the client library assigned and the
            # receiver saw, compared here for every call.
            assert a["jsonrpc.request.id"] == line["rpc_id"]
            assert f"{span.context.trace_id:032x}" == line["trace_id"]
            assert a["lacuna.tool.is_error"] == (not line["ok"])
        # One trace per plane's share of a session; ids number requests in order.
        assert len({sp.context.trace_id for sp in mine}) == 1
        ids = [line["rpc_id"] for line in lines]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert len({sp.context.trace_id for sp in spans}) == 2


def test_the_planes_of_a_session_share_one_world():
    sessions, rig = _two_planes()
    ledger = rig.service.ledger(sessions[0].session)
    actions = {
        s.plane: sum(1 for st in s.steps if st.tool in BACKOFFICE.action_tools) for s in sessions
    }
    made = {p: sum(1 for e in ledger if e["principal"] == f"plane-{p}") for p in ("a", "b")}
    # Scripted tickets can collide across planes (an order cancelled twice), so a
    # plane makes at most what it attempts, and both planes act.
    assert all(0 < made[p] <= actions[p] for p in made)
    ok = sum(1 for p in ("a", "b") for line in rig.logs[p] if line["ok"] and line["tool"] in
             BACKOFFICE.action_tools)  # fmt: skip
    assert ok == len(ledger)


def test_values_in_a_session_file_do_not_reach_the_ids():
    # A well-formed forgery of every id, placed where a model could put it: the arguments.
    fake_trace = "a" * 32
    forged = {
        "path": "/x.txt",
        "content": "c",
        "id": 999,
        "traceparent": f"00-{fake_trace}-{'b' * 16}-01",
        "jsonrpc.request.id": 777,
        "x-lacuna-session": "other",
        "lacuna.session": "other",
    }
    assert trace_id_of(forged["traceparent"]) == fake_trace
    session = Session("s1", 1, 7, "a", (Step(0, "write_file", forged, 0.0),) * 2)

    async def go():
        async with Rig() as rig:
            rig.service.open("s1", 1, 7)
            await replay_session(session, rig.connect("a"), rig.tracer)
            await replay_session(session, rig.connect("a"), rig.tracer)
            return rig

    rig = run(go())
    lines, spans = rig.logs["a"], rig.tool_spans()
    ids = [line["rpc_id"] for line in lines]
    assert ids == [sp.attributes["jsonrpc.request.id"] for sp in spans]
    assert not {999, 777} & set(ids)
    assert all(line["session"] == "s1" for line in lines)
    assert all(sp.attributes["lacuna.session"] == "s1" for sp in spans)
    traces = {line["trace_id"] for line in lines}
    assert fake_trace not in traces | {f"{sp.context.trace_id:032x}" for sp in spans}
    # The SDK draws a trace id per replay: the same file twice is two traces.
    assert len(traces) == 2
    # The arguments themselves travel untouched: they are payload.
    assert all(line["args"] == forged for line in lines)


def test_replay_all_runs_every_session_and_reports_the_ones_that_broke():
    good = scripted_sessions(2, 7, ("a",))[0]
    other = scripted_sessions(3, 7, ("a",))[0]

    async def go():
        async with Rig() as rig:
            rig.service.open(good.session, 2, 7)
            rig.service.open(other.session, 3, 7)
            calls = 0

            def connect(headers):
                nonlocal calls
                calls += 1
                if headers["x-lacuna-session"] == other.session:
                    raise ConnectionError("refused")
                return rig.connect("a")(headers)

            failed = await replay_all([good, other], connect, rig.tracer, stagger_s=0.01)
            return rig, failed, calls

    rig, failed, calls = run(go())
    assert calls == 2
    assert failed == {other.session: "ConnectionError: refused"}
    assert len(rig.logs["a"]) == len(good.steps)
    assert rig.service.ledger(other.session) == []


def test_a_failed_call_is_marked_on_the_span_and_in_the_log_and_is_not_an_action():
    steps = (
        Step(0, "cancel_order", {"order_id": "nope"}, 0.0),
        Step(0, "no_such_tool", {}, 0.0),
        Step(0, "issue_refund", {"order_id": "O1"}, 0.0),  # a required argument is missing
    )

    async def go():
        async with Rig() as rig:
            rig.service.open("s1", 1, 7)
            session = Session("s1", 1, 7, "b", steps)
            texts = await replay_session(session, rig.connect("b"), rig.tracer)
            return rig, texts

    rig, texts = run(go())
    assert [json.loads(t).keys() for t in texts] == [{"error"}] * 3
    assert [line["ok"] for line in rig.logs["b"]] == [False] * 3
    assert [sp.attributes["lacuna.tool.is_error"] for sp in rig.tool_spans()] == [True] * 3
    assert rig.service.ledger("s1") == []


def test_a_call_for_a_session_nobody_opened_is_an_error_and_is_logged():
    async def go():
        async with Rig() as rig:
            step = Step(0, "cancel_order", {"order_id": "O1"}, 0.0)
            await replay_session(Session("ghost", 1, 7, "a", (step,)), rig.connect("a"), rig.tracer)
            return rig

    rig = run(go())
    assert [(line["session"], line["ok"]) for line in rig.logs["a"]] == [("ghost", False)]


def test_trace_id_of_reads_a_w3c_header_and_refuses_anything_else():
    assert trace_id_of("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01") == (
        "0af7651916cd43dd8448eb211c80319c"
    )
    for bad in (None, "", "garbage", "00-short-b7ad6b7169203331-01", "00-" + "g" * 32 + "-x-01"):
        assert trace_id_of(bad) is None


class FakeTime:
    """A clock that only moves when something sleeps, and oversleeps by `late` each time."""

    def __init__(self, start: float, late: float = 0.0):
        self.now = start
        self.late = late
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.now += seconds + self.late


def _three_steps(plane: str = "a", gaps=(1.0, 1.0, 1.0)) -> Session:
    a = {"customer_id": "C1", "field": "tier", "value": "v"}
    steps = tuple(
        Step(0, "update_customer", {**a, "value": f"v{i}"}, g) for i, g in enumerate(gaps)
    )
    return Session("s1", 1, 7, plane, steps)


def test_steps_are_due_on_a_schedule_so_a_slow_reply_does_not_push_the_rest_back():
    session = _three_steps()
    fake = FakeTime(100.0, late=0.25)

    async def go():
        async with Rig() as rig:
            rig.service.open("s1", 1, 7)
            await replay_session(
                session, rig.connect("a"), rig.tracer, origin=100.0,
                clock=fake.clock, sleep=fake.sleep,
            )  # fmt: skip
            return rig

    rig = run(go())
    # Due at 101, 102, 103. Each sleep overruns by 0.25 s, and the next one is shorter
    # by as much: sleeping a full gap after every reply would give [1.0, 1.0, 1.0].
    assert fake.slept == [1.0, 0.75, 0.75]
    assert len(rig.service.ledger("s1")) == 3


def test_a_session_whose_start_is_still_ahead_waits_for_it_and_a_late_one_catches_up():
    async def go(origin: float, start: float):
        fake = FakeTime(start)
        async with Rig() as rig:
            rig.service.open("s1", 1, 7)
            await replay_session(
                _three_steps(gaps=(0.5, 0.5, 0.5)), rig.connect("a"), rig.tracer,
                origin=origin, clock=fake.clock, sleep=fake.sleep,
            )  # fmt: skip
            return fake.slept, len(rig.service.ledger("s1"))

    # Start 2 s ahead: wait for it, then the three gaps.
    assert run(go(102.0, 100.0)) == ([2.0, 0.5, 0.5, 0.5], 3)
    # Start 1.2 s behind: the first two steps are overdue and go at once.
    assert run(go(100.0, 101.2)) == ([0.3], 3)


def test_both_planes_of_a_session_share_its_start_and_sessions_differ():
    sessions = [_three_steps("a"), _three_steps("b")]
    other = Session("s2", 2, 7, "a", _three_steps().steps)
    seen: dict[tuple[str, str], float] = {}

    async def go():
        fake = FakeTime(50.0)
        async with Rig() as rig:
            rig.service.open("s1", 1, 7)
            rig.service.open("s2", 2, 7)

            async def spy(session, connect, tracer, origin, clock, sleep):
                seen[(session.session, session.plane)] = origin

            import lacuna.rig.replay as replay

            real = replay.replay_session
            replay.replay_session = spy
            try:
                await replay_all(
                    [*sessions, other], rig.connect("a"), rig.tracer, stagger_s=30.0,
                    t0=1000.0, clock=fake.clock, sleep=fake.sleep,
                )  # fmt: skip
            finally:
                replay.replay_session = real

    run(go())
    assert seen[("s1", "a")] == seen[("s1", "b")]
    assert seen[("s1", "a")] != seen[("s2", "a")]
    assert all(1000.0 <= t <= 1030.0 for t in seen.values())
