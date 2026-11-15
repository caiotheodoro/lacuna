"""An originator: replays sessions over MCP and emits one span per tool call.

This is the process that stands for an agent in a run. It reads what a recorded
agent asked for and asks again through a real MCP client. Each call is due at a
time on the session's own clock (`lacuna.rig.sessions.offsets`), counted from a
start both originators of a run are given, so the planes of one session make
their calls in the order the session had. A reply slower than the time to the
next call does delay that call; how late the latest call of a run was is in the
originator's closing report, and a run where it is large fails its integrity
check (`lacuna.rig.integrity`).
Its telemetry is the OpenTelemetry SDK with every setting at its default; what
is lost when the process is killed or the collector is away is the SDK's doing.

The ids a span carries (AGENTS.md rule 4):

- the trace id is drawn by the SDK when a session's root span starts, and is
  sent on every request of the session as `traceparent`;
- the JSON-RPC id is the MCP client library's counter, read just before the call
  it will number;
- the session is the file's session name, sent as a header.

Tool arguments are payload. They are sent and recorded, and nothing in them is
used as an id (`tests/test_rig_front_replay.py`).

Needs the `rig` extra.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

from mcp import ClientSession
from opentelemetry import trace
from opentelemetry.trace import Tracer

from lacuna.rig.front import SESSION_HEADER
from lacuna.rig.sessions import Session, load, offsets, session_delay

# headers -> an initialised MCP client session
Connect = Callable[[dict[str, str]], AbstractAsyncContextManager[ClientSession]]


async def _until(due: float, clock: Callable[[], float], sleep) -> None:
    wait = due - clock()
    if wait > 0:
        await sleep(wait)


async def replay_session(
    session: Session,
    connect: Connect,
    tracer: Tracer,
    origin: float | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    lags: list[float] | None = None,
) -> list[str]:
    """Replay one plane's share of a session. Returns the text of each reply, in order.

    `origin` is the session's start on `clock` (now, if not given). Step `i` is due at
    `origin + offsets(session)[i]`; a step that is already overdue goes at once. How
    long after it was due each call was made is appended to `lags`.
    """
    if origin is None:
        origin = clock()
    await _until(origin, clock, sleep)
    root = tracer.start_span(
        "session", attributes={"lacuna.session": session.session, "lacuna.plane": session.plane}
    )
    ctx = trace.set_span_in_context(root)
    sc = root.get_span_context()
    headers = {
        "traceparent": f"00-{sc.trace_id:032x}-{sc.span_id:016x}-01",
        SESSION_HEADER: session.session,
    }
    texts: list[str] = []
    try:
        async with connect(headers) as client:
            # An agent lists the tools before it calls one. It also means the client
            # makes no request of its own between the id read below and the call.
            await client.list_tools()
            for step, due in zip(session.steps, offsets(session), strict=True):
                await _until(origin + due, clock, sleep)
                if lags is not None:
                    lags.append(max(0.0, clock() - (origin + due)))
                with tracer.start_as_current_span(
                    f"execute_tool {step.tool}",
                    context=ctx,
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": step.tool,
                        "gen_ai.tool.call.arguments": json.dumps(step.args, sort_keys=True),
                        "mcp.method.name": "tools/call",
                        "lacuna.session": session.session,
                        "lacuna.plane": session.plane,
                    },
                ) as span:
                    # The id the library gives its next request (mcp 1.x, BaseSession).
                    span.set_attribute("jsonrpc.request.id", client._request_id)
                    result = await client.call_tool(step.tool, step.args)
                    span.set_attribute("lacuna.tool.is_error", bool(result.isError))
                    texts.append(result.content[0].text if result.content else "")
    finally:
        root.end()
    return texts


def http_connect(url: str) -> Connect:
    import httpx
    from mcp.client.streamable_http import streamable_http_client

    @contextlib.asynccontextmanager
    async def connect(headers: dict[str, str]) -> AsyncIterator[ClientSession]:
        # The MCP transport's own default timeouts: a server may hold a stream open.
        timeout = httpx.Timeout(30.0, read=300.0)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http:
            async with streamable_http_client(url, http_client=http) as (read, write, _):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    yield client

    return connect


async def replay_all(
    sessions: list[Session],
    connect: Connect,
    tracer: Tracer,
    stagger_s: float = 0.0,
    t0: float | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    timing: dict[str, float] | None = None,
) -> dict[str, str]:
    """Every session at once, each from its own seeded start. Returns failures by session.

    `t0` is the run's start on `clock`; two originators given the same `t0` start each
    session at the same moment. `timing`, if given, is filled with `start_lag_s` (how
    long after `t0` this began; 0 if it was ready in time) and `max_lag_s` (the
    latest any call was made after it was due).
    """
    failed: dict[str, str] = {}
    now = clock()
    start = now if t0 is None else t0
    lags: list[float] = []

    async def one(s: Session) -> None:
        origin = start + session_delay(s.session, stagger_s)
        try:
            await replay_session(s, connect, tracer, origin, clock, sleep, lags)
        except Exception as e:  # one broken session must not stop the others
            failed[s.session] = f"{type(e).__name__}: {e}"

    await asyncio.gather(*(one(s) for s in sessions))
    if timing is not None:
        timing["start_lag_s"] = round(max(0.0, now - start), 3)
        timing["max_lag_s"] = round(max(lags, default=0.0), 3)
    return failed


def main() -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sessions", type=Path, required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--url", required=True, help="the MCP endpoint of this plane")
    p.add_argument("--otlp", required=True, help="OTLP/HTTP traces endpoint")
    p.add_argument("--service", required=True, help="service.name of this originator")
    p.add_argument("--stagger", type=float, default=0.0)
    p.add_argument("--t0", type=float, default=None, help="the run's start, unix seconds")
    p.add_argument(
        "--warmup", type=float, default=0.0, help="seconds to wait before the SDK is created"
    )
    a = p.parse_args()

    # Two processes started together would export in step for the whole run. The wait
    # gives each its own phase; the SDK itself is untouched.
    time.sleep(max(0.0, a.warmup))

    # Every SDK setting is its default: batch size, queue size, export schedule, retries.
    provider = TracerProvider(resource=Resource.create({"service.name": a.service}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=a.otlp)))
    tracer = provider.get_tracer("lacuna.rig.replay")

    sessions = [s for s in load(a.sessions.read_text()) if s.plane == a.plane]
    timing: dict[str, float] = {}
    failed = asyncio.run(
        replay_all(sessions, http_connect(a.url), tracer, a.stagger, a.t0, timing=timing)
    )
    provider.shutdown()  # flushes what is queued; a killed process never gets here
    print(json.dumps({"plane": a.plane, "sessions": len(sessions), "failed": failed, **timing}))


if __name__ == "__main__":
    main()
