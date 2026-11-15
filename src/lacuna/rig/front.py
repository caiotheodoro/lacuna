"""An MCP front for an environment: the receiver on a plane.

It offers the env's tools over streamable HTTP, forwards each call to the kernel
service under its own key, and writes one access-log line per `tools/call`. On
plane B that line is the receiver's witness. On plane A the gateway in front
writes its own, and this one is not collected.

The reply is the router's text, with `isError` set when the call did not
succeed, so an agent sees what it would see in-process.

What the line says about the caller comes from the request, not from the tool
arguments: the session from the `x-lacuna-session` header, the trace id from
`traceparent`, the JSON-RPC id from the envelope. The client library stamps all
three (`lacuna.rig.replay`); no model output reaches them.

Needs the `rig` extra.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route

from lacuna.envs.base import Env

SESSION_HEADER = "x-lacuna-session"
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")

# (session, tool, args) -> (succeeded, text). Raises if the kernel service refuses.
Forward = Callable[[str, str, dict], Awaitable[tuple[bool, str]]]


def trace_id_of(traceparent: str | None) -> str | None:
    """The trace id of a W3C `traceparent` header, or None if it is not one."""
    m = _TRACEPARENT.match(traceparent or "")
    return m.group(1) if m else None


def build_server(env: Env, forward: Forward, log: Callable[[dict], None]) -> Server:
    server: Server = Server(f"lacuna-{env.name}")
    tools = [
        types.Tool(
            name=t["function"]["name"],
            description=t["function"]["description"],
            inputSchema=t["function"]["parameters"],
        )
        for t in env.tools
    ]

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return tools

    # No schema validation here: a bad call gets the kernel's own error text, as it
    # does in-process, and is logged like any other failed call.
    @server.call_tool(validate_input=False)
    async def _call(name: str, arguments: dict) -> types.CallToolResult:
        ctx = server.request_context
        headers = ctx.request.headers if ctx.request is not None else {}
        session = headers.get(SESSION_HEADER) or ""
        try:
            ok, text = await forward(session, name, arguments)
        except Exception as e:  # the kernel service refused, or could not be reached
            ok, text = False, json.dumps({"error": f"{type(e).__name__}: {e}"})
        log(
            {
                "ts": round(time.time(), 6),
                "session": session,
                "trace_id": trace_id_of(headers.get("traceparent")),
                "rpc_id": ctx.request_id,
                "tool": name,
                "args": arguments,
                "ok": ok,
            }
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], isError=not ok
        )

    return server


def make_app(server: Server) -> tuple[Starlette, StreamableHTTPSessionManager]:
    """The ASGI app serving `server` at `/mcp`, and its session manager.

    The app's lifespan runs the manager, which is what an ASGI server does. A caller
    that drives the app without a lifespan (a test) enters `manager.run()` itself.
    """
    manager = StreamableHTTPSessionManager(app=server)

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        async with manager.run():
            yield

    class Endpoint:
        # An object, not a function, so the route hands it the raw ASGI call. A mount
        # would answer `/mcp` with a redirect to `/mcp/`, which a gateway does not follow.
        async def __call__(self, scope, receive, send) -> None:
            await manager.handle_request(scope, receive, send)

    app = Starlette(routes=[Route("/mcp", endpoint=Endpoint())], lifespan=lifespan)
    return app, manager


def main() -> None:
    import httpx
    import uvicorn

    from lacuna.envs.registry import ENVS

    env = ENVS[os.environ.get("LACUNA_ENV", "backoffice")]
    kernel_url = os.environ["KERNEL_URL"].rstrip("/")
    key = os.environ["LACUNA_KEY"]
    http = httpx.AsyncClient(timeout=30.0)

    async def forward(session: str, tool: str, args: dict) -> tuple[bool, str]:
        r = await http.post(
            f"{kernel_url}/execute",
            headers={"x-lacuna-key": key},
            json={"session": session, "tool": tool, "args": args},
        )
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(body.get("error", f"kernel answered {r.status_code}"))
        return bool(body["ok"]), str(body["text"])

    def log(line: dict) -> None:
        # stdout is the access log; everything else this process prints goes to stderr.
        sys.stdout.write(json.dumps(line, sort_keys=True) + "\n")
        sys.stdout.flush()

    app, _ = make_app(build_server(env, forward, log))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning")


if __name__ == "__main__":
    main()
