"""The kernel behind a network service, for runs where callers are separate processes.

One kernel per session. Every call goes through the same `Router` an in-process
episode uses, with its two loss rates at zero, so a tool's reply is the router's
byte for byte. What the service adds is *who*: each caller presents a key, a key
maps to a principal, and every ledger entry made by a call is stamped with it.

Two things read that stamp. The ledger (truth, admin only) carries it for
scoring. The env witness carries it as the last writer of each row it shows,
which is what a "last modified by" column gives an auditor, and it is how an env
row gets a plane when no log line names one. The env witness still reads the
end state and the sink, never the ledger (AGENTS.md rule 3).

The HTTP shell is the standard library's. Keys are passed in by the caller of
`serve`; none is in the tree (rule 6).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from lacuna.envs.base import Env
from lacuna.envs.tools import Router


class ServiceError(Exception):
    """A refused request: `status` is what the HTTP shell answers with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class _Session:
    router: Router
    lock: threading.Lock = field(default_factory=threading.Lock)
    principal_of: dict[int, str] = field(default_factory=dict)  # ledger id -> principal


class KernelService:
    def __init__(self, env: Env, keys: dict[str, str]):
        """`keys` maps a caller's key to its principal (one per plane)."""
        self.env = env
        self.keys = dict(keys)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def open(self, session: str, seed: int, n_tickets: int) -> None:
        ep = self.env.make_episode(seed, n_tickets)
        # Loss rates are zero, so the router's generator is never consulted for a drop.
        kernel = self.env.make_kernel(ep.world)
        router = Router(kernel, np.random.default_rng(0), 0.0, 0.0, env=self.env)
        with self._lock:
            if session in self._sessions:
                raise ServiceError(f"session {session} is already open", 409)
            self._sessions[session] = _Session(router)

    def sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def _get(self, session: str) -> _Session:
        with self._lock:
            s = self._sessions.get(session)
        if s is None:
            raise ServiceError(f"no session {session}", 404)
        return s

    def execute(self, session: str, key: str, tool: str, args: dict) -> tuple[bool, str]:
        """One tool call. Returns (succeeded, the text the agent sees)."""
        principal = self.keys.get(key)
        if principal is None:
            raise ServiceError("unknown key", 403)
        s = self._get(session)
        with s.lock:
            kernel = s.router.kernel
            before = len(kernel.ledger)
            text = s.router.execute(f"svc{len(s.router.gateway)}", tool, args)
            for entry in kernel.ledger[before:]:
                s.principal_of[entry.id] = principal
            # The router returns only the text. Its gateway log has the outcome, and with
            # the drop rate at zero the last line is always this call's.
            return bool(s.router.gateway[-1]["ok"]), text

    def ledger(self, session: str) -> list[dict]:
        """Ground truth, with the principal of each entry. For scoring only."""
        s = self._get(session)
        with s.lock:
            return [
                {**asdict(e), "principal": s.principal_of[e.id]} for e in s.router.kernel.ledger
            ]

    def observe(self, session: str) -> list[dict]:
        """The env witness: what the end state and the sink show, and who wrote it last."""
        s = self._get(session)
        with s.lock:
            return [
                {
                    "op": op,
                    "resource": resource,
                    "value": value,
                    "principal": s.principal_of.get(lid) if lid is not None else None,
                }
                for op, resource, value, lid in self.env.observe(s.router.kernel)
            ]


def serve(service: KernelService, host: str, port: int, admin_key: str) -> ThreadingHTTPServer:
    """An HTTP server over `service`, not yet serving. Port 0 picks a free one."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _admin(self) -> None:
            if self.headers.get("x-lacuna-key") != admin_key:
                raise ServiceError("admin only", 403)

        def _route(self, body: dict[str, Any] | None) -> dict:
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/health":
                return {"ok": True}
            if url.path == "/execute" and body is not None:
                ok, text = service.execute(
                    str(body["session"]),
                    self.headers.get("x-lacuna-key") or "",
                    str(body["tool"]),
                    dict(body.get("args") or {}),
                )
                return {"ok": ok, "text": text}
            if url.path == "/open" and body is not None:
                self._admin()
                service.open(str(body["session"]), int(body["seed"]), int(body["n_tickets"]))
                return {"ok": True}
            if url.path == "/sessions":
                self._admin()
                return {"sessions": service.sessions()}
            if url.path == "/ledger":
                self._admin()
                return {"ledger": service.ledger(query.get("session", ""))}
            if url.path == "/observe":
                self._admin()
                return {"rows": service.observe(query.get("session", ""))}
            raise ServiceError(f"no route {url.path}", 404)

        def _handle(self, body: dict | None) -> None:
            try:
                self._reply(200, self._route(body))
            except ServiceError as e:
                self._reply(e.status, {"error": str(e)})
            except (KeyError, TypeError, ValueError) as e:
                self._reply(400, {"error": f"bad request: {e}"})

        def do_GET(self) -> None:  # noqa: N802 (the base class's name)
            self._handle(None)

        def do_POST(self) -> None:  # noqa: N802
            try:
                n = max(0, int(self.headers.get("content-length") or 0))
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self._reply(400, {"error": "body is not JSON"})
                return
            if not isinstance(body, dict):
                self._reply(400, {"error": "body is not a JSON object"})
                return
            self._handle(body)

        def log_message(self, *_: object) -> None:
            """Quiet: this service's own request log is not a witness."""

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    from lacuna.envs.registry import ENVS

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", default="backoffice", choices=sorted(ENVS))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()
    # LACUNA_KEYS is "key=principal,key=principal"; both variables are required.
    keys = dict(pair.split("=", 1) for pair in os.environ["LACUNA_KEYS"].split(","))
    server = serve(KernelService(ENVS[a.env], keys), a.host, a.port, os.environ["LACUNA_ADMIN_KEY"])
    server.serve_forever()


if __name__ == "__main__":
    main()
