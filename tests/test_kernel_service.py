"""The kernel behind a service: same results as the in-process router, plus who acted."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from lacuna.agents.scripted import ScriptedAgent
from lacuna.envs.kernel import Kernel, LedgerEntry
from lacuna.envs.registry import BACKOFFICE
from lacuna.envs.service import KernelService, ServiceError, serve
from lacuna.envs.tasks import make_episode
from lacuna.envs.tools import Router

KEYS = {"ka": "plane-a", "kb": "plane-b"}


def _service() -> KernelService:
    return KernelService(BACKOFFICE, KEYS)


def _plan(seed: int):
    ep = make_episode(seed, 7)
    return ScriptedAgent(np.random.default_rng([seed, 1])).plan(ep)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_results_and_ledger_equal_the_in_process_router(seed):
    plan = _plan(seed)
    # A call that fails a business rule and one with bad arguments, as an agent makes.
    plan = plan + [("cancel_order", {"order_id": "nope"}), ("issue_refund", {"order_id": "O1"})]
    kernel = Kernel(make_episode(seed, 7).world)
    router = Router(kernel, np.random.default_rng(0), 0.0, 0.0)
    svc = _service()
    svc.open("s", seed, 7)
    for i, (tool, args) in enumerate(plan):
        want = router.execute(f"c{i}", tool, args)
        ok, text = svc.execute("s", "ka", tool, args)
        assert text == want
        assert ok == router.gateway[-1]["ok"]
    got = [(e["id"], e["op"], e["resource"], e["value"]) for e in svc.ledger("s")]
    assert got == [(e.id, e.op, e.resource, e.value) for e in kernel.ledger]
    assert len(got) > 5


def test_every_ledger_entry_names_the_principal_that_made_it():
    svc = _service()
    svc.open("s", 0, 7)
    made = []
    for i, (tool, args) in enumerate(_plan(0)):
        key = "ka" if i % 2 == 0 else "kb"
        ok, _ = svc.execute("s", key, tool, args)
        if ok and tool in BACKOFFICE.action_tools:
            made.append(KEYS[key])
    assert [e["principal"] for e in svc.ledger("s")] == made
    assert set(made) == {"plane-a", "plane-b"}


def test_an_env_row_carries_the_last_writer_and_an_overwritten_action_leaves_none():
    svc = _service()
    svc.open("s", 0, 7)
    a = {"customer_id": "C1", "field": "address", "value": "1 Elm St"}
    assert svc.execute("s", "ka", "update_customer", a)[0]
    assert svc.execute("s", "kb", "update_customer", {**a, "value": "2 Oak Ave"})[0]
    assert svc.execute("s", "ka", "cancel_order", {"order_id": _processing(svc)})[0]
    rows = svc.observe("s")
    addr = [r for r in rows if r["resource"] == "C1.address"]
    assert addr == [
        {
            "op": "update_customer",
            "resource": "C1.address",
            "value": "2 Oak Ave",
            "principal": "plane-b",
        }
    ]
    assert [r["principal"] for r in rows if r["op"] == "cancel_order"] == ["plane-a"]
    # Two actions on one field, one row: the snapshot's blind spot, not a simulated one.
    assert len(svc.ledger("s")) == 3 and len(rows) == 2
    assert all(set(r) == {"op", "resource", "value", "principal"} for r in rows)


def _processing(svc: KernelService) -> str:
    world = make_episode(0, 7).world
    return next(k for k, v in world.orders.items() if v["status"] == "processing")


def test_a_failed_call_is_not_an_action():
    svc = _service()
    svc.open("s", 0, 7)
    ok, text = svc.execute("s", "ka", "cancel_order", {"order_id": "nope"})
    assert not ok and "error" in json.loads(text)
    ok, text = svc.execute("s", "ka", "no_such_tool", {})
    assert not ok and "unknown tool" in text
    assert svc.ledger("s") == []


def test_unknown_session_unknown_key_and_a_second_open_are_refused():
    svc = _service()
    svc.open("s", 0, 7)
    with pytest.raises(ServiceError, match="no session"):
        svc.execute("other", "ka", "get_order", {"order_id": "O1"})
    with pytest.raises(ServiceError, match="key"):
        svc.execute("s", "nope", "get_order", {"order_id": "O1"})
    with pytest.raises(ServiceError, match="already open"):
        svc.open("s", 1, 7)
    with pytest.raises(ServiceError, match="no session"):
        svc.ledger("other")


def test_sessions_do_not_share_a_world():
    svc = _service()
    svc.open("s1", 0, 7)
    svc.open("s2", 0, 7)
    a = {"customer_id": "C1", "field": "tier", "value": "gold-x"}
    svc.execute("s1", "ka", "update_customer", a)
    assert len(svc.ledger("s1")) == 1 and svc.ledger("s2") == []
    assert svc.sessions() == ["s1", "s2"]


def test_concurrent_calls_on_one_session_keep_the_ledger_whole(monkeypatch):
    # Widen the kernel's own race (read the ledger's length, then append) so that
    # without the session lock two threads take the same id nearly every time.
    def slow_record(self, op, resource, value):
        entry = LedgerEntry(len(self.ledger), op, resource, value, self.step)
        time.sleep(0.0005)
        self.ledger.append(entry)
        return entry.id

    monkeypatch.setattr(Kernel, "_record", slow_record)
    svc = _service()
    svc.open("s", 0, 7)
    n_threads, per = 8, 10

    def work(t: int) -> None:
        for i in range(per):
            args = {"path": f"/t{t}/f{i}.txt", "content": f"{t}-{i}"}
            assert svc.execute("s", "ka" if t % 2 else "kb", "write_file", args)[0]

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ledger = svc.ledger("s")
    assert [e["id"] for e in ledger] == list(range(n_threads * per))
    assert len({e["resource"] for e in ledger}) == n_threads * per
    for e in ledger:
        assert e["principal"] == ("plane-a" if e["resource"][2] in "1357" else "plane-b")


def _call(url: str, body: dict | None = None, key: str | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json", **({"x-lacuna-key": key} if key else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_http_shell_round_trips_and_keeps_truth_behind_the_admin_key():
    server = serve(_service(), "127.0.0.1", 0, admin_key="adm")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert _call(f"{base}/health") == (200, {"ok": True})
        opened = {"session": "s", "seed": 0, "n_tickets": 7}
        assert _call(f"{base}/open", opened, key="ka")[0] == 403
        assert _call(f"{base}/open", opened, key="adm") == (200, {"ok": True})
        a = {"customer_id": "C1", "field": "tier", "value": "gold-x"}
        call = {"session": "s", "tool": "update_customer", "args": a}
        assert _call(f"{base}/execute", call, key="ka") == (
            200,
            {"ok": True, "text": '{"status": "done"}'},
        )
        assert _call(f"{base}/execute", call, key="nope")[0] == 403
        assert _call(f"{base}/execute", call)[0] == 403
        assert _call(f"{base}/execute", {**call, "session": "other"}, key="ka")[0] == 404
        # A plane's key cannot read the truth or the env witness.
        assert _call(f"{base}/ledger?session=s", key="ka")[0] == 403
        status, body = _call(f"{base}/ledger?session=s", key="adm")
        assert status == 200 and [e["principal"] for e in body["ledger"]] == ["plane-a"]
        status, body = _call(f"{base}/observe?session=s", key="adm")
        assert status == 200 and body["rows"][0]["principal"] == "plane-a"
        assert _call(f"{base}/sessions", key="adm") == (200, {"sessions": ["s"]})
        assert _call(f"{base}/nothing", key="adm")[0] == 404
        # The clock route needs no key and tells the time of the host it runs on.
        before = time.time()
        status, body = _call(f"{base}/time")
        assert status == 200 and before <= body["unix"] <= time.time()
    finally:
        server.shutdown()
        server.server_close()


def test_the_http_shell_answers_malformed_requests_with_400():
    server = serve(_service(), "127.0.0.1", 0, admin_key="adm")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def raw(path: str, data: bytes, key: str = "adm") -> int:
        req = urllib.request.Request(f"{base}{path}", data=data, headers={"x-lacuna-key": key})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert raw("/open", b"{not json") == 400
        assert raw("/open", b"[1, 2]") == 400
        assert raw("/open", b'{"session": "s"}') == 400  # no seed
        assert raw("/open", b'{"session": "s", "seed": "x", "n_tickets": 7}') == 400
        assert raw("/open", b'{"session": "s", "seed": 0, "n_tickets": 7}') == 200
        assert raw("/open", b'{"session": "s", "seed": 0, "n_tickets": 7}') == 409
        assert raw("/execute", b'{"session": "s"}', key="ka") == 400  # no tool
        assert raw("/execute", b'{"session": "s", "tool": "get_order", "args": [1]}', "ka") == 400
        assert _call(f"{base}/ledger?session=s", key="adm") == (200, {"ledger": []})
    finally:
        server.shutdown()
        server.server_close()


def test_a_bad_content_length_is_answered_and_does_not_hang_the_server():
    server = serve(_service(), "127.0.0.1", 0, admin_key="adm")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def status(length: str) -> str:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
            head = "POST /open HTTP/1.1\r\nHost: x\r\nx-lacuna-key: adm\r\n"
            sock.sendall((head + f"Content-Length: {length}\r\n\r\n").encode())
            return sock.recv(64).decode().split("\r\n")[0]

    try:
        assert " 400 " in status("-5")  # read as empty: no session in the body
        assert " 413 " in status(str(50 * 1024 * 1024))
        assert " 400 " in status("abc")
        assert _call(f"http://127.0.0.1:{port}/health") == (200, {"ok": True})
    finally:
        server.shutdown()
        server.server_close()
