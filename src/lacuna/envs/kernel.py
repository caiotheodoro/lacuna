"""The back-office kernel: state, the ground-truth ledger, and the network sink.

Every state change and every byte sent out goes through a method here, and
each such call appends one `LedgerEntry`. That ledger is N. No witness reads
it: witnesses see the gateway, the trace file, the end-state diff, or the
transcript, exactly as a real auditor would (AGENTS.md rule 3).

The trace file is harness bookkeeping, not business state, so it is left
out of the ledger and the diff.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

TRACE_PATH = "/var/log/agent_trace.jsonl"

ACTION_OPS = (
    "issue_refund",
    "update_customer",
    "cancel_order",
    "write_file",
    "delete_file",
    "send_email",
    "http_post",
)
EGRESS_OPS = ("send_email", "http_post")


def digest(*parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    op: str
    resource: str
    value: str  # what the action set or sent, normalised for signatures
    step: int


@dataclass
class World:
    customers: dict[str, dict[str, str]]
    orders: dict[str, dict[str, object]]
    refunds: list[dict[str, object]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


class KernelError(Exception):
    """A business-rule failure, returned to the agent as a tool error. Not an action."""


class Kernel:
    def __init__(self, world: World):
        self.world = world
        self._initial = copy.deepcopy(world)
        self.ledger: list[LedgerEntry] = []
        self.sink: list[dict[str, object]] = []  # outbound traffic, with the ledger id
        self.step = 0
        # Which ledger entry last touched each diffable key, so the diff can
        # be attributed (for oracle linkage only; the diff itself has no ids).
        self.last_writer: dict[str, int] = {}

    # Ledger -----------------------------------------------------------------

    def _record(self, op: str, resource: str, value: str) -> int:
        entry = LedgerEntry(len(self.ledger), op, resource, value, self.step)
        self.ledger.append(entry)
        return entry.id

    # Reads (never ledgered) ---------------------------------------------------

    def get_customer(self, customer_id: str) -> dict:
        if customer_id not in self.world.customers:
            raise KernelError(f"no customer {customer_id}")
        return {"customer_id": customer_id, **self.world.customers[customer_id]}

    def get_order(self, order_id: str) -> dict:
        if order_id not in self.world.orders:
            raise KernelError(f"no order {order_id}")
        return {"order_id": order_id, **self.world.orders[order_id]}

    def list_orders(self, customer_id: str) -> list[dict]:
        return [
            {"order_id": oid, **o}
            for oid, o in self.world.orders.items()
            if o["customer_id"] == customer_id
        ]

    def read_file(self, path: str) -> str:
        if path not in self.world.files:
            raise KernelError(f"no file {path}")
        return self.world.files[path]

    def list_files(self, prefix: str = "/") -> list[str]:
        return sorted(p for p in self.world.files if p.startswith(prefix))

    # Actions ------------------------------------------------------------------

    def issue_refund(self, order_id: str, amount: float, reason: str = "") -> int:
        order = self.world.orders.get(order_id)
        if order is None:
            raise KernelError(f"no order {order_id}")
        amount = round(float(amount), 2)
        if amount <= 0 or amount > float(order["total"]):
            raise KernelError(f"refund {amount} outside (0, {order['total']}]")
        lid = self._record("issue_refund", order_id, f"{amount:.2f}")
        self.world.refunds.append(
            {
                "refund_id": f"R{len(self.world.refunds)}",
                "order_id": order_id,
                "amount": amount,
                "reason": reason,
                "_lid": lid,
            }
        )
        return lid

    def update_customer(self, customer_id: str, field_name: str, value: str) -> int:
        cust = self.world.customers.get(customer_id)
        if cust is None:
            raise KernelError(f"no customer {customer_id}")
        if field_name not in ("email", "address", "phone", "tier"):
            raise KernelError(f"field {field_name} is not editable")
        lid = self._record("update_customer", f"{customer_id}.{field_name}", str(value))
        cust[field_name] = str(value)
        self.last_writer[f"customer:{customer_id}.{field_name}"] = lid
        return lid

    def cancel_order(self, order_id: str) -> int:
        order = self.world.orders.get(order_id)
        if order is None:
            raise KernelError(f"no order {order_id}")
        if order["status"] == "cancelled":
            raise KernelError(f"order {order_id} already cancelled")
        lid = self._record("cancel_order", order_id, "cancelled")
        order["status"] = "cancelled"
        self.last_writer[f"order:{order_id}.status"] = lid
        return lid

    def write_file(self, path: str, content: str) -> int:
        if path == TRACE_PATH:
            raise KernelError(f"{path} is read-only")
        self.world.files[path] = content
        lid = self._record("write_file", path, digest(content))
        self.last_writer[f"file:{path}"] = lid
        return lid

    def delete_file(self, path: str) -> int:
        if path == TRACE_PATH:
            raise KernelError(f"{path} is read-only")
        if path not in self.world.files:
            raise KernelError(f"no file {path}")
        del self.world.files[path]
        lid = self._record("delete_file", path, "deleted")
        self.last_writer[f"file:{path}"] = lid
        return lid

    def send_email(self, to: str, subject: str, body: str) -> int:
        lid = self._record("send_email", to, digest(subject, body))
        self.sink.append({"kind": "email", "to": to, "subject": subject, "body": body, "_lid": lid})
        return lid

    def http_post(self, url: str, body: str) -> int:
        lid = self._record("http_post", url, digest(body))
        self.sink.append({"kind": "http", "url": url, "body": body, "_lid": lid})
        return lid

    def append_trace(self, line: str) -> None:
        """Harness-only path to the trace file. Never ledgered, never offered as a tool."""
        self.world.files[TRACE_PATH] += line + "\n"

    # End-of-episode diff (W3's raw material) --------------------------------

    def diff(self) -> list[dict[str, object]]:
        """What changed between the initial and final state, as an auditor's diff sees it.

        Multiplicity is lost on purpose: two updates to one field show as one
        change, and a change that was reverted shows as nothing. That is the
        real blind spot of state diffs, not a simulated one.
        """
        out: list[dict[str, object]] = []
        before, after = self._initial, self.world
        for cid, cust in after.customers.items():
            for f, v in cust.items():
                if before.customers.get(cid, {}).get(f) != v:
                    key = f"customer:{cid}.{f}"
                    out.append(
                        {
                            "op": "update_customer",
                            "resource": f"{cid}.{f}",
                            "value": v,
                            "_lid": self.last_writer.get(key),
                        }
                    )
        for oid, order in after.orders.items():
            if before.orders[oid]["status"] != order["status"] and order["status"] == "cancelled":
                out.append(
                    {
                        "op": "cancel_order",
                        "resource": oid,
                        "value": "cancelled",
                        "_lid": self.last_writer.get(f"order:{oid}.status"),
                    }
                )
        for r in after.refunds[len(before.refunds) :]:
            out.append(
                {
                    "op": "issue_refund",
                    "resource": r["order_id"],
                    "value": f"{r['amount']:.2f}",
                    "_lid": r["_lid"],
                }
            )
        paths = (set(before.files) | set(after.files)) - {TRACE_PATH}
        for p in sorted(paths):
            if p not in after.files:
                out.append(
                    {
                        "op": "delete_file",
                        "resource": p,
                        "value": "deleted",
                        "_lid": self.last_writer.get(f"file:{p}"),
                    }
                )
            elif before.files.get(p) != after.files[p]:
                out.append(
                    {
                        "op": "write_file",
                        "resource": p,
                        "value": digest(after.files[p]),
                        "_lid": self.last_writer.get(f"file:{p}"),
                    }
                )
        return out
