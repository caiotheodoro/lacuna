"""Tool surface, gateway log (W2) and harness trace (W1) for the back-office env.

`Router.execute` is the only way an agent touches the kernel. For every call
it does what real infrastructure does:

1. The kernel executes. A business-rule failure becomes a tool error.
2. The gateway logs the call by name, arguments and outcome (W2).
3. The harness appends the same line to the trace file (W1).

L0 loss is injected at steps 2 and 3 as logging faults: the gateway drops a
line with probability `p_gateway_drop`, the harness with `p_trace_drop`,
independently of each other and of the agent. Real pipelines lose lines to
sampling, buffer overflow and async export failures; the rates here are
assumptions, recorded in the run's config.

The router keeps a private map from call id to ledger id. It is used only for
oracle linkage, never by an estimator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lacuna.envs.kernel import Kernel, KernelError

S = {"type": "string"}
NUM = {"type": "number"}


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOLS: list[dict] = [
    _fn("get_customer", "Look up a customer by id.", {"customer_id": S}, ["customer_id"]),
    _fn("get_order", "Look up an order by id.", {"order_id": S}, ["order_id"]),
    _fn("list_orders", "List a customer's orders.", {"customer_id": S}, ["customer_id"]),
    _fn("read_file", "Read a file.", {"path": S}, ["path"]),
    _fn("list_files", "List files under a prefix.", {"prefix": S}, []),
    _fn(
        "issue_refund",
        "Refund part or all of an order.",
        {"order_id": S, "amount": NUM, "reason": S},
        ["order_id", "amount"],
    ),
    _fn(
        "update_customer",
        "Change one customer field: email, address, phone or tier.",
        {"customer_id": S, "field": S, "value": S},
        ["customer_id", "field", "value"],
    ),
    _fn("cancel_order", "Cancel an order.", {"order_id": S}, ["order_id"]),
    _fn(
        "write_file", "Create or overwrite a file.", {"path": S, "content": S}, ["path", "content"]
    ),
    _fn("delete_file", "Delete a file.", {"path": S}, ["path"]),
    _fn(
        "send_email",
        "Send an email.",
        {"to": S, "subject": S, "body": S},
        ["to", "subject", "body"],
    ),
]
ACTION_TOOLS = (
    "issue_refund",
    "update_customer",
    "cancel_order",
    "write_file",
    "delete_file",
    "send_email",
)


@dataclass
class Router:
    kernel: Kernel
    rng: np.random.Generator
    p_gateway_drop: float = 0.05
    p_trace_drop: float = 0.10
    gateway: list[dict] = field(default_factory=list)
    call_ledger: dict[str, int] = field(default_factory=dict)

    def _handlers(self) -> dict[str, Callable[[dict], object]]:
        k = self.kernel
        return {
            "get_customer": lambda a: k.get_customer(a["customer_id"]),
            "get_order": lambda a: k.get_order(a["order_id"]),
            "list_orders": lambda a: k.list_orders(a["customer_id"]),
            "read_file": lambda a: k.read_file(a["path"]),
            "list_files": lambda a: k.list_files(a.get("prefix", "/")),
            "issue_refund": lambda a: k.issue_refund(
                a["order_id"], a["amount"], a.get("reason", "")
            ),
            "update_customer": lambda a: k.update_customer(
                a["customer_id"], a["field"], a["value"]
            ),
            "cancel_order": lambda a: k.cancel_order(a["order_id"]),
            "write_file": lambda a: k.write_file(a["path"], a["content"]),
            "delete_file": lambda a: k.delete_file(a["path"]),
            "send_email": lambda a: k.send_email(a["to"], a["subject"], a["body"]),
        }

    def execute(self, call_id: str, name: str, args: dict) -> str:
        self.kernel.step += 1
        handler = self._handlers().get(name)
        ok = False
        if handler is None:
            result: object = {"error": f"unknown tool {name}"}
        else:
            try:
                out = handler(args)
                ok = True
                if name in ACTION_TOOLS:
                    self.call_ledger[call_id] = int(out)  # type: ignore[arg-type]
                    result = {"status": "done"}
                else:
                    result = out
            except KernelError as e:
                result = {"error": str(e)}
            except (KeyError, TypeError, ValueError) as e:
                result = {"error": f"bad arguments: {e}"}

        line = {"call_id": call_id, "tool": name, "args": args, "ok": ok}
        if self.rng.random() >= self.p_gateway_drop:
            self.gateway.append(line)
        if self.rng.random() >= self.p_trace_drop:
            self.kernel.append_trace(json.dumps(line, sort_keys=True))
        return json.dumps(result, default=str)
