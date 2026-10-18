"""What the harness needs to know about an environment, and nothing more.

The router, the witnesses and the agents were written against the
back-office tools. An `Env` bundles the parts of them that are specific to a
domain, so a second environment plugs in without touching the witnesses'
logic. Every function that takes one treats `None` as the back-office
environment, through the original code path, so back-office runs are
unchanged byte for byte (`tests/test_infra.py` checks it).

Which ops carry their value into the effect signature is not here: it
lives in `lacuna.linkage.VALUE_OPS`, one union over every env, because the
signature tables are built without knowing which env produced a record.
That is safe only because op names are disjoint across envs (tested).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# (op, resource, value, ledger id or None): W3's raw material, one row per
# change the end-state diff shows or per item the network sink caught.
Observed = tuple[str, str, str, int | None]


@dataclass(frozen=True)
class Env:
    name: str
    # OpenAI function schemas offered to the agent.
    tools: list[dict]
    # Tools that change state or send data out. Each successful call is one
    # ledger entry; every other tool is a read and is never ledgered.
    action_tools: tuple[str, ...]
    # kernel -> {tool name: handler(args)}. Action handlers return a ledger id.
    handlers: Callable[[Any], dict[str, Callable[[dict], object]]]
    # A tool call as the kernel ledgers it: (op, resource, value), or None.
    normalise: Callable[[str, dict], tuple[str, str, str] | None]
    # kernel -> what W3 sees: the end-state diff plus the network sink.
    observe: Callable[[Any], list[Observed]]
    # (seed, n_tickets) -> lacuna.envs.tasks.Episode
    make_episode: Callable[[int, int], Any]
    # world -> kernel
    make_kernel: Callable[[Any], Any]
    agent_system: str
