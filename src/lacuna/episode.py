"""Run one episode end to end and collect every witness's records.

The ledger is copied out for scoring and nothing else: estimators only ever
receive count tables built from witness records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lacuna.agents.scripted import ScriptedAgent, transcript_from_calls
from lacuna.envs.kernel import Kernel, LedgerEntry
from lacuna.envs.tasks import make_episode
from lacuna.envs.tools import Router
from lacuna.witnesses import (
    Record,
    auditor_records,
    env_records,
    gateway_records,
    trace_records,
)


@dataclass
class EpisodeResult:
    seed: int
    ledger: list[LedgerEntry]
    records: list[Record]
    n_calls: int
    n_tool_errors: int
    transcript: list[dict]

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "ledger": [asdict(e) for e in self.ledger],
            "records": [asdict(r) for r in self.records],
            "n_calls": self.n_calls,
            "n_tool_errors": self.n_tool_errors,
            # Stored so a different W4 can be scored without paying for the
            # agent again (docs/DECISIONS.md D14).
            "transcript": self.transcript,
        }


def run_episode(
    seed: int,
    agent,
    auditor,
    p_gateway_drop: float = 0.05,
    p_trace_drop: float = 0.10,
    n_tickets: int = 7,
    max_turns: int = 40,
) -> EpisodeResult:
    """`agent` is a ScriptedAgent or a ChatClient; `auditor` has `.reconstruct(transcript)`."""
    ep = make_episode(seed, n_tickets)
    kernel = Kernel(ep.world)
    # Logging faults draw from their own stream, so they do not depend on
    # which agent ran or how many calls it made before.
    router = Router(kernel, np.random.default_rng([seed, 7]), p_gateway_drop, p_trace_drop)

    if isinstance(agent, ScriptedAgent):
        done = []
        for i, (tool, args) in enumerate(agent.plan(ep)):
            cid = f"call_{i}"
            done.append((cid, tool, args, router.execute(cid, tool, args)))
        transcript = transcript_from_calls(done)
    else:
        from lacuna.agents.llm import run_agent

        transcript = run_agent(agent, ep, router, max_turns)

    calls = [tc for m in transcript for tc in (m.get("tool_calls") or [])]
    errors = sum(1 for m in transcript if m["role"] == "tool" and '"error"' in m["content"])
    records = (
        trace_records(kernel, router.call_ledger)
        + gateway_records(router.gateway, router.call_ledger)
        + env_records(kernel)
        + auditor_records(auditor.reconstruct(transcript))
    )
    return EpisodeResult(seed, list(kernel.ledger), records, len(calls), errors, transcript)
