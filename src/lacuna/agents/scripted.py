"""Scripted agent and auditor: the zero-cost path through the real env.

The scripted agent follows each ticket's plan with two honest imperfections
that real agents have and that matter to witnesses:

- with probability `p_redo` it gets a customer field wrong first and then
  corrects it (two actions, one change in the end-state diff);
- with probability `p_redraft` it writes the summary twice (two actions, one
  file in the diff).

The scripted auditor stands in for the LLM auditor (W4). It reads the
transcript's successful action calls and misses each one with probability
`p_miss`, which makes it a noisy copy of what the agent did, not of what
the logs kept. Its numbers are assumptions; the LLM pilot measures the real
ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from lacuna.envs.tasks import Call, Episode
from lacuna.envs.tools import ACTION_TOOLS
from lacuna.witnesses import from_call


@dataclass
class ScriptedAgent:
    rng: np.random.Generator
    p_redo: float = 0.15
    p_redraft: float = 0.2

    def plan(self, ep: Episode) -> list[Call]:
        calls: list[Call] = []
        for t in ep.tickets:
            for tool, args in t.plan:
                if tool == "update_customer" and self.rng.random() < self.p_redo:
                    calls.append((tool, {**args, "value": args["value"] + "x"}))
                if tool == "write_file" and self.rng.random() < self.p_redraft:
                    calls.append((tool, {**args, "content": "Draft."}))
                calls.append((tool, args))
        return calls


def transcript_from_calls(calls: list[tuple[str, str, dict, str]]) -> list[dict]:
    """OpenAI-style messages for a scripted run, so both auditors read the same format."""
    msgs: list[dict] = []
    for call_id, tool, args, result in calls:
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args)},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": result})
    return msgs


@dataclass
class ScriptedAuditor:
    rng: np.random.Generator
    p_miss: float = 0.15

    def reconstruct(self, transcript: list[dict]) -> list[dict]:
        results = {m["tool_call_id"]: m["content"] for m in transcript if m["role"] == "tool"}
        out = []
        for m in transcript:
            for tc in m.get("tool_calls") or []:
                name = tc["function"]["name"]
                if name not in ACTION_TOOLS or "error" in results.get(tc["id"], ""):
                    continue
                if self.rng.random() < self.p_miss:
                    continue
                norm = from_call(name, json.loads(tc["function"]["arguments"]))
                if norm:
                    out.append({"op": norm[0], "resource": norm[1], "value": norm[2]})
        return out
