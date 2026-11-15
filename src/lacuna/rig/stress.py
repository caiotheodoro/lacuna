"""When a run kills what, as a function of a condition's name and a seed.

Only telemetry components and originators are ever killed: the collector the
SDKs export to, the node collector that tails access logs, and an originator.
Receivers, the kernel and the backend that writes the witness files are left
alone, so every lost record is lost by an unmodified component doing what it
does at default settings when something upstream or downstream of it dies.

The conditions here are the ones the rig's checkpoint uses. They are not a
registered schedule. A 20 s outage of the span relay is long enough that the
first refused batch is always given up (`docs/RIG-FINDINGS.md`, section 3), and
short enough that whether a second one is given up is chance. The study's
schedule is fixed in its registration after the pilot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    at_s: float  # seconds after the originators start
    action: str  # "kill" or "start"
    service: str


# An item with `down_s` is an outage (kill, then start); without, a kill for good.
# `at` is the range the kill time is drawn from.
CONDITIONS: dict[str, tuple[dict, ...]] = {
    "none": (),
    "gw-outage": ({"service": "otelcol-gw", "at": (4.0, 10.0), "down_s": 20.0},),
    "node-outage": ({"service": "otelcol-node", "at": (4.0, 10.0), "down_s": 10.0},),
    "agent-kill": ({"service": "agent-b", "at": (8.0, 16.0)},),
    "mixed": (
        {"service": "otelcol-gw", "at": (4.0, 10.0), "down_s": 20.0},
        {"service": "otelcol-node", "at": (4.0, 10.0), "down_s": 10.0},
        {"service": "agent-b", "at": (8.0, 16.0)},
    ),
}


def plan(condition: str, seed: int) -> list[Event]:
    """The events of one run, in time order."""
    rng = random.Random(f"{condition}/{seed}")
    events: list[Event] = []
    for item in CONDITIONS[condition]:
        at = round(rng.uniform(*item["at"]), 3)
        events.append(Event(at, "kill", item["service"]))
        if "down_s" in item:
            events.append(Event(round(at + item["down_s"], 3), "start", item["service"]))
    return sorted(events, key=lambda e: e.at_s)
