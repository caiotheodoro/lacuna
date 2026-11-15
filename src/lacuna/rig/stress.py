"""When a run kills what, as a function of a condition's name and a seed.

Only telemetry components and originators are ever killed: the collector the
SDKs export to, the node collector that tails access logs, and an originator.
Receivers, the kernel and the backend that writes the witness files are left
alone, so every lost record is lost by an unmodified component doing what it
does at default settings when something upstream or downstream of it dies.

The first five conditions are the ones the rig's checkpoint used. A 20 s outage
of the span relay is long enough that the first refused batch is always given up
(`docs/RIG-FINDINGS.md`, section 3), and short enough that whether a second one
is given up is chance.

The next five are the pilot's, and none of those is a registered schedule. They
exist because an action is in no telemetry log only where the two pipelines' losses
overlap:

- `node-down`: both relays die together and come back together, as when the node
  that hosts them restarts; long enough to cost the span pipeline two batches;
- `node-blip`: the same, shorter: long enough to cost one batch for certain and
  short enough that a second is given up only by chance;
- `node-restart`: the same at any length from the blip's to the long one's, at any
  time in a longer run, so the chance bands are sampled and not avoided;
- `relays`: each relay has its own outage, which may or may not overlap;
- `node-down-kill`: `node-down`, and one originator is killed for good.

`s0`, `s1` and `s2` are the study's own conditions, as the registration names them
(`docs/PREREG-STUDY5.md`): no stress; a restart of both relays (the pilot's
`node-restart` under the name the registration uses, with its own draws); and the
heavy one, each relay with its own long outage and one originator killed for good.
They are fixed by the registration's freeze, not by this file's history. For a few
hours `s1` was a restart of 28 s or more with the shorter ones split off as `s3`; a
refutation showed that line was drawn around the pilot's misses, and it was withdrawn
before any run was made under it (D85).

Times count from the run's start, the moment both originators are given.
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
# `at` is the range the kill time is drawn from. `down_s` is a length or a range one
# length is drawn from. `services` die and return together; `one_of` picks one.
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
    "node-down": (
        {"services": ("otelcol-gw", "otelcol-node"), "at": (8.0, 30.0), "down_s": (28.0, 45.0)},
    ),
    "node-blip": (
        {"services": ("otelcol-gw", "otelcol-node"), "at": (8.0, 40.0), "down_s": (15.0, 24.0)},
    ),
    "node-restart": (
        {"services": ("otelcol-gw", "otelcol-node"), "at": (8.0, 90.0), "down_s": (15.0, 45.0)},
    ),
    "relays": (
        {"service": "otelcol-gw", "at": (8.0, 40.0), "down_s": (25.0, 45.0)},
        {"service": "otelcol-node", "at": (8.0, 40.0), "down_s": (10.0, 40.0)},
    ),
    "node-down-kill": (
        {"services": ("otelcol-gw", "otelcol-node"), "at": (8.0, 30.0), "down_s": (28.0, 45.0)},
        {"one_of": ("agent-a", "agent-b"), "at": (15.0, 50.0)},
    ),
    "s0": (),
    "s1": (
        {"services": ("otelcol-gw", "otelcol-node"), "at": (8.0, 90.0), "down_s": (15.0, 45.0)},
    ),
    "s2": (
        {"service": "otelcol-gw", "at": (8.0, 70.0), "down_s": (25.0, 45.0)},
        {"service": "otelcol-node", "at": (8.0, 70.0), "down_s": (10.0, 40.0)},
        {"one_of": ("agent-a", "agent-b"), "at": (15.0, 90.0)},
    ),
}


def services_of(item: dict) -> tuple[str, ...]:
    """Every service an item can touch."""
    return tuple(item.get("services") or item.get("one_of") or (item["service"],))


def misfit(events: list[Event], planned_end_s: float) -> list[Event]:
    """Kills drawn for when the run's last call is already due: stress that hits nothing.

    A restart drawn for after the end is not one: the outage began inside the run and
    lasts to its end.
    """
    return [e for e in events if e.action == "kill" and e.at_s >= planned_end_s]


def plan(condition: str, seed: int) -> list[Event]:
    """The events of one run, in time order."""
    rng = random.Random(f"{condition}/{seed}")
    events: list[Event] = []
    for item in CONDITIONS[condition]:
        at = round(rng.uniform(*item["at"]), 3)
        services = services_of(item)
        if "one_of" in item:
            services = (rng.choice(item["one_of"]),)
        down = item.get("down_s")
        if isinstance(down, tuple):
            down = round(rng.uniform(*down), 3)
        for service in services:
            events.append(Event(at, "kill", service))
            if down is not None:
                events.append(Event(round(at + down, 3), "start", service))
    return sorted(events, key=lambda e: e.at_s)
