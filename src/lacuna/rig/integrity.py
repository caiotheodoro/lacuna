"""Did a run's plumbing work? Loss must come from the telemetry components, so
anything else that went wrong makes the run unusable, not lossy.

`problems` takes facts the run driver gathered and names what is wrong. A run
with a non-empty list fails the integrity gate: it is kept, marked, and not
analysed.
"""

from __future__ import annotations

from collections.abc import Iterable

from lacuna.rig.stress import Event


def killed_for_good(schedule: Iterable[Event]) -> set[str]:
    """Services whose last scheduled event is a kill."""
    last: dict[str, str] = {}
    for e in sorted(schedule, key=lambda e: e.at_s):
        last[e.service] = e.action
    return {s for s, action in last.items() if action == "kill"}


def problems(
    *,
    schedule: list[Event],
    events_done: list[dict],
    exit_codes: dict[str, int],
    reports: dict[str, dict | None],
    file_sizes: dict[str, int],
    not_running: list[str],
    follower_lines: dict[str, int],
    full_lines: dict[str, int],
    planned_calls: dict[str, int],
    written_calls: dict[str, int],
    agent_plane: dict[str, str],
) -> list[str]:
    """What is wrong with the run, in words. Empty means the plumbing held.

    `reports` is each originator's own closing report of failed sessions (None if it
    printed none). `not_running` lists services that should have been up and were not,
    before the originators started or after the run. `follower_lines` and `full_lines`
    count each access log as followed during the run and as dumped whole at its end.
    """
    out: list[str] = []
    happened = {(d["action"], d["service"], d["at_s"]) for d in events_done if d.get("ok")}
    for e in schedule:
        if (e.action, e.service, e.at_s) not in happened:
            out.append(f"stress event did not happen: {e.action} {e.service} at {e.at_s} s")
    dead = killed_for_good(schedule)
    for agent, code in sorted(exit_codes.items()):
        want = 137 if agent in dead else 0
        if code != want:
            out.append(f"{agent} exited {code}, expected {want}")
        if agent in dead:
            continue
        report = reports.get(agent)
        if report is None:
            out.append(f"{agent} left no report")
        elif report:
            out.append(f"{agent} could not replay sessions: {sorted(report)}")
    for name, size in sorted(file_sizes.items()):
        if size <= 0:
            out.append(f"witness file {name} is missing or empty")
    for service in not_running:
        out.append(f"{service} was not running when it should have been")
    for name in sorted(full_lines):
        if follower_lines.get(name) != full_lines[name]:
            out.append(
                f"access log {name}: followed {follower_lines.get(name)} lines, "
                f"the container wrote {full_lines[name]}"
            )
    for agent, plane in sorted(agent_plane.items()):
        if agent not in dead and written_calls.get(plane) != planned_calls.get(plane):
            out.append(
                f"plane {plane}: the receiver wrote {written_calls.get(plane)} calls, "
                f"the sessions hold {planned_calls.get(plane)}"
            )
    return out
