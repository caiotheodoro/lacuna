"""Did a run's plumbing work? Loss must come from the telemetry components, so
anything else that went wrong makes the run unusable, not lossy.

`problems` takes facts the run driver gathered and names what is wrong.
`file_problems` does the same from a run's summary alone, so it can be applied
again to a run made earlier. A run with a non-empty list from either fails the
integrity gate: it is kept, marked, and not analysed.
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
    timing: dict[str, dict | None] | None = None,
    clock_offset_s: float | None = None,
    max_lag_s: float = 0.25,
) -> list[str]:
    """What is wrong with the run, in words. Empty means the plumbing held.

    `reports` is each originator's own closing report of failed sessions (None if it
    printed none). `not_running` lists services that should have been up and were not,
    before the originators started or after the run. `follower_lines` and `full_lines`
    count each access log as followed during the run and as dumped whole at its end.

    `timing` is each originator's own `start_lag_s` and `max_lag_s`
    (`lacuna.rig.replay`): a session's planes keep their order, and the stress
    schedule falls where it was drawn, only if calls are made when they are due.
    `clock_offset_s` is the containers' clock minus the driver's, which times the
    stress events. A run made before these were recorded passes None and is not
    checked for them.
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
        if service not in dead:
            out.append(f"{service} was not running when it should have been")
    for name in sorted(full_lines):
        if follower_lines.get(name) != full_lines[name]:
            out.append(
                f"access log {name}: followed {follower_lines.get(name)} lines, "
                f"the container wrote {full_lines[name]}"
            )
    for agent, t in sorted((timing or {}).items()):
        if agent in dead or t is None:
            continue
        if t.get("start_lag_s", 0.0) > 0:
            out.append(f"{agent} was ready {t['start_lag_s']} s after the run's start")
        if t.get("max_lag_s", 0.0) > max_lag_s:
            out.append(f"{agent} made a call {t['max_lag_s']} s late (limit {max_lag_s} s)")
    if clock_offset_s is not None and abs(clock_offset_s) > 2 * max_lag_s:
        out.append(
            f"the containers' clock is {clock_offset_s} s from the driver's, "
            "which times the stress events"
        )
    for agent, plane in sorted(agent_plane.items()):
        if agent not in dead and written_calls.get(plane) != planned_calls.get(plane):
            out.append(
                f"plane {plane}: the receiver wrote {written_calls.get(plane)} calls, "
                f"the sessions hold {planned_calls.get(plane)}"
            )
    return out


def file_problems(summary: dict, dead_planes: set[str]) -> list[str]:
    """What a run's summary (`lacuna.rig.witnesses.summarise`) shows to be wrong.

    `dead_planes` are the planes whose originator was killed for good. Such an
    originator can die between the kernel acting and the receiver confirming, which
    leaves an action no receiver line calls a success; that is a consequence of the
    kill, is counted in the summary, and is not a fault. The reverse is always one.
    """
    out: list[str] = []
    for plane, p in sorted(summary["planes"].items()):
        if p["records_without_session"]:
            out.append(
                f"plane {plane}: {p['records_without_session']} telemetry records carry "
                "no session (a hop dropped the header)"
            )
        for w in ("spans", "access"):
            n = p["witnesses"][w]["matching_no_action"]
            if n:
                out.append(f"plane {plane}: {n} {w} records match no ledger entry")
        if p["calls_written_by_receiver"] and not p["access_lines_delivered"]:
            out.append(f"plane {plane}: the log relay delivered no line of its access log")
        got, want = p["receiver_ok_actions"], p["actions_in_ledger"]
        if got > want or (got < want and plane not in dead_planes):
            out.append(
                f"plane {plane}: the receiver confirmed {got} actions, the ledger holds {want}"
            )
    return out
