"""The stress schedule is a function of its seed; the manifest hashes what a run used."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from lacuna.rig.manifest import config_hashes, sha256_file
from lacuna.rig.stress import CONDITIONS, Event, plan, services_of


def test_no_stress_is_no_events():
    assert plan("none", 0) == []


def test_the_schedule_is_a_function_of_condition_and_seed():
    for condition in CONDITIONS:
        assert plan(condition, 7) == plan(condition, 7)
    assert plan("gw-outage", 7) != plan("gw-outage", 8)
    assert len({tuple(plan("mixed", s)) for s in range(20)}) == 20


def test_an_outage_is_a_kill_and_a_start_the_stated_time_apart():
    for seed in range(30):
        kill, start = plan("gw-outage", seed)
        spec = CONDITIONS["gw-outage"][0]
        assert (kill.action, start.action) == ("kill", "start")
        assert kill.service == start.service == "otelcol-gw"
        assert spec["at"][0] <= kill.at_s <= spec["at"][1]
        assert start.at_s == pytest.approx(kill.at_s + spec["down_s"])


def test_a_kill_has_no_start_and_events_come_in_time_order():
    assert [(e.action, e.service) for e in plan("agent-kill", 3)] == [("kill", "agent-b")]
    for seed in range(30):
        events = plan("mixed", seed)
        assert events == sorted(events, key=lambda e: e.at_s)
        assert {e.service for e in events} == {"otelcol-gw", "otelcol-node", "agent-b"}
        assert sum(e.action == "kill" for e in events) == 3
        assert sum(e.action == "start" for e in events) == 2
        assert all(isinstance(e, Event) and e.at_s >= 0 for e in events)


def test_the_checkpoint_schedules_are_the_ones_its_runs_used():
    # Pinned: `artifacts/rig/checkpoint.json` holds runs made under these.
    assert plan("gw-outage", 0) == [
        Event(8.273, "kill", "otelcol-gw"),
        Event(28.273, "start", "otelcol-gw"),
    ]
    assert [e.at_s for e in plan("mixed", 1)] == [4.963, 6.372, 15.367, 16.372, 24.963]


def test_a_node_restart_takes_both_relays_down_and_brings_them_back_together():
    lengths = set()
    for seed in range(40):
        events = plan("node-down", seed)
        assert [(e.action, e.service) for e in events] == [
            ("kill", "otelcol-gw"),
            ("kill", "otelcol-node"),
            ("start", "otelcol-gw"),
            ("start", "otelcol-node"),
        ]
        assert events[0].at_s == events[1].at_s and events[2].at_s == events[3].at_s
        assert 8.0 <= events[0].at_s <= 30.0
        lengths.add(round(events[2].at_s - events[0].at_s, 3))
    assert len(lengths) == 40 and all(28.0 <= d <= 45.0 for d in lengths)


def test_a_blip_is_the_same_restart_but_shorter():
    for seed in range(40):
        events = plan("node-blip", seed)
        assert [e.action for e in events] == ["kill", "kill", "start", "start"]
        assert {e.service for e in events} == {"otelcol-gw", "otelcol-node"}
        assert events[0].at_s == events[1].at_s and 8.0 <= events[0].at_s <= 40.0
        assert 15.0 <= events[2].at_s - events[0].at_s <= 24.0


def test_a_restart_of_any_length_at_any_time():
    lengths = []
    for seed in range(60):
        events = plan("node-restart", seed)
        assert [e.action for e in events] == ["kill", "kill", "start", "start"]
        assert events[0].at_s == events[1].at_s and 8.0 <= events[0].at_s <= 90.0
        lengths.append(events[2].at_s - events[0].at_s)
    assert all(15.0 <= d <= 45.0 for d in lengths)
    assert min(lengths) < 24.0 and max(lengths) > 28.0  # both sides of the second batch


def test_separate_relay_outages_draw_their_own_times_and_lengths():
    overlap = 0
    for seed in range(60):
        by = {(e.service, e.action): e.at_s for e in plan("relays", seed)}
        gw = (by[("otelcol-gw", "kill")], by[("otelcol-gw", "start")])
        node = (by[("otelcol-node", "kill")], by[("otelcol-node", "start")])
        assert 25.0 <= gw[1] - gw[0] <= 45.0 and 10.0 <= node[1] - node[0] <= 40.0
        assert gw[0] != node[0]
        overlap += min(gw[1], node[1]) > max(gw[0], node[0])
    assert 0 < overlap < 60  # some runs overlap and some do not


def test_one_originator_is_killed_for_good_and_which_one_follows_the_seed():
    killed = []
    for seed in range(40):
        events = plan("node-down-kill", seed)
        agents = [e for e in events if e.service.startswith("agent-")]
        assert len(agents) == 1 and agents[0].action == "kill"
        assert 15.0 <= agents[0].at_s <= 50.0
        assert len(events) == 5
        killed.append(agents[0].service)
    assert set(killed) == {"agent-a", "agent-b"}


def test_an_unknown_condition_is_refused():
    with pytest.raises(KeyError):
        plan("nope", 0)


def test_config_hashes_are_by_relative_path_and_follow_the_content(tmp_path):
    (tmp_path / "config").mkdir()
    a, b = tmp_path / "config" / "a.yaml", tmp_path / "compose.yaml"
    a.write_text("x: 1\n")
    b.write_text("services: {}\n")
    got = config_hashes(tmp_path, [b, a])
    assert list(got) == ["compose.yaml", "config/a.yaml"]
    assert got["config/a.yaml"] == hashlib.sha256(b"x: 1\n").hexdigest() == sha256_file(a)
    a.write_text("x: 2\n")
    assert config_hashes(tmp_path, [a, b])["config/a.yaml"] != got["config/a.yaml"]
    assert config_hashes(tmp_path, [a, b])["compose.yaml"] == got["compose.yaml"]
    with pytest.raises(FileNotFoundError):
        config_hashes(tmp_path, [tmp_path / "missing.yaml"])


RIG = Path(__file__).resolve().parents[1] / "rig"


def test_only_telemetry_components_and_originators_are_ever_killed():
    allowed = {"otelcol-gw", "otelcol-node", "agent-a", "agent-b"}
    touched = {s for items in CONDITIONS.values() for item in items for s in services_of(item)}
    assert touched == allowed


def test_every_released_image_in_the_rig_is_pinned_by_digest():
    text = (RIG / "compose.yaml").read_text()
    images = re.findall(r"(?m)^\s+image:\s*(\S+)", text)
    assert len(images) == 3 and "lacuna-rig:dev" in images
    for image in images:
        assert image == "lacuna-rig:dev" or re.search(r"@sha256:[0-9a-f]{64}$", image), image
    assert re.search(r"(?m)^FROM \S+@sha256:[0-9a-f]{64}$", (RIG / "Dockerfile").read_text())
    assert "--require-hashes" in (RIG / "Dockerfile").read_text()


def _networks(text: str, service: str) -> set[str]:
    block = text.split(f"\n  {service}:\n", 1)[1].split("\n\n", 1)[0]
    return set(re.search(r"networks: \[([^\]]*)\]", block).group(1).replace(" ", "").split(","))


def test_the_kernel_shares_no_network_with_an_originator():
    text = (RIG / "compose.yaml").read_text()
    kernel = _networks(text, "kernel")
    for agent in ("agent-a", "agent-b"):
        assert not kernel & _networks(text, agent)
    # The witness-file writer is on the plane network and is in no stress condition.
    assert _networks(text, "backend") == {"plane"}
    assert "backend" not in {
        s for items in CONDITIONS.values() for i in items for s in services_of(i)
    }


def test_every_python_service_of_the_rig_is_told_the_env_and_originators_the_start():
    text = (RIG / "compose.yaml").read_text()
    assert "--env ${LACUNA_ENV:-backoffice}" in text.split("\n  kernel:\n", 1)[1].split("\n\n")[0]
    for front in ("mcp-a", "mcp-b"):
        block = text.split(f"\n  {front}:\n", 1)[1].split("\n\n", 1)[0]
        assert 'LACUNA_ENV: "${LACUNA_ENV:-backoffice}"' in block
    for agent, warm in (("agent-a", "WARMUP_A"), ("agent-b", "WARMUP_B")):
        block = text.split(f"\n  {agent}:\n", 1)[1].split("\n\n", 1)[0]
        assert "--t0 ${T0:?}" in block and f"--warmup ${{{warm}:-0}}" in block
