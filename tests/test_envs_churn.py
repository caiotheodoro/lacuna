"""The churn task mix: back-office tickets whose later tickets undo or overwrite earlier ones."""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.agents.scripted import ScriptedAgent
from lacuna.envs import churn
from lacuna.envs.kernel import Kernel
from lacuna.envs.registry import BACKOFFICE, BACKOFFICE_CHURN, ENVS
from lacuna.envs.tools import ACTION_TOOLS, TOOLS, Router

SEEDS = range(40)
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def _run(seed: int, n: int = 10) -> tuple[Kernel, list[str]]:
    """Every ticket's plan, in queue order, through the router. Returns the kernel and replies."""
    ep = churn.make_episode(seed, n)
    kernel = Kernel(ep.world)
    router = Router(kernel, np.random.default_rng(0), 0.0, 0.0)
    replies = [
        router.execute(f"c{i}", tool, args)
        for i, (tool, args) in enumerate(c for t in ep.tickets for c in t.plan)
    ]
    return kernel, replies


def test_it_is_registered_with_the_back_office_tools():
    assert ENVS["backoffice_churn"] is BACKOFFICE_CHURN
    assert BACKOFFICE_CHURN.tools is BACKOFFICE.tools
    assert BACKOFFICE_CHURN.action_tools == BACKOFFICE.action_tools
    assert BACKOFFICE_CHURN.make_episode is churn.make_episode


@pytest.mark.parametrize("n", [6, 10, 16])
def test_an_episode_has_n_tickets_and_the_closing_summary(n):
    for seed in SEEDS:
        ep = churn.make_episode(seed, n)
        assert len(ep.tickets) == n + 1
        assert ep.tickets[-1].plan[0][0] == "write_file"
        assert "/reports/summary.txt" in ep.tickets[-1].text
        for t in ep.tickets:
            assert t.text and t.plan
            assert {tool for tool, _ in t.plan} <= TOOL_NAMES


def test_the_same_seed_gives_the_same_episode_and_seeds_differ():
    a, b = churn.make_episode(7, 10), churn.make_episode(7, 10)
    assert [t.text for t in a.tickets] == [t.text for t in b.tickets]
    assert [t.plan for t in a.tickets] == [t.plan for t in b.tickets]
    assert a.world == b.world
    assert [t.text for t in a.tickets] != [t.text for t in churn.make_episode(8, 10).tickets]


def test_every_plan_succeeds_in_queue_order():
    # A follow-up that came before its opener would fail here (deleting a draft that
    # was never written) or would show in the thread check below.
    for seed in SEEDS:
        kernel, replies = _run(seed)
        assert not [r for r in replies if "error" in r], seed
        assert len(kernel.ledger) == sum(
            1 for t in churn.make_episode(seed, 10).tickets for tool, _ in t.plan
            if tool in ACTION_TOOLS
        )  # fmt: skip


def test_threads_keep_their_order_and_cover_every_ticket_once():
    for seed in SEEDS:
        ep = churn.make_episode(seed, 10)
        threads = ep.meta["threads"]
        flat = sorted(i for th in threads for i in th["tickets"])
        assert flat == list(range(10))
        for th in threads:
            assert th["tickets"] == sorted(th["tickets"])
            assert th["kind"] in churn.KINDS


def test_no_two_threads_touch_the_same_field_order_or_file():
    for seed in SEEDS:
        ep = churn.make_episode(seed, 10)
        owner: dict[tuple[str, str], int] = {}
        for k, th in enumerate(ep.meta["threads"]):
            for i in th["tickets"]:
                for tool, args in ep.tickets[i].plan:
                    norm = BACKOFFICE.normalise(tool, args)
                    if tool == "send_email" or norm is None:
                        continue
                    res = ("order", norm[1]) if "order_id" in args else ("state", norm[1])
                    assert owner.setdefault(res, k) == k, (seed, res)


def test_a_material_share_of_actions_leaves_no_trace_in_the_end_state():
    ledger = rows = 0
    shares = []
    for seed in SEEDS:
        kernel, _ = _run(seed)
        seen = len(BACKOFFICE_CHURN.observe(kernel))
        ledger += len(kernel.ledger)
        rows += seen
        shares.append(1 - seen / len(kernel.ledger))
    pooled = 1 - rows / ledger
    assert 0.30 <= pooled <= 0.50, pooled
    # Not one kind of episode carrying the share: most episodes hide something.
    assert sum(s >= 0.15 for s in shares) >= 0.8 * len(shares)


def test_each_kind_hides_what_its_docstring_says():
    # (actions, rows in the end state or the sink) per thread kind, from the plans alone.
    want = {
        "refund": (2, 2),
        "cancel": (2, 2),
        "contact": (1, 1),
        "move": (2, 1),
        "promo": (2, 0),
        "phone": (2, 0),
        "draft": (3, 1),
    }
    seen: dict[str, set[tuple[int, int]]] = {}
    for seed in range(120):
        ep = churn.make_episode(seed, 10)
        for th in ep.meta["threads"]:
            if th["kind"] == "handoff":
                continue
            world = churn.make_episode(seed, 10).world
            kernel = Kernel(world)
            router = Router(kernel, np.random.default_rng(0), 0.0, 0.0)
            for i in th["tickets"]:
                for j, (tool, args) in enumerate(ep.tickets[i].plan):
                    assert "error" not in router.execute(f"c{i}-{j}", tool, args)
            got = (len(kernel.ledger), len(BACKOFFICE_CHURN.observe(kernel)))
            seen.setdefault(th["kind"], set()).add(got)
    assert seen == {k: {v} for k, v in want.items()}


def test_handoff_notes_are_one_file_written_again_and_again():
    counts = []
    for seed in range(120):
        ep = churn.make_episode(seed, 10)
        hand = [th for th in ep.meta["threads"] if th["kind"] == "handoff"]
        assert len(hand) <= 1
        for th in hand:
            plans = [ep.tickets[i].plan for i in th["tickets"]]
            assert all(len(p) == 1 and p[0][0] == "write_file" for p in plans)
            assert {p[0][1]["path"] for p in plans} == {churn.HANDOFF_PATH}
            contents = [p[0][1]["content"] for p in plans]
            assert all(
                b.startswith(a) and b != a for a, b in zip(contents, contents[1:], strict=False)
            )
            counts.append(len(plans))
    assert counts and set(counts) <= {2, 3}


def test_the_scripted_agent_can_follow_it():
    ep = churn.make_episode(3, 10)
    calls = ScriptedAgent(np.random.default_rng([3, 1])).plan(ep)
    kernel = Kernel(churn.make_episode(3, 10).world)
    router = Router(kernel, np.random.default_rng(0), 0.0, 0.0)
    for i, (tool, args) in enumerate(calls):
        assert "error" not in router.execute(f"c{i}", tool, args)
    assert len(kernel.ledger) >= sum(len(t.plan) for t in ep.tickets)
