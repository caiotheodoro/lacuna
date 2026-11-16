"""The `lacuna` command: lint a map, print tables, estimate. Runs on recorded rig files."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lacuna import cli, mapfile
from lacuna.envs.registry import BACKOFFICE, ENVS
from lacuna.estimators import Estimate
from lacuna.estimators.stratified import strat_bic
from lacuna.rig import witnesses as rw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib  # noqa: E402

W = ROOT / "tests" / "fixtures" / "rig" / "run" / "witnesses"
MAP = str(ROOT / "rig" / "map.toml")
INPUTS = [f"--in={n}={W / (n + '.jsonl')}" for n in ("spans", "logs", "env")]

TWO = """
version = 1
planes = ["p"]
actions = ["pay"]
[[witness]]
name = "log"
format = "jsonl"
pipeline = "shipper"
plane = "p"
session = "sid"
op = "op"
resource = "res"
value = "val"
id = ["n"]
[[witness]]
name = "state"
format = "jsonl"
pipeline = "diff"
plane = "p"
session = "sid"
op = "op"
resource = "res"
value = "val"
"""
ONE = TWO.split('[[witness]]\nname = "state"')[0]
# Study 5's layout on one plane: two logs linked by an id and one state diff.
SECOND_LOG = ONE.split("[[witness]]")[1].replace('"log"', '"relay"').replace("shipper", "tap")
THREE = TWO + "[[witness]]" + SECOND_LOG
# The same on two planes, each record saying which it went through.
PLANES = THREE.replace('planes = ["p"]', 'planes = ["p", "q"]').replace(
    'plane = "p"', 'plane = { path = ["pl"], map = { "p" = "p", "q" = "q" } }'
)


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def lossy(tmp_path: Path) -> list[str]:
    """The fixture with a different part of each log gone, as three files."""
    cut = {"spans": lambda x: x[:-1], "logs": lambda x: x[2:], "env": lambda x: x[::2]}
    out = []
    for name, keep in cut.items():
        lines = (W / f"{name}.jsonl").read_text().splitlines()
        (tmp_path / f"{name}.jsonl").write_text("\n".join(keep(lines)) + "\n")
        out.append(f"--in={name}={tmp_path / (name + '.jsonl')}")
    return out


def lossy_run(tmp_path: Path) -> tuple[Path, list[str]]:
    """The same cut fixture as a run folder `lacuna.rig.witnesses.load` reads, and as arguments."""
    run_dir = tmp_path / "run"
    shutil.copytree(W.parent, run_dir)
    argv = lossy(run_dir / "witnesses")
    return run_dir, argv


def files(tmp_path: Path, text: str = TWO, **logs: list[dict]) -> list[str]:
    """A map and its log files on disk, as arguments."""
    (tmp_path / "m.toml").write_text(text)
    argv = ["--map", str(tmp_path / "m.toml")]
    for name, rows in logs.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        argv.append(f"--in={name}={tmp_path / (name + '.jsonl')}")
    return argv


def pay(n: int, **kw) -> dict:
    return {"sid": "s", "n": n, "op": "pay", "res": f"o{n}", "val": "1"} | kw


# lint ------------------------------------------------------------------------------


def test_lint_says_what_it_checked_and_no_more(capsys, tmp_path):
    code, out, err = run(capsys, "lint", "--map", MAP)
    assert (code, err) == (0, "")
    assert out == "ok: two witnesses on every plane, by two pipelines as the map names them\n"
    bad = tmp_path / "m.toml"
    bad.write_text(ONE)
    code, out, _ = run(capsys, "lint", "--map", str(bad))
    assert code == 1
    assert "error: plane p: 1 of the map's witnesses can see it (['log']); needs two" in out
    assert out.rstrip().endswith("errors: 1. No count can be made with this map as it is.")
    code, out, _ = run(capsys, "lint", "--map", str(bad), "--json")
    assert code == 1 and len(json.loads(out)["errors"]) == 1 and json.loads(out)["warnings"]


def test_lint_with_warnings_only_does_not_say_ok(capsys, tmp_path):
    (tmp_path / "m.toml").write_text(TWO.replace('pipeline = "diff"\n', ""))
    code, out, _ = run(capsys, "lint", "--map", str(tmp_path / "m.toml"))
    assert code == 0 and "warning: witness[1] (state): no `pipeline` named" in out
    last = out.rstrip().split("\n")[-1]
    assert last == "no error, warnings: 3" and not last.startswith("ok")


# table -----------------------------------------------------------------------------


def test_table_prints_what_each_combination_of_witnesses_recorded(capsys):
    code, out, err = run(capsys, "table", "--map", MAP, *INPUTS, "--json")
    assert code == 0 and err == ""
    got = json.loads(out)
    summary = rw.summarise(W.parent, BACKOFFICE)
    for plane, p in got["planes"].items():
        want = summary["planes"][plane]
        assert p["witnesses"] == ["spans", "access", "env"]
        assert p["union_count"] == want["union_count"] == sum(p["recorded_by"].values())
        # The floors of AGENTS.md rule 5, under their names.
        assert p["union_bound"] == {"at_least": want["union_count"], "at_most": None}
        assert p["trust"] == {"witness": "spans", "count": want["trust_trace"]}
        assert p["recorded_by"]["spans+access+env"] == want["table"][7]
        assert len(p["recorded_by"]) == 7  # no cell for "recorded by nobody"
        assert "estimate" not in p and p["recorded_nothing"] == []
    assert "total" not in got and "assumes" not in got and "estimator" not in got
    assert not [k for k in got if k.startswith("_")]
    assert [b["witness"] for b in got["blocks"]] == ["spans", "access", "access", "env"]


def test_the_text_table_names_the_floors_and_leaves_empty_cells_out(capsys):
    code, out, _ = run(capsys, "table", "--map", MAP, *INPUTS)
    assert code == 0
    assert "plane a: union_count 10 (union_bound: at least 10), trust spans 10" in out
    assert "      10  spans+access+env" in out
    assert "  0  " not in out and "estimate" not in out


def test_a_witness_that_recorded_nothing_is_named_with_its_plane(capsys, tmp_path):
    argv = files(tmp_path, log=[pay(1), pay(2)], state=[])
    code, out, _ = run(capsys, "table", *argv)
    assert code == 0 and "  state recorded nothing on this plane" in out
    assert "state (block 1, state): no record read" in out
    assert "warning: this block read no record" in out
    got = json.loads(run(capsys, "table", *argv, "--json")[1])
    assert got["planes"]["p"]["recorded_nothing"] == ["state"]
    assert got["planes"]["p"]["trust"] == {"witness": "log", "count": 2}


def test_a_line_that_is_not_a_record_is_reported_and_other_separators_do_not_split_one(
    capsys, tmp_path
):
    argv = files(tmp_path, log=[pay(1, res="a b"), pay(2)], state=[pay(1, res="a b")])
    with (tmp_path / "log.jsonl").open("a") as f:
        f.write('{"sid": "s", "n": 3, "op": "pa')  # cut by a kill
    code, out, _ = run(capsys, "table", *argv)
    assert code == 0
    assert "       1  log+state" in out and "       1  log\n" in out
    assert "warning: 1 lines are not JSON objects (a cut line?)" in out


# estimate ----------------------------------------------------------------------------


def test_logs_that_agree_on_everything_give_the_union_and_say_why_there_is_no_ceiling(
    capsys, tmp_path
):
    rows = [pay(i) for i in range(10)]
    code, out, _ = run(capsys, "estimate", *files(tmp_path, log=rows, state=rows))
    assert code == 0
    assert "  estimate: at least 10 (what the logs hold), no upper bound" in out
    assert "  no upper bound: its witnesses recorded the same actions, or only one" in out
    assert "all planes: union_count 10; estimate: at least 10 (what the logs hold)" in out
    assert out.rstrip().endswith(cli.ASSUMES)


def test_on_the_rigs_layout_logs_that_agree_on_everything_give_the_count(capsys):
    code, out, _ = run(capsys, "estimate", "--map", MAP, *INPUTS)
    assert code == 0
    assert "  estimate: 10 actions (interval 10 to 10)" in out
    assert "all planes: union_count 16; estimate: 16 actions (interval 16 to 16)" in out
    assert out.rstrip().endswith(cli.TESTED + "\n" + cli.ASSUMES)


@pytest.mark.parametrize("cut", [False, True])
def test_on_the_rigs_layout_the_estimate_is_the_one_study_5_registered(capsys, tmp_path, cut):
    run_dir, argv = lossy_run(tmp_path) if cut else (W.parent, INPUTS)
    # The study's own path from a run's files to `ll_sa_hom`: no map file in it.
    env = ENVS["backoffice_churn"]
    loaded = rw.load(run_dir, env)
    per_plane = study5_lib.plane_tables(loaded["obs"])
    per_session = study5_lib.session_tables(loaded["obs"])
    want = study5_lib.estimates(per_plane, per_session, study5_lib.streams(loaded, env))[
        "ll_sa_hom"
    ]
    if "nonconverged" in want.detail:
        # The registered code's defect, which rounding decides (`tests/test_sa_hom.py`).
        # It does not happen on these tables where this was written; where it does, the
        # tool is held to the library's estimate on the study's tables.
        pooled = sum(per_plane.values())
        want = cli.sa_hom(pooled, per_session)
    assert math.isfinite(want.hi) and (want.hi > want.n_observed) is cut

    # What the tool hands the estimator is what the study handed it, cell for cell.
    m = mapfile.load(Path(MAP))
    made = cli.tables(m, cli._sources([a.removeprefix("--in=") for a in argv]))
    mine, rows = cli.study5_tables(m, made["_built"], made["_obs"])
    assert {p: list(t) for p, t in mine.items()} == {p: list(t) for p, t in per_plane.items()}
    assert [list(t) for _, t in rows] == [list(t) for t in per_session]
    assert len(rows) == 4 and [p for p, _ in rows] == ["a", "b", "a", "b"]

    code, out, _ = run(capsys, "estimate", "--map", MAP, *argv, "--json")
    assert code == 0
    got = json.loads(out)
    assert got["estimator"] == {"name": "ll_sa_hom", "study5_tested": True, "says": cli.TESTED}
    assert "the one study 5 tested" in cli.TESTED and "not the one" not in cli.TESTED
    # It says what the map shows and what the tool cannot know about it.
    assert "one witness without an id per plane" in cli.TESTED
    assert "reads that witness as a state diff and cannot check that it is one" in cli.TESTED
    total = got["total"]
    assert (total["actions"], total["interval_low"], total["interval_high"]) == (
        round(want.point, 1),
        round(want.lo, 1),
        round(want.hi, 1),
    )
    assert total["bounded"] and total["union_count"] == want.n_observed == 16
    for plane, p in got["planes"].items():
        e = p["estimate"]
        one = [t for q, t in rows if q == plane]
        alone = cli.sa_hom(mine[plane], one)  # the same estimator on one plane's tables
        assert (e["actions"], e["interval_high"]) == (round(alone.point, 1), round(alone.hi, 1))


def test_the_tested_layout_is_read_from_the_map_in_any_order_of_blocks(capsys, tmp_path):
    log = [pay(i) for i in range(40)]
    relay = [pay(i) for i in range(5, 45)]
    state = [pay(i) for i in range(10, 50, 2)]
    argv = files(tmp_path, THREE, log=log, relay=relay, state=state)
    got = json.loads(run(capsys, "estimate", *argv, "--json")[1])
    assert got["planes"]["p"]["witnesses"] == ["log", "state", "relay"]
    assert got["estimator"]["name"] == "ll_sa_hom" and got["estimator"]["study5_tested"]
    # 45 in a log, 20 in the state, 18 in both: Lincoln-Petersen between the two lists.
    assert got["total"]["actions"] == pytest.approx(47 + 2 * 27 / 18, abs=0.05)
    assert got["total"] == got["planes"]["p"]["estimate"] | {
        k: got["total"][k] for k in ("union_count", "union_bound")
    }
    # The state diff last, as on the rig: the same table under other bits, the same estimate.
    blocks = THREE.split("[[witness]]")
    reordered = "[[witness]]".join([blocks[0], blocks[1], blocks[3], blocks[2]])
    argv = files(tmp_path, reordered, log=log, relay=relay, state=state)
    again = json.loads(run(capsys, "estimate", *argv, "--json")[1])
    assert again["planes"]["p"]["witnesses"] == ["log", "relay", "state"]
    assert again["total"] == got["total"]


@pytest.mark.parametrize(
    "text, why",
    [
        (TWO, "one log and a state diff"),
        (
            THREE.replace('pipeline = "diff"', 'pipeline = "diff"\nid = ["n"]'),
            "three logs, no diff",
        ),
        (
            THREE.replace('plane = "p"\nsession = "sid"', 'plane = "p"', 1),
            "a block with no session",
        ),
    ],
)
def test_any_other_layout_keeps_strat_bic_and_says_it_is_not_the_tested_one(
    capsys, tmp_path, text, why
):
    log = [pay(i) for i in range(40)]
    relay = [pay(i) for i in range(5, 45)]
    state = [pay(i) for i in range(10, 50)]
    logs = {"log": log, "state": state} | ({} if text is TWO else {"relay": relay})
    argv = files(tmp_path, text, **logs)
    code, out, _ = run(capsys, "estimate", *argv)
    assert code == 0, why
    assert out.rstrip().endswith(cli.NOT_TESTED + "\n" + cli.ASSUMES), why
    assert "strat_bic" in cli.NOT_TESTED and "not the one study 5 tested" in cli.NOT_TESTED
    got = json.loads(run(capsys, "estimate", *argv, "--json")[1])
    assert got["estimator"] == {"name": "strat_bic", "study5_tested": False, "says": cli.NOT_TESTED}
    # The numbers are `strat_bic`'s on the same tables, as before this estimator was added.
    m = mapfile.load(tmp_path / "m.toml")
    built = cli.tables(m, cli._sources([a.removeprefix("--in=") for a in argv[2:]]))["_built"]
    (names, table) = built["p"]
    want = strat_bic({"p": table}, live={"p": tuple(range(len(names)))})
    assert math.isfinite(want.hi) or text is not TWO, why
    for e in (got["total"], got["planes"]["p"]["estimate"]):
        assert (e["actions"], e["interval_low"], e["interval_high"], e["bounded"]) == (
            cli._num(want.point),
            cli._num(want.lo),
            cli._num(want.hi),
            want.bounded and math.isfinite(want.point),
        ), why


def test_on_the_tested_layout_no_ceiling_has_the_two_lists_reason(capsys, tmp_path):
    log = [pay(i) for i in range(20)]
    state = [pay(i) for i in range(20, 30)]  # nothing is in both a log and the state diff
    code, out, _ = run(capsys, "estimate", *files(tmp_path, THREE, log=log, relay=log, state=state))
    assert code == 0
    assert "  estimate: at least 30 (what the logs hold), no upper bound" in out
    assert "  no upper bound: no action is in both a log and the state diff" in out
    state = [pay(i) for i in range(19, 60)]  # one is
    out = run(capsys, "estimate", *files(tmp_path, THREE, log=log, relay=log, state=state))[1]
    assert "no upper bound: too few actions are in both a log and the state diff" in out
    out = run(capsys, "estimate", *files(tmp_path, THREE, log=[], relay=[], state=[]))[1]
    assert "no upper bound: no witness recorded an action" in out


def test_an_estimate_made_where_the_registered_fit_fails_says_so(capsys, tmp_path, monkeypatch):
    from lacuna.estimators import loglinear

    def fails(self, big_n):
        raise loglinear.ConvergenceError(f"IRLS did not converge at N={big_n}")

    # Every cell of the table holds an action, so the registered fit is not on a boundary.
    log = [pay(i) for i in range(40)]
    relay = [pay(i) for i in range(5, 45)]
    argv = files(tmp_path, THREE, log=log, relay=relay, state=[pay(i) for i in range(0, 60, 2)])
    plain = json.loads(run(capsys, "estimate", *argv, "--json")[1])
    assert all(plain["planes"]["p"]["recorded_by"].values())
    assert "closed_form" not in plain["total"] and plain["total"]["bounded"]
    monkeypatch.setattr(loglinear.Profile, "__call__", fails)
    got = json.loads(run(capsys, "estimate", *argv, "--json")[1])
    assert got["total"] == plain["total"] | {"closed_form": True}
    out = run(capsys, "estimate", *argv)[1]
    assert out.count("did not converge on this table") == 2  # the plane's line and the total's
    assert "added after the study" in out


def test_a_point_outside_its_interval_is_reported_with_the_end_moved_to_it(capsys, tmp_path):
    # The point is real-valued (41 + 4/41 more) and the interval search is over whole
    # numbers, so the estimator's own interval can end under its point.
    e = cli.sa_hom(np.array([0, 4, 0, 0, 1, 0, 29, 12]), [])
    assert (round(e.point, 4), e.lo, e.hi) == (46.0976, 46.0, 46.0)
    assert cli._estimate(e) == {
        "actions": 46.1,
        "interval_low": 46.0,
        "interval_high": 46.1,
        "bounded": True,
        "interval_extended_to_point": True,
    }
    below = cli._estimate(Estimate("strat_bic", 10, 9.5, 10.0, 12.0))
    assert (below["interval_low"], below["interval_high"]) == (9.5, 12.0)
    assert below["interval_extended_to_point"] is True
    for inside in (
        Estimate("x", 10, 11.0, 10.0, 12.0),
        Estimate("x", 10, math.inf, 10.0, math.inf),
    ):
        assert "interval_extended_to_point" not in cli._estimate(inside)
    assert cli._estimate(Estimate("x", 10, 11.0, 10.0, math.inf))["interval_high"] is None

    # The same table through the command: 4 in `log` alone, 29 in `relay` and the state,
    # 12 in all three, 1 in the state alone.
    log = [pay(i) for i in [*range(4), *range(33, 45)]]
    relay = [pay(i) for i in range(4, 45)]
    state = [pay(i) for i in range(4, 46)]
    argv = files(tmp_path, THREE, log=log, relay=relay, state=state)
    total = json.loads(run(capsys, "estimate", *argv, "--json")[1])["total"]
    assert (total["actions"], total["interval_low"], total["interval_high"]) == (46.1, 46.0, 46.1)
    assert total["interval_extended_to_point"] is True and total["bounded"]
    out = run(capsys, "estimate", *argv)[1]
    assert "all planes: union_count 46; estimate: 46.1 actions (interval 46 to 46.1)" in out


def test_link_ids_that_repeat_across_sessions_are_refused_on_the_tested_layout(capsys, tmp_path):
    # Two sessions that each number their calls from 0: a plane's table links by the id
    # alone and joins the two, a session's table does not, and the two stop adding up.
    def rows(keep) -> list[dict]:
        return [pay(n, sid=sid, res=f"{sid}-{n}") for sid in ("s1", "s2") for n in keep]

    logs = {"log": rows(range(0, 14)), "relay": rows(range(6, 20)), "state": rows(range(0, 20, 2))}
    code, out, err = run(capsys, "estimate", *files(tmp_path, THREE, **logs))
    assert (code, out) == (2, "") and err.count("\n") == 1
    assert "plane p: link ids repeat across sessions" in err
    assert "must be unique across the plane" in err and "`id`" in err
    # `table` builds no session table and is not stopped; nor is another layout.
    assert run(capsys, "table", *files(tmp_path, THREE, **logs))[0] == 0
    assert run(capsys, "estimate", *files(tmp_path, log=logs["log"], state=logs["state"]))[0] == 0
    # With the session in the id the same logs are 40 actions, and the count is made.
    unique = THREE.replace('id = ["n"]', 'id = ["sid", "n"]')
    got = json.loads(run(capsys, "estimate", *files(tmp_path, unique, **logs), "--json")[1])
    assert got["total"]["union_count"] == 40 and got["total"]["bounded"]
    assert got["estimator"]["name"] == "ll_sa_hom"


def test_a_plane_whose_state_diff_recorded_nothing_leaves_the_total_with_no_ceiling(
    capsys, tmp_path
):
    def on(plane: str, keep) -> list[dict]:
        return [pay(n, pl=plane, sid=f"s-{plane}", res=f"{plane}-{n}") for n in keep]

    p = {
        "log": on("p", range(0, 30)),
        "relay": on("p", range(5, 35)),
        "state": on("p", range(10, 40)),
    }
    q = {"log": on("q", range(0, 30)), "relay": on("q", range(5, 35)), "state": []}
    both = {name: p[name] + q[name] for name in p}
    argv = files(tmp_path, PLANES, **both)
    got = json.loads(run(capsys, "estimate", *argv, "--json")[1])
    assert got["estimator"]["name"] == "ll_sa_hom"
    assert got["planes"]["p"]["estimate"]["bounded"]
    assert got["planes"]["q"]["estimate"]["why_unbounded"] == (
        "no action is in both a log and the state diff"
    )
    # Plane q's 35 actions have no state diff behind them, so the sum has no ceiling either.
    assert got["total"] == {
        "actions": None,
        "interval_low": 75.0,
        "interval_high": None,
        "bounded": False,
        "why_unbounded": "on plane q no action is in both a log and the state diff",
        "union_count": 75,
        "union_bound": {"at_least": 75, "at_most": None},
    }
    out = run(capsys, "estimate", *argv)[1]
    assert "all planes: union_count 75; estimate: at least 75 (what the logs hold), no upper" in out
    assert "\n  no upper bound: on plane q no action is in both a log and the state diff\n" in out

    # A plane nothing was recorded on is not such a plane: the total is plane p's estimate.
    alone = json.loads(run(capsys, "estimate", *files(tmp_path, PLANES, **p), "--json")[1])
    assert alone["planes"]["q"]["union_count"] == 0
    assert alone["total"]["bounded"] and "why_unbounded" not in alone["total"]
    assert alone["total"] == alone["planes"]["p"]["estimate"] | {
        k: alone["total"][k] for k in ("union_count", "union_bound")
    }


def test_an_estimate_carries_the_floors_and_what_it_assumes(capsys, tmp_path):
    code, out, _ = run(capsys, "estimate", "--map", MAP, *lossy(tmp_path), "--json")
    assert code == 0
    got = json.loads(out)
    union = sum(p["union_count"] for p in got["planes"].values())
    total = got["total"]
    assert total["union_count"] == union and total["union_bound"]["at_least"] == union
    assert total["bounded"] is (total["interval_high"] is not None)
    assert got["assumes"] == cli.ASSUMES and "independently" in cli.ASSUMES
    assert "not calibrated" in cli.ASSUMES and "too high" in cli.ASSUMES
    for p in got["planes"].values():
        assert p["union_count"] >= p["trust"]["count"]
        assert sum(1 for n in p["recorded_by"].values() if n) >= 2  # the logs disagree
        if p["estimate"]["bounded"]:
            assert p["estimate"]["interval_low"] >= p["union_count"]
        else:
            assert p["estimate"]["actions"] is None and "why_unbounded" in p["estimate"]
    again = json.loads(run(capsys, "estimate", "--map", MAP, *lossy(tmp_path), "--json")[1])
    assert again == got  # the same files give the same numbers


def test_a_bounded_estimate_is_a_point_and_an_interval_with_no_percentage_claimed(capsys, tmp_path):
    log = [pay(i) for i in range(40)]
    state = [pay(i) for i in range(10, 50)]  # 30 in both, 10 in each alone
    argv = files(tmp_path, log=log, state=state)
    code, out, _ = run(capsys, "estimate", *argv)
    assert code == 0
    assert "plane p: union_count 50 (union_bound: at least 50), trust log 40" in out
    (line,) = [x for x in out.splitlines() if x.startswith("all planes")]
    assert "estimate: " in line and " actions (interval " in line and "%" not in out
    got = json.loads(run(capsys, "estimate", *argv, "--json")[1])["total"]
    assert got["bounded"] and 50 <= got["interval_low"] <= got["actions"] <= got["interval_high"]
    assert got["actions"] == pytest.approx(40 * 40 / 30, abs=2.5)  # near Lincoln-Petersen


def test_nearly_disjoint_logs_are_unbounded_for_their_own_reason(capsys, tmp_path):
    log = [pay(i) for i in range(31)]
    state = [pay(i) for i in range(30, 61)]  # one action in both
    code, out, _ = run(capsys, "estimate", *files(tmp_path, log=log, state=state))
    assert code == 0 and "no upper bound: too few actions were recorded by more than one" in out
    assert "recorded the same actions" not in out


def test_empty_logs_are_said_to_be_empty(capsys, tmp_path):
    code, out, _ = run(capsys, "estimate", *files(tmp_path, log=[], state=[]))
    assert code == 0
    assert "no upper bound: no witness recorded an action" in out
    assert "all planes: union_count 0; estimate: at least 0" in out


# Wrong input ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, message",
    [
        (["estimate", "--map", MAP], "no input for sources ['env', 'logs', 'spans']"),
        (["table", "--map", MAP, "--in", "spans"], "--in takes NAME=PATH"),
        (["table", "--map", MAP, "--in", "spans=/no/such/file"], "cannot read /no/such/file"),
        (["table", "--map", MAP, f"--in=a={MAP}", f"--in=a={MAP}"], "names 'a' twice"),
        (["lint", "--map", "/no/such/map.toml"], "No such file"),
    ],
)
def test_a_wrong_invocation_exits_2_with_one_line_and_no_traceback(capsys, argv, message):
    code, out, err = run(capsys, *argv)
    assert code == 2 and out == ""
    assert err.startswith("lacuna: ") and message in err and err.count("\n") == 1


def test_a_map_that_cannot_be_followed_exits_2(capsys, tmp_path):
    bad = tmp_path / "m.toml"
    bad.write_text(TWO.replace("version = 1", "version = 7"))
    code, _, err = run(capsys, "lint", "--map", str(bad))
    assert code == 2 and "version = 1" in err
    bad.write_bytes(b"\xff\xfe\x00bad")
    code, _, err = run(capsys, "lint", "--map", str(bad))
    assert code == 2 and "is not UTF-8 text" in err and err.count("\n") == 1


def test_a_map_with_one_witness_is_refused_by_table_and_estimate_too(capsys, tmp_path):
    for command in ("table", "estimate"):
        code, out, err = run(capsys, command, *files(tmp_path, ONE, log=[pay(1)]))
        assert (code, out) == (2, "")
        assert "plane p: 1 of the map's witnesses can see it; needs two" in err


@pytest.mark.parametrize(
    "rows",
    [
        [{"sid": ["x"], "n": 1, "op": "pay", "res": "o", "val": "1"}],
        [{"sid": "s", "n": [1], "op": "pay", "res": "o", "val": "1"}],
        [{"sid": "s", "n": 1, "op": {"a": 1}, "res": "o", "val": "1"}],
        [{"sid": "s", "n": 1, "op": "pay", "res": ["o"], "val": {"v": 1}}],
    ],
)
def test_a_log_with_values_of_the_wrong_kind_is_counted_not_a_traceback(capsys, tmp_path, rows):
    code, out, err = run(capsys, "table", *files(tmp_path, log=[*rows, pay(2)], state=[pay(2)]))
    assert (code, err) == (0, "")
    assert "       1  log+state" in out


def test_a_file_this_cannot_digest_exits_2_with_one_line(capsys, tmp_path):
    argv = files(tmp_path, log=[pay(1)], state=[pay(1)])
    (tmp_path / "log.jsonl").write_bytes(b"\xff\xfe\x00\x01")
    code, out, err = run(capsys, "table", *argv)
    assert (code, out) == (2, "") and "is not UTF-8 text" in err and err.count("\n") == 1
    (tmp_path / "log.jsonl").write_text("[" * 100000 + "\n")  # nesting no parser survives
    code, out, err = run(capsys, "table", *argv)
    assert code in (0, 2) and "Traceback" not in err  # counted as unreadable, or refused


def test_help_for_estimate_says_what_it_assumes(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["estimate", "--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0
    assert "independently" in out and "not calibrated" in out
    assert "names the estimator and says whether it is the one study 5 tested" in " ".join(
        out.split()
    )


def test_the_command_runs_as_a_module_and_needs_no_agent_or_rig_package():
    code = (
        "import sys, lacuna.cli\n"
        f"code = lacuna.cli.main(['estimate', '--map', {MAP!r}, *{INPUTS!r}])\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('httpx', 'mcp', 'opentelemetry')]\n"
        "assert not bad, bad\n"
        "sys.exit(code)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""}, timeout=120)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "all planes: union_count 16" in r.stdout
