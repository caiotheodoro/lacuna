"""The `lacuna` command: lint a map, print tables, estimate. Runs on recorded rig files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lacuna import cli
from lacuna.envs.registry import BACKOFFICE
from lacuna.rig import witnesses as rw

ROOT = Path(__file__).resolve().parents[1]
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
    assert "total" not in got and "_built" not in got and "assumes" not in got
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


def test_logs_that_agree_on_everything_give_the_union_and_say_why_there_is_no_ceiling(capsys):
    code, out, _ = run(capsys, "estimate", "--map", MAP, *INPUTS)
    assert code == 0
    assert "  estimate: at least 10 (what the logs hold), no upper bound" in out
    assert "  no upper bound: its witnesses recorded the same actions, or only one" in out
    assert "all planes: union_count 16; estimate: at least 16 (what the logs hold)" in out
    assert out.rstrip().endswith(cli.ASSUMES)


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
