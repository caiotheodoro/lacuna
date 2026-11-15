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

ONE_RECORDER = """
version = 1
planes = ["p"]
actions = ["pay"]
[[witness]]
name = "log"
format = "jsonl"
plane = "p"
op = "op"
resource = "res"
value = "val"
id = ["n"]
"""


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


def test_lint_passes_the_rigs_map_and_fails_a_map_with_one_recorder(capsys, tmp_path):
    assert run(capsys, "lint", "--map", MAP) == (0, "ok\n", "")
    bad = tmp_path / "m.toml"
    bad.write_text(ONE_RECORDER)
    code, out, _ = run(capsys, "lint", "--map", str(bad))
    assert code == 1
    assert "error: plane p: 1 witness can see it (['log']); needs two" in out
    assert "1 planes cannot be audited with this map" in out
    code, out, _ = run(capsys, "lint", "--map", str(bad), "--json")
    assert code == 1 and len(json.loads(out)["errors"]) == 1 and json.loads(out)["warnings"]


def test_table_prints_what_each_combination_of_witnesses_recorded(capsys):
    code, out, err = run(capsys, "table", "--map", MAP, *INPUTS, "--json")
    assert code == 0 and err == ""
    got = json.loads(out)
    summary = rw.summarise(W.parent, BACKOFFICE)
    for plane, p in got["planes"].items():
        want = summary["planes"][plane]
        assert p["witnesses"] == ["spans", "access", "env"]
        assert p["union_count"] == want["union_count"] == sum(p["recorded_by"].values())
        assert p["trust_spans"] == want["trust_trace"]
        assert p["recorded_by"]["spans+access+env"] == want["table"][7]
        assert "estimate" not in p
    assert "total" not in got and "_built" not in got
    assert [b["witness"] for b in got["blocks"]] == ["spans", "access", "access", "env"]
    assert all(b["read"] == sum(v for k, v in b.items() if isinstance(v, int)
                                and k not in ("read", "block")) for b in got["blocks"])  # fmt: skip


def test_logs_that_agree_on_everything_give_a_floor_and_say_why_there_is_no_ceiling(capsys):
    code, out, _ = run(capsys, "estimate", "--map", MAP, *INPUTS)
    assert code == 0
    assert "all planes: 16 actions in some log; estimate: at least 16, no upper bound" in out
    assert "nothing says what all of them missed" in out


def test_an_estimate_is_at_least_the_union_and_carries_the_floors(capsys, tmp_path):
    code, out, _ = run(capsys, "estimate", "--map", MAP, *lossy(tmp_path), "--json")
    assert code == 0
    got = json.loads(out)
    union = sum(p["union_count"] for p in got["planes"].values())
    total = got["total"]
    assert total["union_count"] == union and total["at_least"] >= union
    assert total["bounded"] is (total["at_most"] is not None)
    for p in got["planes"].values():
        assert p["estimate"]["at_least"] >= p["union_count"] >= p["trust_spans"]
        # More than one kind of record now: the logs disagree somewhere.
        assert sum(1 for n in p["recorded_by"].values() if n) >= 2
    # The same files give the same numbers.
    again = json.loads(run(capsys, "estimate", "--map", MAP, *lossy(tmp_path), "--json")[1])
    assert again == got
    text = run(capsys, "estimate", "--map", MAP, *lossy(tmp_path))[1]
    assert f"all planes: {union} actions in some log; estimate:" in text
    assert "plane a:" in text and "spans+access+env" in text


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
    bad.write_text(ONE_RECORDER.replace("version = 1", "version = 7"))
    code, _, err = run(capsys, "lint", "--map", str(bad))
    assert code == 2 and "version = 1" in err


def test_the_command_runs_as_a_module_and_needs_no_agent_or_rig_package():
    code = (
        "import sys, lacuna.cli\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('httpx', 'mcp', 'opentelemetry')]\n"
        "assert not bad, bad\n"
        f"sys.exit(lacuna.cli.main(['lint', '--map', {MAP!r}]))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""}, timeout=120)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert r.stdout == "ok\n"
