"""`--help` on any script prints usage and exits; it never runs the job.

Before this, `python scripts/precision.py --help` ran the whole sweep and
overwrote artifacts/robustness/precision.json. This runs every script with
`--help` in a subprocess and checks that exit status is 0 and that nothing the
scripts write to changed.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
# Trees and files that scripts write into.
WATCHED = ("artifacts", "paper", "docs", "dataset", "MEASUREMENT_CARD.json")


def _snapshot() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in WATCHED:
        base = ROOT / name
        files = [base] if base.is_file() else sorted(base.rglob("*")) if base.exists() else []
        for p in files:
            if p.is_file():
                out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _help(name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        cwd=ROOT,
        timeout=120,
    )


# loss_sim.py is a library, not a job: it ships byte-for-byte in the dataset, is
# standard-library only and is imported (`from loss_sim import load, thin`). It
# has no entry point and writes nothing, so it needs no CLI.
LIBRARIES = {"loss_sim.py"}


def test_every_script_has_a_cli():
    missing = [
        p.name for p in SCRIPTS if p.name not in LIBRARIES and "argparse" not in p.read_text()
    ]
    assert not missing, f"no argparse in: {missing}"


def test_help_runs_nothing_and_writes_nothing():
    before = _snapshot()
    bad = []
    for p in SCRIPTS:
        if "argparse" not in p.read_text():
            continue
        r = _help(p.name)
        if r.returncode != 0:
            bad.append(f"{p.name}: exit {r.returncode}: {r.stderr[-200:]}")
        elif "usage:" not in r.stdout:
            bad.append(f"{p.name}: no usage line")
    assert not bad, "\n".join(bad)
    after = _snapshot()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"--help modified: {changed}"


@pytest.mark.parametrize("name", ["precision.py", "make_card.py", "burst_loss.py"])
def test_artifact_writers_take_out(name):
    assert "--out" in _help(name).stdout
