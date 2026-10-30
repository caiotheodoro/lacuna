"""`make validate` is the CI gate, so it must run on a bare runner.

The workflow installs once and then validates. Nothing `validate` runs may need
the network, a key, `tectonic` (the paper build) or `docker`; a test that wants
one must skip when it is absent, not fail. The first is checked on the commands
`make -n validate` expands to (sub-makes included), the second by reading the tests.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "validate.yml"

# What the gate may run: its three sub-targets and the tools they call, nothing else.
ALLOWED = re.compile(r"^(\S*/)?make (lint|test|privacy-gate)$|^uv run (ruff|pytest)\b|^files=")
# Anything that reaches off the machine, builds the paper, runs a script or needs a key.
FORBIDDEN = re.compile(
    r"\b(curl|wget|ssh|scp|rsync|docker|podman|tectonic|latexmk|huggingface-cli|npm|pip|brew|apt)\b"
    r"|https?://|\bgit (push|pull|fetch|clone|remote)\b|\buv (sync|pip|add|publish|build)\b"
    r"|--group|API_KEY|_TOKEN|\bscripts/|\bpython3?\b"
    r"|\bmake (paper|dataset|sim|sim-smoke|repro-headline|card|paper-check)\b"
)


def _expanded_validate() -> list[str]:
    """The commands `make validate` would run, sub-makes expanded (`make -n` runs those)."""
    make = shutil.which("make")
    if make is None:
        pytest.skip("no make on this machine")
    r = subprocess.run(
        [make, "-n", "validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")},
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    # A recipe line continued with a backslash is one command.
    joined = re.sub(r"\\\n\s*", " ", r.stdout)
    return [c.strip() for c in joined.splitlines() if c.strip()]


def test_workflow_installs_then_validates():
    text = WORKFLOW.read_text()
    assert "pull_request" in text and "push" in text
    assert "ubuntu-latest" in text
    assert "astral-sh/setup-uv" in text and '"3.11"' in text
    runs = re.findall(r"(?m)^\s*run:\s*(.+)$", text)
    assert runs == ["make install", "make validate"], runs


def test_validate_target_stays_offline():
    commands = _expanded_validate()
    joined = "\n".join(commands)
    # lint (check, format), test, privacy-gate: three sub-makes and four commands.
    assert "uv run ruff check" in joined and "uv run ruff format --check" in joined, joined
    assert "uv run pytest -q" in joined and "privacy-gate" in joined, joined
    for c in commands:
        assert ALLOWED.search(c), f"validate runs something not reviewed here: {c}"
        # The privacy gate names token prefixes in its grep pattern; those are patterns, not uses.
        probe = c.split("xargs grep", 1)[0] if c.startswith("files=") else c
        assert not FORBIDDEN.search(probe), f"validate would reach out, build or run a script: {c}"


@pytest.mark.parametrize(
    "bad",
    [
        "curl -s https://example.com/x",
        "uv run python scripts/paper_tables.py",
        "cd paper && tectonic -X compile lacuna.tex",
        "docker run --rm img",
        "make paper",
        "uv sync --group dev",
        "git push origin main",
    ],
)
def test_the_detector_refuses_what_validate_must_not_run(bad):
    assert FORBIDDEN.search(bad) or not ALLOWED.search(bad)


def test_no_test_shells_out_to_tectonic_or_docker():
    hits = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        text = p.read_text()
        if p.name == Path(__file__).name or "shutil.which(" in text:
            continue  # a test that probes for the tool is the skip we ask for
        for line in text.splitlines():
            if re.search(r"\b(tectonic|docker)\b", line):
                hits.append(f"{p.name}: {line.strip()}")
    assert not hits, "tests referencing tectonic/docker must skip when absent:\n" + "\n".join(hits)
