"""`make validate` is the CI gate, so it must run on a bare runner.

The workflow installs once and then validates. Nothing `validate` runs may need
the network, a key, `tectonic` (the paper build) or `docker`; a test that wants
one must skip when it is absent, not fail. Both are checked here by reading, and
`PATH` manipulation was used to confirm it by running (see the commit message).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "validate.yml"


def _recipe(target: str) -> str:
    m = re.search(rf"(?m)^{target}:.*\n((?:\t.*\n?)*)", (ROOT / "Makefile").read_text())
    assert m, f"no {target} target"
    return m.group(1)


def test_workflow_installs_then_validates():
    text = WORKFLOW.read_text()
    assert "pull_request" in text and "push" in text
    assert "ubuntu-latest" in text
    assert "astral-sh/setup-uv" in text and '"3.11"' in text
    runs = re.findall(r"(?m)^\s*run:\s*(.+)$", text)
    assert runs == ["make install", "make validate"], runs


def test_validate_target_stays_offline():
    recipe = _recipe("validate")
    for heavy in (
        "paper",
        "dataset",
        "sim",
        "repro-headline",
        "tectonic",
        "docker",
        "run_episodes",
    ):
        assert not re.search(rf"\b{re.escape(heavy)}\b", recipe), f"validate runs {heavy}"


def test_no_test_shells_out_to_tectonic_or_docker():
    # The paper build is `make paper`; the tests only read the .tex sources.
    hits = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        text = p.read_text()
        if p.name == Path(__file__).name or "shutil.which(" in text:
            continue  # a test that probes for the tool is the skip we ask for
        for line in text.splitlines():
            if re.search(r"\b(tectonic|docker)\b", line):
                hits.append(f"{p.name}: {line.strip()}")
    assert not hits, "tests referencing tectonic/docker must skip when absent:\n" + "\n".join(hits)
