"""What the estimator needs to import: numpy, and nothing an agent run needs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _modules_after(imports: str) -> set[str]:
    code = f"import sys\n{imports}\nprint('\\n'.join(sorted(sys.modules)))"
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return set(r.stdout.split())


def test_core_imports_no_environment_and_no_agent():
    mods = _modules_after("import lacuna.core.tables, lacuna.core.records")
    assert not [m for m in mods if m.startswith(("lacuna.envs", "lacuna.agents"))]
    assert "httpx" not in mods


def test_estimators_and_environments_import_without_httpx():
    mods = _modules_after(
        "import lacuna.estimators.stratified, lacuna.estimators.loglinear\n"
        "import lacuna.linkage, lacuna.thinning, lacuna.envs.registry"
    )
    assert "httpx" not in mods


def test_the_prompt_is_one_object_wherever_it_is_imported_from():
    from lacuna.agents.llm import AGENT_SYSTEM as from_llm
    from lacuna.envs.prompts import AGENT_SYSTEM as from_prompts
    from lacuna.envs.registry import BACKOFFICE

    assert from_llm is from_prompts is BACKOFFICE.agent_system


def test_ingest_the_kernel_service_and_the_rig_helpers_need_no_rig_extra():
    mods = _modules_after(
        "import lacuna.ingest.otlp, lacuna.ingest.jsonl, lacuna.envs.service\n"
        "import lacuna.rig.sessions, lacuna.rig.stress, lacuna.rig.manifest, lacuna.rig.witnesses"
    )
    heavy = ("mcp", "opentelemetry", "httpx", "starlette", "uvicorn")
    assert not [m for m in mods if m.split(".")[0] in heavy]


def test_the_estimator_does_not_import_the_rig():
    mods = _modules_after(
        "import lacuna.core.tables, lacuna.estimators.stratified, lacuna.ingest.otlp"
    )
    assert not [m for m in mods if m.startswith("lacuna.rig")]
