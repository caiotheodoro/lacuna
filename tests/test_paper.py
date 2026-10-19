"""Every figure in paper/lacuna.tex must trace to an artifact, the frozen
pre-registration, or a code constant checked here.

Tables are generated (scripts/paper_tables.py) and excluded. Prose is not, and
prose is where a hand-typed number slips in: D26 caught four, the README gate
caught six more.

`make paper-check` additionally sets PAPER_FINAL=1, which fails while any
\\pending{} box or MISSING ARTIFACT table remains.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "lacuna.tex"
# The workshop version condenses the report; its numbers face the same gate.
TEXES = [TEX, ROOT / "paper" / "lacuna-workshop.tex"]
TABLES = ROOT / "paper" / "tables"

# Cited from another paper, with the citation next to it in the text.
CITED = {"4.1"}
# Names, not measurements: ISO/IEC 42001, SB 53, the 2^3 cells of a 3-witness table.
NAMES = {"42001", "53", "2", "3"}
# Structural integers: Gamma values, section-sized counts, rungs.
STRUCTURAL = {str(i) for i in range(0, 11)}


def _code_facts() -> set[str]:
    import inspect

    from lacuna.envs.tasks import make_world
    from lacuna.envs.tools import TOOLS
    from lacuna.estimators.diagnostics import DELTA_BIC
    from lacuna.estimators.latent_class import sample_posterior
    from lacuna.estimators.loglinear import CAP_RATIO

    sig = inspect.signature(make_world).parameters
    lc = inspect.signature(sample_posterior).parameters
    facts = {
        str(len(TOOLS)),  # "eleven tools" is spelled out, but keep the check honest
        str(sig["n_customers"].default),
        str(sig["n_orders"].default),
        str(int(DELTA_BIC)),
        # The methods appendix: search cap and the latent-class sampler.
        str(CAP_RATIO),
        *(str(lc[p].default) for p in ("k_max", "burn", "draws", "a_alpha", "b_alpha")),
    }
    return facts


def _numbers_in(obj, out: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        for x in (obj, abs(obj)):
            out.add(str(x))
            for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}"):
                out.add(fmt.format(float(x)))
            out.add(f"{float(x) * 100:.0f}")
    return out


def _allowed() -> set[str]:
    allowed: set[str] = set()
    for p in list((ROOT / "artifacts").rglob("*.json")) + [ROOT / "MEASUREMENT_CARD.json"]:
        try:
            _numbers_in(json.loads(p.read_text()), allowed)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    prereg = (ROOT / "docs" / "PRE-REGISTRATION.md").read_text()
    allowed |= set(re.findall(r"\d+(?:\.\d+)?", prereg))
    return allowed | CITED | NAMES | STRUCTURAL | _code_facts()


def _prose(tex: str) -> str:
    body = tex.split("\\begin{document}", 1)[1]
    body = re.sub(r"(?m)%.*$", "", body)  # comments
    body = re.sub(r"\\pending\{(?:[^{}]|\{[^{}]*\})*\}", " ", body)  # unfinished boxes
    body = re.sub(r"``.*?''", " ", body, flags=re.S)  # quoted model output
    body = re.sub(r"\\(?:input|label|ref|citep?|bibliography\w*)\{[^}]*\}", " ", body)
    body = re.sub(r"\\texttt\{[^}]*\}", " ", body)  # identifiers like fin_mid_rho100
    return body


@pytest.mark.parametrize("tex", TEXES, ids=lambda p: p.name)
def test_every_prose_number_has_a_source(tex):
    if not tex.exists():
        pytest.skip(f"{tex.name} not written")
    allowed = _allowed()
    found = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", _prose(tex.read_text())))
    missing = sorted(found - allowed, key=lambda x: float(x))
    assert not missing, f"{tex.name} states figures with no artifact behind them: {missing}"


@pytest.mark.skipif(os.environ.get("PAPER_FINAL") != "1", reason="draft: pending sections allowed")
def test_paper_is_finished():
    tex = TEX.read_text()
    assert "\\pending{" not in tex.split("\\begin{document}", 1)[1], "unfinished \\pending box"
    bad = [p.name for p in TABLES.glob("*.tex") if "MISSING ARTIFACT" in p.read_text()]
    assert not bad, f"tables without an artifact: {bad}"
