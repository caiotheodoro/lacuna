"""Every number the README states must come from MEASUREMENT_CARD.json.

AGENTS.md rule 2. Four figures were once typed from the wrong artifact
(D26); this is what stops that reaching a reader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "MEASUREMENT_CARD.json"
README = ROOT / "README.md"

# Figures that are not measurements: table dimensions, section numbers,
# years, model names, and the nominal level the whole project is stated at.
EXEMPT = {"0", "1", "2", "3", "4", "95", "100", "2026", "4.1", "0.0.0"}


def _numbers(text: str) -> set[str]:
    # Only inside the Status section, which is where results are quoted.
    body = text.split("## Status", 1)[-1].split("## Run", 1)[0]
    body = re.sub(r"`[^`]*`", " ", body)  # code spans name files, not findings
    body = re.sub(r"\bD\d+\b", " ", body)  # D17 is a decision reference, not a measurement
    return {m for m in re.findall(r"\d+\.\d+|\d+", body)} - EXEMPT


def _card_values(obj, out: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            _card_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _card_values(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(str(obj))
        out.add(str(round(float(obj))))
        out.add(f"{float(obj) * 100:.0f}")  # a rate quoted as a percentage
        out.add(f"{float(obj):.1f}")
        out.add(f"{float(obj):.2f}")
    return out


@pytest.mark.skipif(not CARD.exists(), reason="card not built yet")
def test_readme_numbers_are_all_in_the_card():
    allowed = _card_values(json.loads(CARD.read_text()), set())
    missing = sorted(n for n in _numbers(README.read_text()) if n not in allowed)
    assert not missing, (
        f"README states figures absent from MEASUREMENT_CARD.json: {missing}. "
        "Either the card is stale (rerun scripts/make_card.py) or the claim is unsupported."
    )


@pytest.mark.skipif(not CARD.exists(), reason="card not built yet")
def test_card_names_its_sources():
    card = json.loads(CARD.read_text())
    for section in ("exploratory", "auditor"):
        assert card[section].get("source"), f"{section} does not name the artifact it came from"
    assert card["limitation"]
