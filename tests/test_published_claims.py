"""Every number the README states must come from MEASUREMENT_CARD.json.

AGENTS.md rule 2. Four figures were once typed from the wrong artifact
(D26); this is what stops that reaching a reader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import claims_lib
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
        # "undercounting by 93.63" states the magnitude of a bias of -93.63.
        out.add(str(abs(obj)))
        out.add(f"{abs(float(obj)):.1f}")
        out.add(f"{abs(float(obj)):.2f}")
        out.add(str(round(float(obj))))
        out.add(f"{float(obj) * 100:.0f}")  # a rate quoted as a percentage
        out.add(f"{float(obj):.1f}")
        out.add(f"{float(obj):.2f}")
    return out


@pytest.mark.skipif(not CARD.exists(), reason="card not built yet")
def test_readme_numbers_are_all_in_the_card():
    allowed = _card_values(json.loads(CARD.read_text()), set())
    # Thresholds and design constants from the frozen pre-registration are
    # stated facts, not measurements.
    prereg = (ROOT / "docs" / "PRE-REGISTRATION.md").read_text()
    allowed |= set(re.findall(r"\d+(?:\.\d+)?", prereg))
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


# --- key-aware gate ---------------------------------------------------------
# The test above passes a number found anywhere in the card. This one binds
# each Status number to the artifact cell it was read from (tests/claims_map.json).


def _status_numbers() -> tuple[str, dict[int, str]]:
    """The squashed Status text and {offset: number} for each figure needing a source."""
    body = claims_lib.squash(README.read_text().split("## Status", 1)[-1].split("## Run", 1)[0])
    masked = re.sub(r"`[^`]*`", lambda m: " " * len(m.group()), body)
    masked = re.sub(r"\bD\d+\b", lambda m: " " * len(m.group()), masked)
    return body, {
        m.start(): m.group()
        for m in re.finditer(r"\d+\.\d+|\d+", masked)
        if m.group() not in EXEMPT
    }


def _claims() -> list[dict]:
    return json.loads(claims_lib.MAP.read_text())["readme_status"]


def test_every_status_number_is_mapped_to_an_artifact_value():
    body, numbers = _status_numbers()
    problems = claims_lib.unmapped_and_failing(body, numbers, _claims())
    assert not problems, "README Status numbers not backed by their mapped artifact cell:\n" + (
        "\n".join(problems)
    )


def test_a_number_from_the_wrong_cell_is_caught():
    # 0.946 appears in the card for several cells; the old gate accepted all of
    # them. Pointing a claim at another cell must fail even when the value
    # exists elsewhere.
    body, numbers = _status_numbers()
    claims = [dict(c) for c in _claims()]
    claim = next(c for c in claims if "[[0.893]] over 3000" in c["context"])
    claim["path"] = ["runs", "confirm", "fin_mid/oracle", "ll_bic", "coverage"]  # 0.948
    problems = claims_lib.unmapped_and_failing(body, numbers, claims)
    assert any("0.893" in p and "0.948" in p for p in problems), problems


def test_a_new_unmapped_number_is_caught():
    body, numbers = _status_numbers()
    pos = len(body) + len(" covers ")
    problems = claims_lib.unmapped_and_failing(
        body + " covers 0.777 here", {**numbers, pos: "0.777"}, _claims()
    )
    assert any("0.777" in p and "no claim" in p for p in problems), problems
