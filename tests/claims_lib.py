"""Key-aware claim checking, shared by the README and paper gates.

The older gates ask whether a quoted number appears *anywhere* in the card or
the artifacts. That passes a number copied from the wrong cell whenever some
other cell happens to hold the same value (0.946 is in the card four times).
`tests/claims_map.json` instead binds each quoted number, found by its
surrounding words, to one JSON path in one artifact; the check is that this
value, rounded half-up to the decimals the text shows, is the text.
"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "tests" / "claims_map.json"
MARK = re.compile(r"\[\[([^\]]+)\]\]")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def locate(body: str, context: str) -> tuple[int, str]:
    """Position and text of the number a claim's context marks in `body`.

    `body` must have its whitespace squashed. The context, brackets removed,
    has to occur exactly once, so a claim cannot silently bind to another copy
    of the same words.
    """
    m = MARK.search(context)
    assert m, f"no [[number]] marker in context {context!r}"
    fragment = MARK.sub(lambda x: x.group(1), context)
    hits = [h.start() for h in re.finditer(re.escape(fragment), body)]
    assert len(hits) == 1, f"context {context!r} occurs {len(hits)} times, need exactly 1"
    return hits[0] + m.start(), m.group(1)


def resolve(claim: dict) -> float:
    """The artifact value a claim points at, after its transform."""
    src = ROOT / claim["source"]
    assert src.exists(), f"{claim['source']} does not exist"
    if claim.get("constant"):
        raise ValueError("constants have no path")
    value = _dig(json.loads(src.read_text()), claim["path"], claim["source"])
    transform = claim.get("transform")
    if transform == "span":  # [first, last] seed -> how many seeds
        return value[1] - value[0] + 1
    if transform == "span_times":  # seeds x replicates per seed
        times = _dig(json.loads(src.read_text()), claim["times_path"], claim["source"])
        return (value[1] - value[0] + 1) * times
    if transform == "abs":
        return abs(value)
    assert transform is None, f"unknown transform {transform!r}"
    return value


def _dig(obj, path: list, where: str):
    for key in path:
        try:
            obj = obj[key]
        except (KeyError, IndexError, TypeError):
            raise AssertionError(f"{where}: no {path!r} (stopped at {key!r})") from None
    return obj


def rounded(value: float, quoted: str) -> str:
    """`value` half-up to the decimals `quoted` shows, as text."""
    decimals = len(quoted.split(".")[1]) if "." in quoted else 0
    step = Decimal(1).scaleb(-decimals)
    return str(Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP))


def check(claim: dict, quoted: str) -> str | None:
    """None if the claim holds, else why not."""
    if claim.get("constant"):
        text = (ROOT / claim["source"]).read_text()
        if not re.search(rf"(?<![\w.]){re.escape(quoted)}(?![\w])", text):
            return f"{quoted} is not stated in {claim['source']}"
        return None
    try:
        value = resolve(claim)
    except AssertionError as e:
        return str(e)
    got = rounded(value, quoted)
    if got != quoted:
        return f"{claim['source']} {claim['path']} = {value}, which reads {got}, not {quoted}"
    return None


def unmapped_and_failing(body: str, numbers: dict[int, str], claims: list[dict]) -> list[str]:
    """Problems found comparing the numbers in `body` with the claims.

    `numbers` maps start position to the number text for every figure that
    needs a source. Each must have a claim, each claim must find its number,
    and each must round to it.
    """
    problems: list[str] = []
    claimed: dict[int, dict] = {}
    for claim in claims:
        try:
            pos, quoted = locate(body, claim["context"])
        except AssertionError as e:
            problems.append(str(e))
            continue
        if numbers.get(pos) != quoted:
            problems.append(f"{claim['context']!r} marks a number the text does not count")
        if pos in claimed:
            problems.append(f"{claim['context']!r} duplicates another claim")
        claimed[pos] = claim
        why = check(claim, quoted)
        if why:
            problems.append(f"{claim['context']!r}: {why}")
    for pos, n in sorted(numbers.items()):
        if pos not in claimed:
            problems.append(
                f"{n} at ...{body[max(0, pos - 40) : pos + len(n) + 20]!r} has no claim"
            )
    return problems
