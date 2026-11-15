"""JSON lines: one object per line, as an access log is written."""

from __future__ import annotations

import json
from collections.abc import Iterable


def read(lines: Iterable[str]) -> list[dict]:
    """The lines that are JSON objects, in order. Anything else is skipped: a log
    file also holds what the process printed at start, and a last line can be cut."""
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out
