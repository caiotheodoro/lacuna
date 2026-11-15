"""OTLP JSON, as the Collector's file exporter writes it: one export request per line."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def _value(v: dict) -> Any:
    """An OTLP AnyValue as a Python value. Integers travel as strings in OTLP JSON."""
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return _attrs(v["kvlistValue"].get("values", []))
    return None


def _attrs(pairs: list[dict] | None) -> dict[str, Any]:
    return {p["key"]: _value(p.get("value", {})) for p in pairs or []}


def _requests(lines: Iterable[str]) -> Iterable[dict]:
    for line in lines:
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a kill is not a record
        if isinstance(d, dict):
            yield d


def read_spans(lines: Iterable[str]) -> list[dict]:
    """Every span once, in file order.

    A retried export can deliver a span twice; the second copy of a
    (trace id, span id) is dropped. A span with either id empty is never a copy.
    """
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for req in _requests(lines):
        for rs in req.get("resourceSpans", []):
            resource = _attrs(rs.get("resource", {}).get("attributes"))
            for ss in rs.get("scopeSpans", []):
                for sp in ss.get("spans", []):
                    ident = (sp.get("traceId", ""), sp.get("spanId", ""))
                    # A span with no id cannot be told from another, so none is a copy.
                    if all(ident):
                        if ident in seen:
                            continue
                        seen.add(ident)
                    out.append(
                        {
                            "trace_id": ident[0],
                            "span_id": ident[1],
                            "parent_span_id": sp.get("parentSpanId", ""),
                            "name": sp.get("name", ""),
                            "start_ns": int(sp.get("startTimeUnixNano", 0)),
                            "end_ns": int(sp.get("endTimeUnixNano", 0)),
                            "resource": resource,
                            "attributes": _attrs(sp.get("attributes")),
                        }
                    )
    return out


def count_spans(lines: Iterable[str]) -> int:
    """Every span in the file, copies included: `read_spans` returns this many or fewer."""
    return sum(
        len(ss.get("spans", []))
        for req in _requests(lines)
        for rs in req.get("resourceSpans", [])
        for ss in rs.get("scopeSpans", [])
    )


def read_logs(lines: Iterable[str]) -> list[dict]:
    """Every log record, in file order: its body, its attributes and its resource's."""
    out: list[dict] = []
    for req in _requests(lines):
        for rl in req.get("resourceLogs", []):
            resource = _attrs(rl.get("resource", {}).get("attributes"))
            for sl in rl.get("scopeLogs", []):
                for rec in sl.get("logRecords", []):
                    out.append(
                        {
                            "body": _value(rec.get("body", {})),
                            "observed_ns": int(rec.get("observedTimeUnixNano", 0)),
                            "resource": resource,
                            "attributes": _attrs(rec.get("attributes")),
                        }
                    )
    return out
