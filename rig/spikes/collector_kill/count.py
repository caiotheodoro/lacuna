"""Which numbered spans reached the collector's file, and where the gaps are."""

from __future__ import annotations

import json
import os
import sys

n, paths = int(sys.argv[1]), sys.argv[2:]
seen: list[int] = []
for path in paths:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            for rs in json.loads(line).get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for sp in ss.get("spans", []):
                        for attr in sp.get("attributes", []):
                            if attr["key"] == "spike.seq":
                                seen.append(int(attr["value"]["intValue"]))

got = set(seen)
missing = [i for i in range(n) if i not in got]
runs: list[tuple[int, int]] = []
for i in missing:
    if runs and runs[-1][1] == i - 1:
        runs[-1] = (runs[-1][0], i)
    else:
        runs.append((i, i))
print(
    json.dumps(
        {
            "emitted": n,
            "arrived": len(got),
            "duplicates": len(seen) - len(got),
            "missing": len(missing),
            "missing_share": round(len(missing) / n, 4) if n else None,
            "missing_runs": runs,
        }
    )
)
