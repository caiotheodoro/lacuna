"""What a run used, by hash: the part of a run's manifest that needs no docker.

`scripts/rig_run.py` adds the image digests, the commit and the run's parameters.
A run whose hashes differ from the registered ones fails the integrity gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """sha256 of each file, keyed by its path relative to `root`, in path order."""
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def tree_hash(root: Path, paths: Iterable[Path]) -> str:
    """One sha256 over a set of files: their relative paths and contents, in path order.

    The run driver takes it over the source the rig image is built from, so a run made
    from an uncommitted tree still says exactly which source it ran.
    """
    lines = [f"{rel} {digest}" for rel, digest in config_hashes(root, paths).items()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()
