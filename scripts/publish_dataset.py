#!/usr/bin/env python3
"""Check dataset/ from scratch and show what would be uploaded. Dry run by default.

  uv run --group dataset python scripts/publish_dataset.py              # dry run
  uv run --group dataset python scripts/publish_dataset.py --dry-run    # the same

A real upload needs --yes-upload, typed by the person who decided to publish.
The repo is private unless --public is passed as well.

Nothing from an earlier run is trusted. In order, and any failure stops it:
1. this script's own release check (no transcript, no provider reasoning text),
   which does not depend on the validator;
2. the dataset tests, run now;
3. the validator, run now, including the `datasets` load;
4. the privacy scan again, on the bytes about to leave;
5. the card must have been built from the current HEAD with a clean tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import git_commit  # noqa: E402
from validate_dataset import DATASET, check_privacy, report, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "caiotheodoro/lacuna-capture-histories"
TESTS = ("tests/test_dataset.py", "tests/test_dataset_mutations.py")
# Written here, not imported: the release check must not share code with the
# validator it backs up.
_REASONING_KEY = re.compile(r'\\*"reasoning_content\\*"\s*:')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list what would go up (default)")
    mode.add_argument("--yes-upload", action="store_true", help="create the repo and upload")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--private", action="store_true", help="private repo (default)")
    vis.add_argument("--public", action="store_true", help="public repo")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dataset-dir", type=Path, default=DATASET)
    args = ap.parse_args(argv)
    args.dry_run = not args.yes_upload
    args.private = not args.public
    return args


def release_blockers(folder: Path) -> list[str]:
    """Transcripts or provider reasoning anywhere in the folder. Independent of the validator."""
    found = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(folder))
        text = p.read_text(errors="replace")
        if "transcript" in p.name.lower():
            found.append(f"{rel}: a transcript file")
        if _REASONING_KEY.search(text):
            found.append(f"{rel}: provider reasoning (a reasoning_content key)")
        if p.suffix == ".jsonl":
            for line in text.splitlines():
                row = json.loads(line) if line.strip() else {}
                if "transcript" in row or row.get("transcripts_included") is True:
                    found.append(f"{rel}: a transcript in a row")
                    break
    return found


def run_tests() -> tuple[bool, str]:
    """The dataset tests, run now rather than trusted from an earlier run."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    tail = (proc.stdout.strip().splitlines() or ["no output"])[-1]
    return proc.returncode == 0, tail


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)


def card_commit(dataset_dir: Path) -> str:
    m = re.search(r"from lacuna commit `([^`]+)`", (dataset_dir / "README.md").read_text())
    return m.group(1) if m else "unknown"


def upload_refusal(card: str, head: str) -> str | None:
    """Why an upload must not happen from this build, or None if it may."""
    if "unknown" in (card, head):
        return f"cannot tell which commit built the card ({card}) or is checked out ({head})"
    if head.endswith("-dirty"):
        return f"the working tree at {head} has uncommitted changes"
    if card != head:
        return f"the card was built from {card} but HEAD is {head}; rebuild"
    return None


def upload(repo: str, folder: Path, private: bool, commit: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    # create_repo(exist_ok=True) leaves an existing repo's visibility alone, so
    # a repo someone made public would silently receive a "private" upload.
    actual = api.dataset_info(repo).private
    if actual != private:
        raise SystemExit(f"{repo} is {'private' if actual else 'public'}; refusing to upload")
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(folder),
        commit_message=f"lacuna capture histories from commit {commit}",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    folder = args.dataset_dir

    blockers = release_blockers(folder)
    print(f"RELEASE CHECK: {'FAIL' if blockers else 'PASS'}")
    for b in blockers[:10]:
        print(f"  {b}")
    if blockers:
        print("\nThis build carries transcripts. Nothing created, nothing uploaded.")
        return 1

    ok, tail = run_tests()
    print(f"TESTS: {'PASS' if ok else 'FAIL'} ({tail})")
    if not ok:
        print("\nThe dataset tests fail. Nothing created, nothing uploaded.")
        return 1

    print("VALIDATION")
    if not report(validate(folder)):
        print("\nValidation failed. Nothing created, nothing uploaded.")
        return 1

    files = sorted(p for p in folder.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    visibility = "private" if args.private else "PUBLIC"
    commit, head = card_commit(folder), git_commit()
    print(f"\n{'=' * 72}")
    print(f"{args.repo}  (dataset, {visibility})  {len(files)} files, {human(total)}")
    print(f"card built from lacuna commit {commit}; HEAD is {head}")
    print("=" * 72)
    for p in files:
        print(f"  {human(p.stat().st_size):>8}  {p.relative_to(folder)}")

    # Again, on the bytes about to leave, not only on what validation saw.
    _, ok, detail = check_privacy(folder)
    print(f"\npre-upload re-scan: [{'PASS' if ok else 'FAIL'}] {detail}")
    refusal = upload_refusal(commit, head)
    print(f"upload gate: {'would refuse: ' + refusal if refusal else 'card matches a clean HEAD'}")

    if args.dry_run:
        print(f"\nDry run. Nothing uploaded, no repo created on the Hub. Target: {args.repo}.")
        return 0 if ok else 1
    if not ok:
        print("\nThe privacy re-scan failed. Nothing uploaded.")
        return 1
    if refusal:
        print(f"\n{refusal}. Nothing uploaded.")
        return 1
    if release_blockers(folder):
        print("\nTranscripts appeared after validation. Nothing uploaded.")
        return 1
    upload(args.repo, folder, args.private, commit)
    print(f"\nuploaded {len(files)} files to {args.repo} ({visibility})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
