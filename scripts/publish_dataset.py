#!/usr/bin/env python3
"""Validate dataset/ and show what would be uploaded. Dry run by default.

  uv run --group dataset python scripts/publish_dataset.py              # dry run
  uv run --group dataset python scripts/publish_dataset.py --dry-run    # the same

A real upload needs --yes-upload, typed by the person who decided to publish.
The repo is private unless --public is passed as well. Validation runs first;
the privacy scan runs again immediately before the upload; and the card must
have been built from the current HEAD with a clean tree. Any failure means
nothing is created and nothing is uploaded.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import git_commit  # noqa: E402
from validate_dataset import DATASET, check_privacy, report, validate  # noqa: E402

REPO = "caiotheodoro/lacuna-capture-histories"


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
    name, ok, detail = check_privacy(folder)
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
    upload(args.repo, folder, args.private, commit)
    print(f"\nuploaded {len(files)} files to {args.repo} ({visibility})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
