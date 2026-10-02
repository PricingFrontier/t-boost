"""Decide whether a change touches the model, for CI and the Release workflow.

The slow core tests (the ones marked ``#[ignore = "slow: ..."]``) only need to run when
something that can change a fitted model changed: the core crate, or the root
``Cargo.toml`` / ``Cargo.lock`` (dependency versions, build profiles). A diff in those two
files that is only a release-version bump does not count. Prints ``true`` or ``false``;
with ``--github-output`` it also appends ``model_changed=true|false`` to that file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from release_version import mask_version

MODEL_PATHS = ("crates/t-boost-core/",)
MANIFESTS = ("Cargo.toml", "Cargo.lock")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _show(rev: str, path: str) -> str | None:
    shown = _git("show", f"{rev}:{path}")
    return shown.stdout if shown.returncode == 0 else None


def model_changed(base: str, head: str) -> tuple[bool, str]:
    """Whether *head* changes the model relative to *base*, and why."""
    if not base.strip("0") or _git("cat-file", "-e", f"{base}^{{commit}}").returncode != 0:
        return True, f"no usable base revision ({base or 'none'}); treating as a model change"
    diff = _git("diff", "--name-only", base, head)
    if diff.returncode != 0:
        return True, f"git diff failed ({diff.stderr.strip()}); treating as a model change"
    changed = diff.stdout.split()
    for path in changed:
        if path.startswith(MODEL_PATHS):
            return True, f"{path} changed"
    for manifest in MANIFESTS:
        if manifest in changed:
            before, after = _show(base, manifest), _show(head, manifest)
            if before is None or after is None:
                return True, f"{manifest} was added or removed"
            if mask_version(manifest, before) != mask_version(manifest, after):
                return True, f"{manifest} changed beyond the release version"
    return False, f"{len(changed)} changed file(s), none of them model code"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", help="the revision to compare against (empty or all zeros: none)")
    parser.add_argument("--head", default="HEAD", help="the revision being checked")
    parser.add_argument("--github-output", type=Path, help="file to append `model_changed=...` to")
    args = parser.parse_args(argv)
    changed, reason = model_changed(args.base, args.head)
    print(f"model changed: {str(changed).lower()} ({reason})", file=sys.stderr)
    print(str(changed).lower())
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"model_changed={str(changed).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
