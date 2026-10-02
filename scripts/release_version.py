"""Bump and check the release version, for the Release workflow.

t-boost's version lives in the Cargo workspace, not in pyproject.toml (whose version is
``dynamic``; maturin reads it from the py crate, which inherits the workspace's). One
release version therefore appears in five places, which must agree:

* ``[workspace.package] version`` in ``Cargo.toml`` (the source of truth),
* the internal ``t-boost-core`` requirement in ``crates/t-boost-py/Cargo.toml`` and
  ``xtask/Cargo.toml`` (cargo-deny bans bare path deps, so these carry a version),
* the ``t-boost-core``, ``t-boost-py`` and ``xtask`` entries in ``Cargo.lock``.

``bump`` rewrites all of them, for a release pull request. ``check`` is what the Release
workflow runs: it refuses a version the five places disagree on, one PyPI already has, or
one that is not newer than every release there, so an unbumped ``main`` cannot be
released twice. On success it writes the version to the GitHub output file so later jobs
can check, tag and name the release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT = "t-boost"
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_PINNED_MANIFESTS = ("crates/t-boost-py/Cargo.toml", "xtask/Cargo.toml")
_LOCKED_PACKAGES = ("t-boost-core", "t-boost-py", "xtask")
_BUMP_HINT = (
    "Bump the version in a pull request (`python scripts/release_version.py bump patch`, "
    "or minor / major / X.Y.Z), merge it, then run Release again."
)

# `version      = "0.6.0"   # comment` under [workspace.package]: replace only the quoted value.
_WORKSPACE_VERSION = re.compile(
    r'(\[workspace\.package\][^\[]*?^version\s*=\s*")([^"]*)(")', re.MULTILINE | re.DOTALL
)
_CORE_PIN = re.compile(r'(t-boost-core\s*=\s*\{[^}]*\bversion\s*=\s*")([^"]*)(")')


class ReleaseVersionError(Exception):
    """The declared version cannot be released."""


def _parts(version: str) -> tuple[int, int, int] | None:
    match = _VERSION.fullmatch(version)
    if match is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def _locked_entry(name: str) -> re.Pattern[str]:
    return re.compile(rf'(\[\[package\]\]\nname = "{re.escape(name)}"\nversion = ")([^"]*)(")')


def declared_versions(root: Path) -> dict[str, str]:
    """Every place the release version is recorded, mapped to the version found there."""
    found: dict[str, str] = {}
    with (root / "Cargo.toml").open("rb") as stream:
        found["Cargo.toml [workspace.package]"] = str(
            tomllib.load(stream)["workspace"]["package"]["version"]
        )
    for manifest in _PINNED_MANIFESTS:
        with (root / manifest).open("rb") as stream:
            dep = tomllib.load(stream)["dependencies"]["t-boost-core"]
        found[f"{manifest} t-boost-core"] = str(dep["version"])
    with (root / "Cargo.lock").open("rb") as stream:
        locked = {p["name"]: str(p["version"]) for p in tomllib.load(stream)["package"]}
    for name in _LOCKED_PACKAGES:
        found[f"Cargo.lock {name}"] = locked[name]
    return found


def declared_version(root: Path) -> str:
    """The release version, once every place that records it agrees."""
    found = declared_versions(root)
    if len(set(found.values())) != 1:
        detail = "; ".join(f"{where}: {version}" for where, version in found.items())
        raise ReleaseVersionError(f"the version is not bumped consistently ({detail}). {_BUMP_HINT}")
    return next(iter(found.values()))


def released_versions(pypi_json: Path | None) -> list[str]:
    """The versions PyPI's JSON API lists; none for a project never published."""
    if pypi_json is None:
        return []
    document = json.loads(pypi_json.read_text(encoding="utf-8"))
    return sorted(document["releases"])


def check_release_version(version: str, released: Iterable[str]) -> str:
    """Return *version* when it is a new ``X.Y.Z`` release, else raise.

    Releases PyPI has in another form (a pre-release, say) cannot collide with
    an ``X.Y.Z`` version and are not compared.
    """
    parts = _parts(version)
    if parts is None:
        raise ReleaseVersionError(f"the declared version is {version!r}; a release must be X.Y.Z.")
    published = set(released)
    if version in published:
        raise ReleaseVersionError(f"{PROJECT} {version} is already on PyPI. {_BUMP_HINT}")
    comparable = {parsed: text for text in published if (parsed := _parts(text)) is not None}
    if comparable and parts <= max(comparable):
        latest = comparable[max(comparable)]
        raise ReleaseVersionError(
            f"{PROJECT} {version} is not newer than {latest}, the latest release on PyPI. "
            f"{_BUMP_HINT}"
        )
    return version


def bumped(current: str, bump: str) -> str:
    """The version *bump* (``patch``/``minor``/``major`` or an explicit ``X.Y.Z``) yields."""
    parts = _parts(current)
    if parts is None:
        raise ReleaseVersionError(f"the current version {current!r} is not X.Y.Z")
    major, minor, patch = parts
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    target = _parts(bump)
    if target is None:
        raise ReleaseVersionError(f"{bump!r} is not patch, minor, major or an X.Y.Z version")
    if target <= parts:
        raise ReleaseVersionError(f"{bump} is not newer than the current {current}")
    return bump


def _replace_one(text: str, pattern: re.Pattern[str], version: str, where: str) -> str:
    new, count = pattern.subn(lambda m: f"{m[1]}{version}{m[3]}", text, count=1)
    if count != 1:
        raise ReleaseVersionError(f"could not find the version in {where}")
    return new


def mask_version(filename: str, text: str) -> str:
    """*text* of the root ``Cargo.toml`` or ``Cargo.lock`` with the release version blanked out.

    Two revisions whose masked texts are equal differ at most by a version bump.
    """
    if filename == "Cargo.toml":
        return _WORKSPACE_VERSION.sub(lambda m: f"{m[1]}*{m[3]}", text, count=1)
    if filename == "Cargo.lock":
        for name in _LOCKED_PACKAGES:
            text = _locked_entry(name).sub(lambda m: f"{m[1]}*{m[3]}", text, count=1)
        return text
    raise ValueError(f"no release version is recorded in {filename}")


def write_version(root: Path, version: str) -> None:
    """Rewrite every place the release version is recorded to *version*."""
    edits: list[tuple[Path, list[tuple[re.Pattern[str], str]]]] = [
        (root / "Cargo.toml", [(_WORKSPACE_VERSION, "[workspace.package] version")]),
        *((root / m, [(_CORE_PIN, "the t-boost-core requirement")]) for m in _PINNED_MANIFESTS),
        (root / "Cargo.lock", [(_locked_entry(n), f"the {n} entry") for n in _LOCKED_PACKAGES]),
    ]
    rewritten: dict[Path, str] = {}
    for path, patterns in edits:  # compute every edit before writing any file
        text = path.read_text(encoding="utf-8")
        for pattern, what in patterns:
            text = _replace_one(text, pattern, version, f"{path.relative_to(root)} ({what})")
        rewritten[path] = text
    for path, text in rewritten.items():
        path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    bump = commands.add_parser("bump", help="rewrite the version for a release pull request")
    bump.add_argument("bump", help="patch, minor, major, or an explicit X.Y.Z")
    check = commands.add_parser("check", help="refuse a version that cannot be released")
    check.add_argument(
        "--pypi-json",
        type=Path,
        help="PyPI's JSON API response for the project; omit when it has never been published",
    )
    check.add_argument(
        "--github-output",
        type=Path,
        required=True,
        help="file the version is appended to as `version=X.Y.Z`",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "bump":
            current = declared_version(args.root)
            version = bumped(current, args.bump)
            write_version(args.root, version)
            print(f"Bumped {PROJECT} {current} -> {version}.")
            return 0
        version = check_release_version(
            declared_version(args.root), released_versions(args.pypi_json)
        )
    except ReleaseVersionError as exc:
        print(f"::error::{exc}" if args.command == "check" else f"error: {exc}")
        return 1
    with args.github_output.open("a", encoding="utf-8") as stream:
        stream.write(f"version={version}\n")
    print(f"Releasing {PROJECT} {version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
