"""Keep the hand-written parameter reference in step with the estimators, for the Docs workflow.

The documentation site (``docs/``, built with MkDocs) documents every constructor parameter of
``TBoostRegressor`` / ``TBoostClassifier`` by hand. Two files are derived from the code and
the settings pages instead of being written by hand:

* ``docs/_snippets/tboost{regressor,classifier}_signature.md``, the class signatures shown on
  the class reference pages, generated from the constructor;
* ``docs/training-parameters/index.md``, the overview, generated from the first paragraph of
  each entry on the settings pages.

``check`` (what the Docs workflow runs) fails when a constructor parameter has no entry under
``docs/training-parameters/``, has more than one, or documents a default that differs from the
code; when an entry names a parameter the constructor does not have; or when a derived file is
stale. ``fix`` rewrites the derived files; entries for new parameters are written by hand.

The estimator source is parsed, not imported, so this needs neither the Rust toolchain nor a
built extension.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "python" / "t_boost" / "sklearn.py"
PARAM_DIR = ROOT / "docs" / "training-parameters"
OVERVIEW = PARAM_DIR / "index.md"
SIGNATURES = {
    "TBoostRegressor": ROOT / "docs" / "_snippets" / "tboostregressor_signature.md",
    "TBoostClassifier": ROOT / "docs" / "_snippets" / "tboostclassifier_signature.md",
}
# The order of the settings pages in the overview (and in mkdocs.yml's nav).
PAGES = (
    "common", "overfitting-detection", "quantization", "interaction", "categorical", "bagging",
    "pruning", "pruning-fold-vote", "banding", "graduation", "purification",
    "multiclassification", "performance", "advanced",
)
OVERVIEW_HEAD = """# Overview

These parameters are for the Python package classes `TBoostRegressor` and `TBoostClassifier`.
Both classes accept the same parameters; only the default value of `objective` differs.

`n_trees`, `learning_rate` and `lambda_` can be passed by position. Every other parameter is
keyword-only.
"""
_ENTRY = re.compile(r"(?ms)^## (\w+) \{#\w+\}\n(.*?)(?=^## |\Z)")


def _constructors() -> dict[str, list[tuple[str, object, bool]]]:
    """Each estimator's constructor parameters as ``(name, default, keyword_only)``."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    inits: dict[str, ast.arguments] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    inits[node.name] = item.args

    def value(default: ast.expr) -> object:
        if isinstance(default, ast.Name) and default.id in constants:
            return constants[default.id]
        return ast.literal_eval(default)

    out: dict[str, list[tuple[str, object, bool]]] = {}
    # TBoostRegressor inherits the shared constructor.
    for cls, owner in (("TBoostRegressor", "_BaseTBoost"), ("TBoostClassifier", "TBoostClassifier")):
        args = inits[owner]
        positional = args.args[1:]
        params = [
            (a.arg, value(d), False)
            for a, d in zip(positional[len(positional) - len(args.defaults):], args.defaults)
        ]
        params += [(a.arg, value(d), True) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d]
        out[cls] = params
    return out


def _signature(cls: str, params: list[tuple[str, object, bool]]) -> str:
    items = []
    for name, default, keyword_only in params:
        if keyword_only and "*" not in items:
            items.append("*")
        items.append(f"{name}={default!r}".replace("'", '"'))
    head = f"class {cls}("
    lines = [head + items[0] + ","]
    lines += [" " * len(head) + item + "," for item in items[1:-1]]
    lines.append(" " * len(head) + items[-1] + ")")
    return "```python\n" + "\n".join(lines) + "\n```\n"


def _overview() -> str:
    out = [OVERVIEW_HEAD]
    for page in PAGES:
        text = (PARAM_DIR / f"{page}.md").read_text(encoding="utf-8")
        title = text.splitlines()[0].removeprefix("# ")
        out.append(f"## [{title}]({page}.md)\n")
        for match in _ENTRY.finditer(text):
            name, body = match.groups()
            alias = re.search(r"^_Alias:_ .+$", body, re.M)
            description = body.split("#### Description", 1)[1].split("**Type**", 1)[0]
            paragraphs = [p.strip() for p in description.strip().split("\n\n")]
            lead = paragraphs[:1]
            # A one-line lead ("The learning rate.") takes its second sentence along with it.
            if (len(lead[0]) < 60 and len(paragraphs) > 1 and paragraphs[1].endswith(".")
                    and not paragraphs[1].startswith(("$$", "-", "!!!", "```"))):
                lead.append(paragraphs[1])
            out.append(f"### [{name}]({page}.md#{name})\n")
            if alias:
                out.append(alias.group(0) + "\n")
            for paragraph in lead:
                out.append(re.sub(r"\]\(#(\w+)\)", rf"]({page}.md#\1)", paragraph) + "\n")
    return "\n".join(out)


def _documented_default_matches(code: object, documented: str) -> bool:
    if code is None or isinstance(code, bool):
        return documented.startswith(str(code))
    if isinstance(code, (int, float)):
        number = re.match(r"[-+0-9.e ]+", documented)
        try:
            return number is not None and float(number.group(0).replace(" ", "")) == float(code)
        except ValueError:
            return False
    return documented.startswith(str(code))


def _entry_problems(params: list[tuple[str, object, bool]]) -> list[str]:
    entries: dict[str, list[tuple[str, str]]] = {}
    for page in PAGES:
        text = (PARAM_DIR / f"{page}.md").read_text(encoding="utf-8")
        for match in _ENTRY.finditer(text):
            name, body = match.groups()
            documented = re.search(r"\*\*Default value\*\*\n\n(.+)", body)
            entries.setdefault(name, []).append(
                (page, documented.group(1).strip() if documented else "")
            )
    problems = []
    known = {name for name, _, _ in params}
    for name, default, _ in params:
        found = entries.get(name, [])
        if not found:
            problems.append(f"`{name}` has no entry under docs/training-parameters/")
        elif len(found) > 1:
            problems.append(f"`{name}` has entries on several pages: {[p for p, _ in found]}")
        # `objective` documents one default per class.
        elif name != "objective" and not _documented_default_matches(default, found[0][1]):
            problems.append(
                f"`{name}` defaults to {default!r} but {found[0][0]}.md documents {found[0][1]!r}"
            )
    problems += [
        f"{pages[0][0]}.md documents `{name}`, which is not a constructor parameter"
        for name, pages in entries.items()
        if name not in known
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "fix"))
    args = parser.parse_args(argv)

    constructors = _constructors()
    generated = {path: _signature(cls, constructors[cls]) for cls, path in SIGNATURES.items()}
    generated[OVERVIEW] = _overview()
    if args.command == "fix":
        for path, text in generated.items():
            path.write_text(text, encoding="utf-8")

    problems = _entry_problems(constructors["TBoostRegressor"])
    problems += [
        f"{path.relative_to(ROOT)} is stale: run `python scripts/check_docs.py fix`"
        for path, text in generated.items()
        if path.read_text(encoding="utf-8") != text
    ]
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print(f"docs: {len(constructors['TBoostRegressor'])} parameters documented, generated files current")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
