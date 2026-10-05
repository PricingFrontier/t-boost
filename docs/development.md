# Development and contributions

t-boost is developed on [GitHub](https://github.com/PricingFrontier/t-boost). It has a Rust core
(`crates/t-boost-core`), Python bindings (`crates/t-boost-py`) and a Python package
(`python/t_boost`).

## Report a problem {#issues}

Report bugs and ask questions in the
[issue tracker](https://github.com/PricingFrontier/t-boost/issues). Include the t-boost version
(`t_boost.__version__`), a minimal example and the full error message.

## Contribute {#contribute}

See [CONTRIBUTING.md](https://github.com/PricingFrontier/t-boost/blob/main/CONTRIBUTING.md) for
the development environment, the checks every change must pass and the code rules. To build the
package from source, see [Build from source](installation/build-from-source.md).

## Documentation {#documentation}

This site is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from the `docs` directory of
the repository. To preview it locally:

```no-highlight
uv run --no-project --with-requirements docs/requirements.txt mkdocs serve
```

The site is rebuilt on every pull request and published from `main`.

## License {#license}

t-boost is released under the
[Apache-2.0 license](https://github.com/PricingFrontier/t-boost/blob/main/LICENSE).
