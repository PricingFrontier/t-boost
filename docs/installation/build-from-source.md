# Build from source

## Dependencies and requirements {#dependencies-and-requirements}

- CPython 3.10 or later.
- A Rust toolchain, version 1.85 or later (for example, installed with
  [rustup](https://rustup.rs/)).
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`.

The Python extension is built with [maturin](https://www.maturin.rs/), which `uv` and `pip`
install automatically as the build backend.

## Build and install {#build-and-install}

1. Clone the repository:

    ```no-highlight
    git clone https://github.com/PricingFrontier/t-boost.git
    cd t-boost
    ```

1. Build the package:

    === "uv"

        ```no-highlight
        uv sync
        ```

        This creates `.venv`, builds the Rust extension in it and installs the development
        dependencies. `uv sync` does not notice changes to the Rust code: after editing it,
        rebuild with `uv sync --reinstall-package t-boost`.

    === "pip"

        ```no-highlight
        pip install .
        ```

1. (Optionally) [Test t-boost](test.md).

## Build a wheel package {#build-a-wheel-package}

To build a wheel package from a local copy of the repository, run the following command in
its root directory:

```no-highlight
uv build --wheel
```

The resulting wheel is written to the `dist` directory and can be installed with `uv pip
install` or `pip install`.
