# Install from PyPI

!!! note

    t-boost Python package supports only [CPython Python implementation](https://en.wikipedia.org/wiki/CPython).

To install t-boost from [PyPI](https://pypi.org/project/t-boost/):

1. Run one of the following commands:

    === "uv"

        ```no-highlight
        uv add t-boost
        ```

    === "pip"

        ```no-highlight
        pip install t-boost
        ```

    !!! note

        PyPI contains precompiled wheels for most commonly used platform configurations:

        | Operating system | CPU architectures |
        |------------------|-------------------|
        | macOS (10.12 or later on x86_64, 11.0 or later on arm64) | x86_64 and arm64 |
        | Linux (compatible with [manylinux2014 platform tag](https://peps.python.org/pep-0599/)) | x86_64 and aarch64 |
        | Windows | x86_64 |

        Each wheel is built for the stable ABI of CPython 3.10 (`abi3`), so the same wheel
        works with Python 3.10 and every later version.

        If the platform where the installation is performed is incompatible with platform tags
        of the available precompiled wheels then the installer will try to build t-boost
        python package from source. This approach requires a Rust toolchain to be set up
        before the installation (see [Build from source](build-from-source.md)).

    !!! note

        Release native binaries are built for the default portable CPU baseline of each
        architecture, so no extra instruction sets (such as AVX2) are required.

1. (Optionally) [Test t-boost](test.md).
