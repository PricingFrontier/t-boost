"""Static regression checks for the two source-verified CI findings."""

import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_bug019_fuzz_dictionary_exists_in_invocation_directory():
    workflow = (ROOT / ".github/workflows/fuzz.yml").read_text()
    assert "dtolnay/rust-toolchain@stable" in workflow
    assert "cargo +stable install cargo-fuzz --version 0.13.1 --locked" in workflow
    command = next(line.split("run:", 1)[1].strip() for line in workflow.splitlines()
                   if "run:" in line and "fuzz run fuzz_deserialize" in line)
    dictionary = next(arg.split("=", 1)[1] for arg in shlex.split(command)
                      if arg.startswith("-dict="))
    # cargo-fuzz preserves the invoking directory for libFuzzer arguments.
    assert (ROOT / dictionary).is_file(), f"libFuzzer dictionary is absent: {dictionary}"


def test_bug020_release_smoke_installs_built_artifact_explicitly():
    workflow = (ROOT / ".github/workflows/release-gate.yml").read_text()
    smoke = workflow.split("- name: Clean-env wheel smoke", 1)[1]
    installs = [shlex.split(line.strip()) for line in smoke.splitlines()
                if re.match(r"\s*python -m pip install\b", line)]
    artifact_installs = [args for args in installs if "t-boost" in args or
                         any("dist/" in arg and ".whl" in arg for arg in args)]
    assert artifact_installs, "smoke must install the artifact built in this job"
    for args in artifact_installs:
        assert "--no-index" in args or any("dist/" in arg and ".whl" in arg for arg in args), \
            "an unqualified index requirement can select the published package"
