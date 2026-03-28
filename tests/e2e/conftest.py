"""Fixtures for CLI end-to-end regression tests."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def cli_command() -> str:
    for command in ("com_pac", "com-pac", "principal_axes_calculator", "compac"):
        if shutil.which(command):
            return command

    pytest.skip("No com-pac CLI entrypoint found in PATH for integration tests")


@pytest.fixture(scope="session")
def legacy_input_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "latest.txt"


@pytest.fixture(scope="session")
def golden_csv_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "original_pac.csv"


@pytest.fixture
def run_cli(cli_command: str):
    def _run_cli(
        input_path: Path, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [cli_command, str(input_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return _run_cli
