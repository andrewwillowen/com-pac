"""CLI end-to-end regression tests."""

from pathlib import Path
import csv

import numpy as np
import pytest


pytestmark = pytest.mark.integration


def _read_csv_sections(csv_path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current_section: list[list[str]] = []

    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not any(cell.strip() for cell in row):
                if current_section:
                    section_name = current_section[0][0].strip()
                    sections[section_name] = current_section
                    current_section = []
                continue

            current_section.append(row)

    if current_section:
        section_name = current_section[0][0].strip()
        sections[section_name] = current_section

    return sections


def _get_numeric_values(section_rows: list[list[str]]) -> np.ndarray:
    values: list[float] = []

    for row in section_rows[1:]:
        for cell in row[1:]:
            value = cell.strip()
            if value == "":
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue

    return np.array(values, dtype=float)


@pytest.fixture
def generated_cli_paths(tmp_path: Path, legacy_input_path: Path):
    input_copy = tmp_path / legacy_input_path.name
    input_copy.write_text(
        legacy_input_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    input_base = input_copy.stem
    csv_output = tmp_path / f"{input_base}_pac.csv"
    text_output = tmp_path / f"{input_base}_pac.out"
    return input_copy, csv_output, text_output


class Test_cli_regression:
    def test_cli_creates_expected_outputs(self, run_cli, generated_cli_paths):
        input_path, csv_output, text_output = generated_cli_paths
        result = run_cli(input_path)

        assert result.returncode == 0, (
            "CLI command failed unexpectedly.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert csv_output.exists()
        assert text_output.exists()

    def test_csv_sections_match_legacy_regression(
        self,
        run_cli,
        generated_cli_paths,
        golden_csv_path: Path,
    ):
        input_path, csv_output, _ = generated_cli_paths
        result = run_cli(input_path)

        assert result.returncode == 0, (
            "CLI command failed unexpectedly.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        generated_sections = _read_csv_sections(csv_output)
        golden_sections = _read_csv_sections(golden_csv_path)

        target_sections = (
            "Rotational Constants",
            "Dipole Components",
            "Principal Axes Coordinates",
            "Atomic Masses",
        )

        for section_name in target_sections:
            assert section_name in generated_sections
            assert section_name in golden_sections

            generated_header = generated_sections[section_name][1]
            golden_header = golden_sections[section_name][1]
            assert generated_header == golden_header

            generated_values = _get_numeric_values(generated_sections[section_name])
            golden_values = _get_numeric_values(golden_sections[section_name])

            assert generated_values.shape == golden_values.shape
            np.testing.assert_allclose(
                generated_values, golden_values, rtol=1e-7, atol=1e-9
            )

    def test_missing_input_fails_without_outputs(self, run_cli, tmp_path: Path):
        missing_input = tmp_path / "does_not_exist.txt"
        result = run_cli(missing_input)

        assert result.returncode != 0
        assert not (tmp_path / "does_not_exist_pac.csv").exists()
        assert not (tmp_path / "does_not_exist_pac.out").exists()
