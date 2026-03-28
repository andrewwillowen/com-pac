"""
Unit tests for functions in writer.py
"""

import pytest
import tempfile
import os
import re
import numpy as np
from com_pac.writer import (
    generate_output_file,
    generate_csv_output,
    _build_preamble_section,
    _build_input_section,
    _build_atomic_masses_section,
    _build_rotational_constants_section,
    _build_dipole_components_section,
    _build_com_values_section,
    _build_com_coordinates_section,
    _build_com_inertias_section,
    _build_eigens_section,
    _build_pa_inertias_section,
    _build_results_section,
    header_creator,
    df_text_export,
)


def _parse_float_values(text):
    return np.array(
        [
            float(x)
            for x in re.findall(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", text)
        ],
        dtype=np.float64,
    )


def _extract_section_payload(text):
    return text.split("\n", 3)[-1]


def _extract_iso_entries(section_text, isotopologue_names):
    payload = _extract_section_payload(section_text)
    start_indexes = [payload.find(f"{iso}\n") for iso in isotopologue_names]
    start_indexes = [idx for idx in start_indexes if idx >= 0]
    if not start_indexes:
        return {}

    iso_block_text = payload[min(start_indexes) :].strip()
    entries = {}
    for entry in re.split(r"\n\n=+\n", iso_block_text):
        lines = entry.splitlines()
        if not lines:
            continue
        iso_name = lines[0].strip()
        entries[iso_name] = "\n".join(lines[1:])
    return entries


def _parse_iso_numeric_entries(section_text, isotopologue_names):
    entries = _extract_iso_entries(section_text, isotopologue_names)
    return {iso: _parse_float_values(entries[iso]) for iso in isotopologue_names}


def _parse_eigenvalues_per_iso(section_text, isotopologue_names):
    entries = _extract_iso_entries(section_text, isotopologue_names)
    parsed = {}
    for iso in isotopologue_names:
        iso_entry = entries.get(iso, "")
        parts = iso_entry.split("Eigenvalues")
        if len(parts) < 2:
            parsed[iso] = np.array([], dtype=np.float64)
            continue
        values_line = parts[-1].strip().split("\n")[0]
        parsed[iso] = np.array(
            [float(x) for x in values_line.split()],
            dtype=np.float64,
        )
    return parsed


def _parse_csv_section(csv_text, section_name):
    """Extract a section from CSV output and return as lines."""
    lines = csv_text.strip().split("\n")
    section_lines = []
    in_section = False

    for i, line in enumerate(lines):
        if line.strip() == section_name:
            in_section = True
            continue
        elif in_section:
            # Empty line signals end of section
            if line.strip() == "":
                break
            section_lines.append(line)

    return section_lines


def _extract_csv_numeric_values(section_lines):
    """Extract numeric values from CSV section lines (excluding headers)."""
    import csv
    import io

    if len(section_lines) < 2:
        return np.array([], dtype=np.float64)

    # Parse CSV
    csv_reader = csv.reader(io.StringIO("\n".join(section_lines)))
    rows = list(csv_reader)

    # Extract numeric values, skipping header row and first column (row labels)
    values = []
    for row in rows[1:]:  # Skip header
        for cell in row[1:]:  # Skip first column (labels)
            try:
                values.append(float(cell))
            except ValueError:
                pass

    return np.array(values, dtype=np.float64)


class Test_header_creator:
    def test_expected_output(self):
        """Test header_creator produces expected header format"""
        result = header_creator("Test Section")
        expected = """# ============== #
#  Test Section  #
# ============== #"""
        assert result == expected

    def test_bad_input(self):
        """Test header_creator raises TypeError on bad input"""

        class NoStrMethod:
            def __str__(self):
                raise Exception("Cannot convert to string")

        with pytest.raises(TypeError) as exc:
            header_creator(NoStrMethod())
        assert exc.type is TypeError and "Bad input passed to header_creator" in str(
            exc.value
        )


class Test_build_preamble_section:
    def test_expected_output(self, expected_text_output):
        """Test preamble section matches expected"""
        result = _build_preamble_section(6, "test_pac.csv")
        expected_lines = expected_text_output.split("\n")[:4]
        expected = "\n".join(expected_lines)
        assert result == expected


class Test_build_input_section:
    def test_expected_output(self, test_input_file):
        """Test input section matches expected"""
        result = _build_input_section(test_input_file)
        expected = header_creator("Raw Input") + "\n" + test_input_file.strip()
        assert result == expected


class Test_build_atomic_masses_section:
    @pytest.mark.parametrize(
        "f_atom_masses_df",
        [
            "hn3_dn3_atom_masses_df",
            "pyridazine_pheavy_atom_masses_df",
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(self, f_atom_masses_df, request):
        """Validate atomic masses section labels and values together."""
        atom_masses_df = request.getfixturevalue(f_atom_masses_df)
        result = _build_atomic_masses_section(atom_masses_df, 6)

        assert "Atomic Masses" in result
        assert "Total" in result

        numbers = _parse_float_values(result)
        expected = atom_masses_df.to_numpy().flatten()
        assert np.allclose(numbers, expected, rtol=1e-6, atol=1e-8)


class Test_build_com_values_section:
    @pytest.mark.parametrize(
        "f_com_values_df",
        [
            "hn3_dn3_com_values_df",
            "pyridazine_pheavy_com_values_df",
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(self, f_com_values_df, request):
        """Validate COM values section labels and values together."""
        com_values_df = request.getfixturevalue(f_com_values_df)
        result = _build_com_values_section(com_values_df, 6)

        assert "COM Values" in result
        assert "x" in result and "y" in result and "z" in result

        numbers = _parse_float_values(result)
        expected = com_values_df.to_numpy().flatten()
        assert np.allclose(numbers, expected, rtol=1e-6, atol=1e-8)


class Test_build_rotational_constants_section:
    @pytest.mark.parametrize(
        "f_rotational_constants_df",
        [
            "hn3_dn3_rotational_constants_df",
            "pyridazine_pheavy_rotational_constants_df",
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(self, f_rotational_constants_df, request):
        """Validate rotational constants labels and values together."""
        rotational_constants_df = request.getfixturevalue(f_rotational_constants_df)
        result = _build_rotational_constants_section(rotational_constants_df, 6)

        assert "Rotational Constants" in result
        assert "A" in result and "B" in result and "C" in result

        numbers = _parse_float_values(result)
        expected = rotational_constants_df.to_numpy().flatten()
        assert np.allclose(numbers, expected, rtol=1e-6, atol=1e-8)


class Test_build_dipole_components_section:
    @pytest.mark.parametrize(
        "f_dipole_components_df",
        [
            "hn3_dn3_dipole_components_df",
            "pyridazine_pheavy_dipole_components_df",
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(self, f_dipole_components_df, request):
        """Validate dipole section labels and values together."""
        dipole_components_df = request.getfixturevalue(f_dipole_components_df)
        result = _build_dipole_components_section(dipole_components_df, 6)

        assert "Dipole Components" in result
        assert "mu_A" in result and "mu_B" in result and "mu_C" in result

        numbers = _parse_float_values(result)
        expected = dipole_components_df.to_numpy().flatten()
        assert np.allclose(numbers, expected, rtol=1e-6, atol=1e-8)


class Test_build_com_coordinates_section:
    @pytest.mark.parametrize(
        "f_isotopologue_names,f_com_coordinates_df_dict,f_atom_numbering",
        [
            (
                "hn3_dn3_isotopologue_names",
                "hn3_dn3_com_coordinates_df_dict",
                "hn3_atom_numbering",
            ),
            (
                "pyridazine_pheavy_isotopologue_names",
                "pyridazine_pheavy_com_coordinates_df_dict",
                "pyridazine_atom_numbering",
            ),
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(
        self, f_isotopologue_names, f_com_coordinates_df_dict, f_atom_numbering, request
    ):
        """Validate COM coordinates labels and per-iso values together."""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        com_coordinates_df_dict = request.getfixturevalue(f_com_coordinates_df_dict)
        atom_numbering = request.getfixturevalue(f_atom_numbering)
        result = _build_com_coordinates_section(
            isotopologue_names, com_coordinates_df_dict, atom_numbering, 6
        )

        assert "COM Coordinates" in result
        for iso in isotopologue_names:
            assert iso in result

        parsed_by_iso = _parse_iso_numeric_entries(result, isotopologue_names)
        for iso in isotopologue_names:
            expected = com_coordinates_df_dict[iso].to_numpy().flatten()
            assert np.allclose(parsed_by_iso[iso], expected, rtol=1e-5, atol=1e-6)


class Test_build_com_inertias_section:
    @pytest.mark.parametrize(
        "f_isotopologue_names,f_com_inertias_df_dict",
        [
            ("hn3_dn3_isotopologue_names", "hn3_dn3_com_inertias_df_dict"),
            (
                "pyridazine_pheavy_isotopologue_names",
                "pyridazine_pheavy_com_inertias_df_dict",
            ),
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(
        self, f_isotopologue_names, f_com_inertias_df_dict, request
    ):
        """Validate COM inertia labels and per-iso values together."""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        com_inertias_df_dict = request.getfixturevalue(f_com_inertias_df_dict)
        result = _build_com_inertias_section(
            isotopologue_names, com_inertias_df_dict, 6
        )

        assert "COM Inertia Matrix" in result
        for iso in isotopologue_names:
            assert iso in result

        parsed_by_iso = _parse_iso_numeric_entries(result, isotopologue_names)
        for iso in isotopologue_names:
            expected = com_inertias_df_dict[iso].to_numpy().flatten()
            assert np.allclose(parsed_by_iso[iso], expected, rtol=1e-6, atol=1e-8)


class Test_build_eigens_section:
    @pytest.mark.parametrize(
        "f_isotopologue_names,f_eigenvectors_df_dict,f_evals",
        [
            ("hn3_isotopologue_names", "hn3_eigenvectors_df_dict", "hn3_evals"),
            ("dn3_isotopologue_names", "dn3_eigenvectors_df_dict", "dn3_evals"),
        ],
    )
    def test_expected_output(
        self, f_isotopologue_names, f_eigenvectors_df_dict, f_evals, request
    ):
        """Test eigenvectors/eigenvalues section"""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        eigenvectors_df_dict = request.getfixturevalue(f_eigenvectors_df_dict)
        evals = request.getfixturevalue(f_evals)
        eigenvalues = {iso: evals for iso in isotopologue_names}
        result = _build_eigens_section(
            isotopologue_names, eigenvectors_df_dict, eigenvalues, 6
        )
        assert "Eigenvectors & Eigenvalues" in result
        for iso in isotopologue_names:
            assert iso in result
            assert "Eigenvectors" in result
            assert "Eigenvalues" in result

    @pytest.mark.parametrize(
        "f_isotopologue_names,f_eigenvectors_df_dict,f_evals_list",
        [
            (
                "hn3_dn3_isotopologue_names",
                "hn3_dn3_eigenvectors_df_dict",
                [("hn3", "hn3_evals"), ("dn3", "dn3_evals")],
            ),
            (
                "pyridazine_pheavy_isotopologue_names",
                "pyridazine_pheavy_eigenvectors_df_dict",
                [("pyridazine", "pyridazine_evals"), ("pheavy", "pheavy_evals")],
            ),
        ],
    )
    def test_numeric_values(
        self, f_isotopologue_names, f_eigenvectors_df_dict, f_evals_list, request
    ):
        """Keep split test but enforce explicit per-iso eigenvalue coupling checks."""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        eigenvectors_df_dict = request.getfixturevalue(f_eigenvectors_df_dict)
        # Build eigenvalues dict from fixture names
        eigenvalues = {}
        for iso_name, evals_fixture_name in f_evals_list:
            evals = request.getfixturevalue(evals_fixture_name)
            eigenvalues[iso_name] = evals
        result = _build_eigens_section(
            isotopologue_names, eigenvectors_df_dict, eigenvalues, 6
        )

        assert "Eigenvectors & Eigenvalues" in result
        for iso in isotopologue_names:
            assert iso in result
        assert "Eigenvectors" in result
        assert "Eigenvalues" in result

        parsed_eigenvalues = _parse_eigenvalues_per_iso(result, isotopologue_names)
        for iso in isotopologue_names:
            assert parsed_eigenvalues[iso].shape[0] == 3
            assert np.allclose(
                parsed_eigenvalues[iso],
                np.array(eigenvalues[iso], dtype=np.float64),
                rtol=1e-6,
                atol=1e-8,
            )


class Test_build_pa_inertias_section:
    @pytest.mark.parametrize(
        "f_isotopologue_names,f_pa_inertias_df_dict",
        [
            ("hn3_dn3_isotopologue_names", "hn3_dn3_pa_inertias_df_dict"),
            (
                "pyridazine_pheavy_isotopologue_names",
                "pyridazine_pheavy_pa_inertias_df_dict",
            ),
        ],
        ids=["hn3_dn3", "pyridazine_pheavy"],
    )
    def test_section_output(self, f_isotopologue_names, f_pa_inertias_df_dict, request):
        """Validate PA inertia labels and per-iso values together."""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        pa_inertias_df_dict = request.getfixturevalue(f_pa_inertias_df_dict)
        result = _build_pa_inertias_section(isotopologue_names, pa_inertias_df_dict, 6)

        assert "Principal Axes Inertia Matrix" in result
        assert "(All entries should be diagonal)" in result
        for iso in isotopologue_names:
            assert iso in result

        parsed_by_iso = _parse_iso_numeric_entries(result, isotopologue_names)
        for iso in isotopologue_names:
            expected = pa_inertias_df_dict[iso].to_numpy().flatten()
            assert np.allclose(parsed_by_iso[iso], expected, rtol=1e-6, atol=1e-8)


class Test_build_results_section:
    @pytest.mark.parametrize(
        "f_isotopologue_names,f_pa_coordinates_df_dict,f_dipole_components_df,f_rotational_constants_df,f_atom_numbering",
        [
            (
                "hn3_isotopologue_names",
                "hn3_pa_coordinates_df_dict",
                "hn3_dipole_components_df",
                "hn3_rotational_constants_df",
                "hn3_atom_numbering",
            ),
            (
                "dn3_isotopologue_names",
                "dn3_pa_coordinates_df_dict",
                "dn3_dipole_components_df",
                "dn3_rotational_constants_df",
                "dn3_atom_numbering",
            ),
        ],
    )
    def test_expected_output(
        self,
        f_isotopologue_names,
        f_pa_coordinates_df_dict,
        f_dipole_components_df,
        f_rotational_constants_df,
        f_atom_numbering,
        request,
    ):
        """Test results section"""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        pa_coordinates_df_dict = request.getfixturevalue(f_pa_coordinates_df_dict)
        dipole_components_df = request.getfixturevalue(f_dipole_components_df)
        rotational_constants_df = request.getfixturevalue(f_rotational_constants_df)
        atom_numbering = request.getfixturevalue(f_atom_numbering)
        result = _build_results_section(
            isotopologue_names,
            pa_coordinates_df_dict,
            dipole_components_df,
            rotational_constants_df,
            atom_numbering,
            6,
        )
        assert "Principal Axes Coordinates" in result
        assert (
            "(Includes dipole moments and rotational constants, for easy reference.)"
            in result
        )
        for iso in isotopologue_names:
            assert iso in result

    @pytest.mark.parametrize(
        "f_isotopologue_names,f_pa_coordinates_df_dict,f_dipole_components_df,f_rotational_constants_df,f_atom_numbering",
        [
            (
                "hn3_dn3_isotopologue_names",
                "hn3_dn3_pa_coordinates_df_dict",
                "hn3_dn3_dipole_components_df",
                "hn3_dn3_rotational_constants_df",
                "hn3_atom_numbering",
            ),
            (
                "pyridazine_pheavy_isotopologue_names",
                "pyridazine_pheavy_pa_coordinates_df_dict",
                "pyridazine_pheavy_dipole_components_df",
                "pyridazine_pheavy_rotational_constants_df",
                "pyridazine_atom_numbering",
            ),
        ],
    )
    def test_numeric_values(
        self,
        f_isotopologue_names,
        f_pa_coordinates_df_dict,
        f_dipole_components_df,
        f_rotational_constants_df,
        f_atom_numbering,
        request,
    ):
        """Keep split test but enforce per-iso coupling of coordinates/dipole/rot constants."""
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        pa_coordinates_df_dict = request.getfixturevalue(f_pa_coordinates_df_dict)
        dipole_components_df = request.getfixturevalue(f_dipole_components_df)
        rotational_constants_df = request.getfixturevalue(f_rotational_constants_df)
        atom_numbering = request.getfixturevalue(f_atom_numbering)
        result = _build_results_section(
            isotopologue_names,
            pa_coordinates_df_dict,
            dipole_components_df,
            rotational_constants_df,
            atom_numbering,
            6,
        )
        # For combined pairs, verify structural elements and numeric presence
        assert "Principal Axes Coordinates" in result
        assert (
            "(Includes dipole moments and rotational constants, for easy reference.)"
            in result
        )
        for iso in isotopologue_names:
            assert iso in result

        parsed_by_iso = _parse_iso_numeric_entries(result, isotopologue_names)
        for iso in isotopologue_names:
            expected_values = pa_coordinates_df_dict[iso].to_numpy().flatten()
            expected_values = np.concatenate(
                [
                    expected_values,
                    dipole_components_df[iso].to_numpy().flatten(),
                    rotational_constants_df[iso].to_numpy().flatten(),
                ]
            )
            assert np.allclose(
                parsed_by_iso[iso],
                expected_values,
                rtol=1e-5,
                atol=1e-6,
            )


class Test_generate_output_file:
    @pytest.mark.parametrize(
        "pair_name,iso_names_and_evals,expected_output_fixture",
        [
            (
                "hn3_dn3",
                [("hn3", "hn3_evals"), ("dn3", "dn3_evals")],
                "hn3_dn3_generate_output_expected",
            ),
            (
                "pyridazine_pheavy",
                [("pyridazine", "pyridazine_evals"), ("pheavy", "pheavy_evals")],
                "pyridazine_pheavy_generate_output_expected",
            ),
        ],
    )
    def test_full_output_matches_expected(
        self,
        pair_name,
        iso_names_and_evals,
        expected_output_fixture,
        test_input_file,
        request,
    ):
        """Test full generate_output_file produces expected output for both pairs"""
        # Get fixtures based on pair name
        atom_masses_df = request.getfixturevalue(f"{pair_name}_atom_masses_df")
        rotational_constants_df = request.getfixturevalue(
            f"{pair_name}_rotational_constants_df"
        )
        dipole_components_df = request.getfixturevalue(
            f"{pair_name}_dipole_components_df"
        )
        isotopologue_names = request.getfixturevalue(f"{pair_name}_isotopologue_names")
        com_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_com_coordinates_df_dict"
        )
        com_inertias_df_dict = request.getfixturevalue(
            f"{pair_name}_com_inertias_df_dict"
        )
        eigenvectors_df_dict = request.getfixturevalue(
            f"{pair_name}_eigenvectors_df_dict"
        )
        pa_inertias_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_inertias_df_dict"
        )
        pa_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_coordinates_df_dict"
        )
        com_values_df = request.getfixturevalue(f"{pair_name}_com_values_df")

        # Get atom symbols based on pair name
        if pair_name == "hn3_dn3":
            atom_symbols = ["H", "N", "N", "N"]
        else:  # pyridazine_pheavy
            atom_symbols = ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]

        # Build eigenvalues dict from fixture names
        eigenvalues = {}
        for iso_name, evals_fixture_name in iso_names_and_evals:
            evals = request.getfixturevalue(evals_fixture_name)
            eigenvalues[iso_name] = evals

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".out"
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            generate_output_file(
                num_of_decimals=6,
                csv_output_name="test_pac.csv",
                input_file=test_input_file,
                atom_masses_df=atom_masses_df,
                rotational_constants_df=rotational_constants_df,
                dipole_components_df=dipole_components_df,
                isotopologue_names=isotopologue_names,
                com_coordinates_df_dict=com_coordinates_df_dict,
                atom_symbols=atom_symbols,
                com_inertias_df_dict=com_inertias_df_dict,
                eigenvectors_df_dict=eigenvectors_df_dict,
                eigenvalues=eigenvalues,
                pa_inertias_df_dict=pa_inertias_df_dict,
                pa_coordinates_df_dict=pa_coordinates_df_dict,
                com_values_df=com_values_df,
                text_output_path=tmp_path,
            )

            with open(tmp_path, "r") as f:
                generated_output = f.read()

            expected_output = request.getfixturevalue(expected_output_fixture)

            assert generated_output == expected_output

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class Test_writer_section_golden_outputs:
    @pytest.mark.parametrize(
        "pair_name,iso_names_and_evals,sections_fixture",
        [
            (
                "hn3_dn3",
                [("hn3", "hn3_evals"), ("dn3", "dn3_evals")],
                "hn3_dn3_writer_sections_expected",
            ),
            (
                "pyridazine_pheavy",
                [("pyridazine", "pyridazine_evals"), ("pheavy", "pheavy_evals")],
                "pyridazine_pheavy_writer_sections_expected",
            ),
        ],
    )
    def test_all_build_sections_match_expected(
        self,
        pair_name,
        iso_names_and_evals,
        sections_fixture,
        test_input_file,
        request,
    ):
        atom_masses_df = request.getfixturevalue(f"{pair_name}_atom_masses_df")
        rotational_constants_df = request.getfixturevalue(
            f"{pair_name}_rotational_constants_df"
        )
        dipole_components_df = request.getfixturevalue(
            f"{pair_name}_dipole_components_df"
        )
        isotopologue_names = request.getfixturevalue(f"{pair_name}_isotopologue_names")
        com_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_com_coordinates_df_dict"
        )
        com_inertias_df_dict = request.getfixturevalue(
            f"{pair_name}_com_inertias_df_dict"
        )
        eigenvectors_df_dict = request.getfixturevalue(
            f"{pair_name}_eigenvectors_df_dict"
        )
        pa_inertias_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_inertias_df_dict"
        )
        pa_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_coordinates_df_dict"
        )
        com_values_df = request.getfixturevalue(f"{pair_name}_com_values_df")

        if pair_name == "hn3_dn3":
            atom_symbols = ["H", "N", "N", "N"]
        else:
            atom_symbols = ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]

        eigenvalues = {}
        for iso_name, evals_fixture_name in iso_names_and_evals:
            eigenvalues[iso_name] = request.getfixturevalue(evals_fixture_name)

        expected_sections = request.getfixturevalue(sections_fixture)

        assert (
            _build_preamble_section(6, "test_pac.csv") == expected_sections["preamble"]
        )
        assert _build_input_section(test_input_file) == expected_sections["input"]
        assert (
            _build_atomic_masses_section(atom_masses_df, 6)
            == expected_sections["atomic_masses"]
        )
        assert (
            _build_com_values_section(com_values_df, 6)
            == expected_sections["com_values"]
        )
        assert (
            _build_com_coordinates_section(
                isotopologue_names, com_coordinates_df_dict, atom_symbols, 6
            )
            == expected_sections["com_coordinates"]
        )
        assert (
            _build_com_inertias_section(isotopologue_names, com_inertias_df_dict, 6)
            == expected_sections["com_inertias"]
        )
        assert (
            _build_eigens_section(
                isotopologue_names, eigenvectors_df_dict, eigenvalues, 6
            )
            == expected_sections["eigens"]
        )
        assert (
            _build_pa_inertias_section(isotopologue_names, pa_inertias_df_dict, 6)
            == expected_sections["pa_inertias"]
        )
        assert (
            _build_rotational_constants_section(rotational_constants_df, 6)
            == expected_sections["rotational_constants"]
        )
        assert (
            _build_dipole_components_section(dipole_components_df, 6)
            == expected_sections["dipole_components"]
        )
        assert (
            _build_results_section(
                isotopologue_names,
                pa_coordinates_df_dict,
                dipole_components_df,
                rotational_constants_df,
                atom_symbols,
                6,
            )
            == expected_sections["results"]
        )


class Test_generate_csv_output:
    @pytest.mark.parametrize(
        "pair_name,expected_csv_fixture",
        [
            ("hn3_dn3", "hn3_dn3_generate_csv_expected"),
            ("pyridazine_pheavy", "pyridazine_pheavy_generate_csv_expected"),
        ],
    )
    def test_csv_output_matches_expected(
        self,
        pair_name,
        expected_csv_fixture,
        request,
    ):
        """Test generate_csv_output produces expected CSV for both pairs"""
        # Get fixtures based on pair name
        pa_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_coordinates_df_dict"
        )
        rotational_constants_df = request.getfixturevalue(
            f"{pair_name}_rotational_constants_df"
        )
        dipole_components_df = request.getfixturevalue(
            f"{pair_name}_dipole_components_df"
        )
        atom_masses_df = request.getfixturevalue(f"{pair_name}_atom_masses_df")

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".csv"
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            generate_csv_output(
                pa_coordinates_df_dict=pa_coordinates_df_dict,
                rotational_constants_df=rotational_constants_df,
                dipole_components_df=dipole_components_df,
                atom_masses_df=atom_masses_df,
                csv_output_path=tmp_path,
            )

            with open(tmp_path, "r") as f:
                generated_csv = f.read()

            expected_csv = request.getfixturevalue(expected_csv_fixture)

            assert generated_csv == expected_csv

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.parametrize(
        "pair_name",
        [
            "hn3_dn3",
            "pyridazine_pheavy",
        ],
    )
    def test_csv_structure_and_sections(
        self,
        pair_name,
        request,
    ):
        """Validate CSV structure contains all expected sections and labels"""
        # Get fixtures
        pa_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_coordinates_df_dict"
        )
        rotational_constants_df = request.getfixturevalue(
            f"{pair_name}_rotational_constants_df"
        )
        dipole_components_df = request.getfixturevalue(
            f"{pair_name}_dipole_components_df"
        )
        atom_masses_df = request.getfixturevalue(f"{pair_name}_atom_masses_df")

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".csv"
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            generate_csv_output(
                pa_coordinates_df_dict=pa_coordinates_df_dict,
                rotational_constants_df=rotational_constants_df,
                dipole_components_df=dipole_components_df,
                atom_masses_df=atom_masses_df,
                csv_output_path=tmp_path,
            )

            with open(tmp_path, "r") as f:
                generated_csv = f.read()

            # Check that all key sections are present
            assert "Rotational Constants" in generated_csv
            assert "Dipole Components" in generated_csv
            assert "Principal Axes Coordinates" in generated_csv
            assert "Atomic Masses" in generated_csv

            # Verify sections appear in expected order
            rot_idx = generated_csv.find("Rotational Constants")
            dip_idx = generated_csv.find("Dipole Components")
            coord_idx = generated_csv.find("Principal Axes Coordinates")
            mass_idx = generated_csv.find("Atomic Masses")

            assert rot_idx < dip_idx < coord_idx < mass_idx

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.parametrize(
        "pair_name",
        [
            "hn3_dn3",
            "pyridazine_pheavy",
        ],
    )
    def test_csv_numeric_values(
        self,
        pair_name,
        request,
    ):
        """Validate numeric values in CSV output match source data"""
        # Get fixtures
        pa_coordinates_df_dict = request.getfixturevalue(
            f"{pair_name}_pa_coordinates_df_dict"
        )
        rotational_constants_df = request.getfixturevalue(
            f"{pair_name}_rotational_constants_df"
        )
        dipole_components_df = request.getfixturevalue(
            f"{pair_name}_dipole_components_df"
        )
        atom_masses_df = request.getfixturevalue(f"{pair_name}_atom_masses_df")

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".csv"
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            generate_csv_output(
                pa_coordinates_df_dict=pa_coordinates_df_dict,
                rotational_constants_df=rotational_constants_df,
                dipole_components_df=dipole_components_df,
                atom_masses_df=atom_masses_df,
                csv_output_path=tmp_path,
            )

            with open(tmp_path, "r") as f:
                generated_csv = f.read()

            # Extract and validate rotational constants
            rot_lines = _parse_csv_section(generated_csv, "Rotational Constants")
            rot_values = _extract_csv_numeric_values(rot_lines)
            expected_rot_values = rotational_constants_df.to_numpy().flatten()
            assert np.allclose(rot_values, expected_rot_values, rtol=1e-6, atol=1e-8)

            # Extract and validate dipole components
            dip_lines = _parse_csv_section(generated_csv, "Dipole Components")
            dip_values = _extract_csv_numeric_values(dip_lines)
            expected_dip_values = dipole_components_df.to_numpy().flatten()
            assert np.allclose(dip_values, expected_dip_values, rtol=1e-6, atol=1e-8)

            # Extract and validate atomic masses
            mass_lines = _parse_csv_section(generated_csv, "Atomic Masses")
            mass_values = _extract_csv_numeric_values(mass_lines)
            expected_mass_values = atom_masses_df.to_numpy().flatten()
            assert np.allclose(mass_values, expected_mass_values, rtol=1e-6, atol=1e-8)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class Test_helper_functions:
    def test_header_creator(self):
        """Test header_creator function"""
        result = header_creator("Test")
        expected = "# ====== #\n#  Test  #\n# ====== #"
        assert result == expected

    def test_df_text_export(self):
        """Test df_text_export function"""
        import pandas as pd

        df = pd.DataFrame({"A": [1.123456, 2.654321], "B": [3.14159, 4.56789]})
        result = df_text_export(df, n_decimals=2)
        assert "1.12" in result
        assert "3.14" in result
