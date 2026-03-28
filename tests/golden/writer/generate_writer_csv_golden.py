#!/usr/bin/env python3
"""Generate static writer golden CSV files for regression tests."""

import importlib.util
from pathlib import Path
import tempfile

from com_pac.writer import generate_csv_output


def _load_conftest_module():
    conftest_path = Path("tests/com_pac/conftest.py").resolve()
    spec = importlib.util.spec_from_file_location("writer_test_conftest", conftest_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


fx = _load_conftest_module()


def _call_fixture(fixture_func):
    """Call a fixture function without parameters."""
    return fixture_func.__wrapped__()


def build_hn3_dn3_csv():
    """Generate CSV output for hn3_dn3 pair."""
    # Build base fixtures
    hn3_iso_name = _call_fixture(fx.hn3_iso_name)
    dn3_iso_name = _call_fixture(fx.dn3_iso_name)

    hn3_mol_masses = _call_fixture(fx.hn3_mol_masses)
    dn3_mol_masses = _call_fixture(fx.dn3_mol_masses)

    hn3_symbols = _call_fixture(fx.hn3_symbols)
    dn3_symbols = _call_fixture(fx.dn3_symbols)

    # Build atom masses
    hn3_atom_masses_df = fx.hn3_atom_masses_df.__wrapped__(
        hn3_iso_name, hn3_mol_masses, hn3_symbols
    )
    dn3_atom_masses_df = fx.dn3_atom_masses_df.__wrapped__(
        dn3_iso_name, dn3_mol_masses, dn3_symbols
    )
    atom_masses_df = fx.hn3_dn3_atom_masses_df.__wrapped__(
        hn3_atom_masses_df, dn3_atom_masses_df
    )

    # Build rotational constants
    pa_axis_labels = _call_fixture(fx.pa_axis_labels)
    hn3_rot_consts = _call_fixture(fx.hn3_rot_consts)
    dn3_rot_consts = _call_fixture(fx.dn3_rot_consts)

    hn3_rot_df = fx.hn3_rotational_constants_df.__wrapped__(
        hn3_iso_name, hn3_rot_consts, pa_axis_labels
    )
    dn3_rot_df = fx.dn3_rotational_constants_df.__wrapped__(
        dn3_iso_name, dn3_rot_consts, pa_axis_labels
    )
    rotational_constants_df = fx.hn3_dn3_rotational_constants_df.__wrapped__(
        hn3_rot_df, dn3_rot_df
    )

    # Build dipole components
    dipole_index_labels = _call_fixture(fx.dipole_index_labels)
    hn3_pa_dipole = _call_fixture(fx.hn3_pa_dipole)
    dn3_pa_dipole = _call_fixture(fx.dn3_pa_dipole)

    hn3_dip_df = fx.hn3_dipole_components_df.__wrapped__(
        hn3_iso_name, hn3_pa_dipole, dipole_index_labels
    )
    dn3_dip_df = fx.dn3_dipole_components_df.__wrapped__(
        dn3_iso_name, dn3_pa_dipole, dipole_index_labels
    )
    dipole_components_df = fx.hn3_dn3_dipole_components_df.__wrapped__(
        hn3_dip_df, dn3_dip_df
    )

    # Build PA coordinates
    pa_column_labels = _call_fixture(fx.pa_column_labels)
    com_axis_labels = _call_fixture(fx.com_axis_labels)

    hn3_pa_coords = _call_fixture(fx.hn3_pa_coords)
    dn3_pa_coords = _call_fixture(fx.dn3_pa_coords)

    hn3_mass_numbers = _call_fixture(fx.hn3_mass_numbers)
    dn3_mass_numbers = _call_fixture(fx.dn3_mass_numbers)

    hn3_atom_numbering = fx.hn3_atom_numbering.__wrapped__(
        hn3_symbols, hn3_mass_numbers
    )
    dn3_atom_numbering = fx.dn3_atom_numbering.__wrapped__(hn3_atom_numbering)

    hn3_pa_coords_df = fx.hn3_pa_coordinates_df.__wrapped__(
        hn3_iso_name,
        hn3_pa_coords,
        pa_column_labels,
        com_axis_labels,
        hn3_atom_numbering,
    )
    dn3_pa_coords_df = fx.dn3_pa_coordinates_df.__wrapped__(
        dn3_iso_name,
        dn3_pa_coords,
        pa_column_labels,
        com_axis_labels,
        dn3_atom_numbering,
    )
    pa_coordinates_df_dict = fx.hn3_dn3_pa_coordinates_df_dict.__wrapped__(
        hn3_pa_coords_df, dn3_pa_coords_df
    )

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        tmp_path = tmp.name

    generate_csv_output(
        pa_coordinates_df_dict=pa_coordinates_df_dict,
        rotational_constants_df=rotational_constants_df,
        dipole_components_df=dipole_components_df,
        atom_masses_df=atom_masses_df,
        csv_output_path=tmp_path,
    )

    with open(tmp_path, "r") as f:
        csv_content = f.read()

    Path(tmp_path).unlink()
    return csv_content


def build_pyridazine_pheavy_csv():
    """Generate CSV output for pyridazine_pheavy pair."""
    # Load necessary fixtures
    pyridazine_iso_name = _call_fixture(fx.pyridazine_iso_name)
    pheavy_iso_name = _call_fixture(fx.pheavy_iso_name)

    pyridazine_atom_masses = _call_fixture(fx.pyridazine_atom_masses)
    pheavy_atom_masses = _call_fixture(fx.pheavy_atom_masses)

    pyridazine_symbols = _call_fixture(fx.pyridazine_symbols)
    pheavy_symbols = _call_fixture(fx.pheavy_symbols)

    # Build atom masses
    pyridazine_atom_masses_df = fx.pyridazine_atom_masses_df.__wrapped__(
        pyridazine_iso_name, pyridazine_atom_masses, pyridazine_symbols
    )
    pheavy_atom_masses_df = fx.pheavy_atom_masses_df.__wrapped__(
        pheavy_iso_name, pheavy_atom_masses, pheavy_symbols
    )
    atom_masses_df = fx.pyridazine_pheavy_atom_masses_df.__wrapped__(
        pyridazine_atom_masses_df, pheavy_atom_masses_df
    )

    # Build rotational constants
    pa_axis_labels = _call_fixture(fx.pa_axis_labels)
    pyridazine_rot_consts = _call_fixture(fx.pyridazine_rot_consts)
    pheavy_rot_consts = _call_fixture(fx.pheavy_rot_consts)

    pyridazine_rot_df = fx.pyridazine_rotational_constants_df.__wrapped__(
        pyridazine_iso_name, pyridazine_rot_consts, pa_axis_labels
    )
    pheavy_rot_df = fx.pheavy_rotational_constants_df.__wrapped__(
        pheavy_iso_name, pheavy_rot_consts, pa_axis_labels
    )
    rotational_constants_df = fx.pyridazine_pheavy_rotational_constants_df.__wrapped__(
        pyridazine_rot_df, pheavy_rot_df
    )

    # Build dipole components
    dipole_index_labels = _call_fixture(fx.dipole_index_labels)
    pyridazine_pa_dipole = _call_fixture(fx.pyridazine_pa_dipole)
    pheavy_pa_dipole = _call_fixture(fx.pheavy_pa_dipole)

    pyridazine_dip_df = fx.pyridazine_dipole_components_df.__wrapped__(
        pyridazine_iso_name, pyridazine_pa_dipole, dipole_index_labels
    )
    pheavy_dip_df = fx.pheavy_dipole_components_df.__wrapped__(
        pheavy_iso_name, pheavy_pa_dipole, dipole_index_labels
    )
    dipole_components_df = fx.pyridazine_pheavy_dipole_components_df.__wrapped__(
        pyridazine_dip_df, pheavy_dip_df
    )

    # Build PA coordinates
    pa_column_labels = _call_fixture(fx.pa_column_labels)
    com_axis_labels = _call_fixture(fx.com_axis_labels)

    pyridazine_pa_coords = _call_fixture(fx.pyridazine_pa_coords)
    pheavy_pa_coords = _call_fixture(fx.pheavy_pa_coords)

    pyridazine_mass_numbers = _call_fixture(fx.pyridazine_mass_numbers)
    pheavy_mass_numbers = _call_fixture(fx.pheavy_mass_numbers)

    pyridazine_atom_numbering = fx.pyridazine_atom_numbering.__wrapped__(
        pyridazine_symbols, pyridazine_mass_numbers
    )
    pheavy_atom_numbering = fx.pheavy_atom_numbering.__wrapped__(
        pheavy_symbols, pheavy_mass_numbers
    )

    pyridazine_pa_coords_df = fx.pyridazine_pa_coordinates_df.__wrapped__(
        pyridazine_iso_name,
        pyridazine_pa_coords,
        pa_column_labels,
        com_axis_labels,
        pyridazine_atom_numbering,
    )
    pheavy_pa_coords_df = fx.pheavy_pa_coordinates_df.__wrapped__(
        pheavy_iso_name,
        pheavy_pa_coords,
        pa_column_labels,
        com_axis_labels,
        pheavy_atom_numbering,
    )
    pa_coordinates_df_dict = fx.pyridazine_pheavy_pa_coordinates_df_dict.__wrapped__(
        pyridazine_pa_coords_df, pheavy_pa_coords_df
    )

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        tmp_path = tmp.name

    generate_csv_output(
        pa_coordinates_df_dict=pa_coordinates_df_dict,
        rotational_constants_df=rotational_constants_df,
        dipole_components_df=dipole_components_df,
        atom_masses_df=atom_masses_df,
        csv_output_path=tmp_path,
    )

    with open(tmp_path, "r") as f:
        csv_content = f.read()

    Path(tmp_path).unlink()
    return csv_content


if __name__ == "__main__":
    # Generate and save golden CSV files
    hn3_dn3_csv = build_hn3_dn3_csv()
    hn3_dn3_output_path = Path(__file__).parent / "hn3_dn3" / "output.csv"
    hn3_dn3_output_path.write_text(hn3_dn3_csv)
    print(f"Generated: {hn3_dn3_output_path}")

    pyridazine_pheavy_csv = build_pyridazine_pheavy_csv()
    pyridazine_pheavy_output_path = (
        Path(__file__).parent / "pyridazine_pheavy" / "output.csv"
    )
    pyridazine_pheavy_output_path.write_text(pyridazine_pheavy_csv)
    print(f"Generated: {pyridazine_pheavy_output_path}")
