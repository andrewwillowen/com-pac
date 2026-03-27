#!/usr/bin/env python3
"""Generate static writer golden files for regression tests."""

import importlib.util
from pathlib import Path

from com_pac.writer import (
    generate_output_file,
)


def _load_conftest_module():
    conftest_path = Path("tests/com_pac/conftest.py")
    spec = importlib.util.spec_from_file_location("writer_test_conftest", conftest_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


fx = _load_conftest_module()

DECIMALS = 6
CSV_NAME = "test_pac.csv"
INPUT_PATH = Path("docs/example/com-pac/test.txt")


def build_hn3_dn3_data():
    atom_symbols = ["H", "N", "N", "N"]
    isotopologue_names = fx.hn3_dn3_isotopologue_names.__wrapped__(
        fx.hn3_isotopologue_names.__wrapped__(fx.hn3_iso_name.__wrapped__()),
        fx.dn3_isotopologue_names.__wrapped__(fx.dn3_iso_name.__wrapped__()),
    )

    data = {
        "atom_symbols": atom_symbols,
        "isotopologue_names": isotopologue_names,
        "atom_masses_df": fx.hn3_dn3_atom_masses_df.__wrapped__(
            fx.hn3_atom_masses_df.__wrapped__(
                fx.hn3_iso_name.__wrapped__(),
                fx.hn3_mol_masses.__wrapped__(),
                fx.hn3_symbols.__wrapped__(),
            ),
            fx.dn3_atom_masses_df.__wrapped__(
                fx.dn3_iso_name.__wrapped__(),
                fx.dn3_mol_masses.__wrapped__(),
                fx.dn3_symbols.__wrapped__(),
            ),
        ),
    }

    pa_axis_labels = fx.pa_axis_labels.__wrapped__()
    dipole_index_labels = fx.dipole_index_labels.__wrapped__()
    com_column_labels = fx.com_column_labels.__wrapped__()
    com_axis_labels = fx.com_axis_labels.__wrapped__()
    pa_column_labels = fx.pa_column_labels.__wrapped__()
    evec_column_labels = fx.evec_column_labels.__wrapped__()

    hn3_rot_df = fx.hn3_rotational_constants_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(), fx.hn3_rot_consts.__wrapped__(), pa_axis_labels
    )
    dn3_rot_df = fx.dn3_rotational_constants_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(), fx.dn3_rot_consts.__wrapped__(), pa_axis_labels
    )
    data["rotational_constants_df"] = fx.hn3_dn3_rotational_constants_df.__wrapped__(
        hn3_rot_df, dn3_rot_df
    )

    hn3_dip_df = fx.hn3_dipole_components_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(),
        fx.hn3_pa_dipole.__wrapped__(),
        dipole_index_labels,
    )
    dn3_dip_df = fx.dn3_dipole_components_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(),
        fx.dn3_pa_dipole.__wrapped__(),
        dipole_index_labels,
    )
    data["dipole_components_df"] = fx.hn3_dn3_dipole_components_df.__wrapped__(
        hn3_dip_df, dn3_dip_df
    )

    hn3_atom_numbering = fx.hn3_atom_numbering.__wrapped__(
        fx.hn3_symbols.__wrapped__(), fx.hn3_mass_numbers.__wrapped__()
    )
    dn3_atom_numbering = fx.dn3_atom_numbering.__wrapped__(hn3_atom_numbering)

    hn3_com_df = fx.hn3_com_coordinates_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(),
        fx.hn3_COM_coords.__wrapped__(),
        com_column_labels,
        hn3_atom_numbering,
    )
    dn3_com_df = fx.dn3_com_coordinates_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(),
        fx.dn3_COM_coords.__wrapped__(),
        com_column_labels,
        dn3_atom_numbering,
    )
    data["com_coordinates_df_dict"] = fx.hn3_dn3_com_coordinates_df_dict.__wrapped__(
        {fx.hn3_iso_name.__wrapped__(): hn3_com_df},
        {fx.dn3_iso_name.__wrapped__(): dn3_com_df},
    )

    hn3_ci_df = fx.hn3_com_inertias_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(),
        fx.hn3_COM_inertia.__wrapped__(),
        com_column_labels,
        com_axis_labels,
    )
    dn3_ci_df = fx.dn3_com_inertias_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(),
        fx.dn3_COM_inertia.__wrapped__(),
        com_column_labels,
        com_axis_labels,
    )
    data["com_inertias_df_dict"] = fx.hn3_dn3_com_inertias_df_dict.__wrapped__(
        {fx.hn3_iso_name.__wrapped__(): hn3_ci_df},
        {fx.dn3_iso_name.__wrapped__(): dn3_ci_df},
    )

    hn3_ev_df = fx.hn3_eigenvectors_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(),
        fx.hn3_evecs.__wrapped__(),
        com_axis_labels,
        evec_column_labels,
    )
    dn3_ev_df = fx.dn3_eigenvectors_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(),
        fx.dn3_evecs.__wrapped__(),
        com_axis_labels,
        evec_column_labels,
    )
    data["eigenvectors_df_dict"] = fx.hn3_dn3_eigenvectors_df_dict.__wrapped__(
        {fx.hn3_iso_name.__wrapped__(): hn3_ev_df},
        {fx.dn3_iso_name.__wrapped__(): dn3_ev_df},
    )
    data["eigenvalues"] = {
        fx.hn3_iso_name.__wrapped__(): fx.hn3_evals.__wrapped__(),
        fx.dn3_iso_name.__wrapped__(): fx.dn3_evals.__wrapped__(),
    }

    hn3_pi_df = fx.hn3_pa_inertias_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(),
        fx.hn3_pa_inertia.__wrapped__(),
        pa_axis_labels,
        pa_column_labels,
    )
    dn3_pi_df = fx.dn3_pa_inertias_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(),
        fx.dn3_pa_inertia.__wrapped__(),
        pa_axis_labels,
        pa_column_labels,
    )
    data["pa_inertias_df_dict"] = fx.hn3_dn3_pa_inertias_df_dict.__wrapped__(
        {fx.hn3_iso_name.__wrapped__(): hn3_pi_df},
        {fx.dn3_iso_name.__wrapped__(): dn3_pi_df},
    )

    hn3_pa_df = fx.hn3_pa_coordinates_df.__wrapped__(
        fx.hn3_iso_name.__wrapped__(),
        fx.hn3_pa_coords.__wrapped__(),
        pa_column_labels,
        hn3_atom_numbering,
    )
    dn3_pa_df = fx.dn3_pa_coordinates_df.__wrapped__(
        fx.dn3_iso_name.__wrapped__(),
        fx.dn3_pa_coords.__wrapped__(),
        pa_column_labels,
        dn3_atom_numbering,
    )
    data["pa_coordinates_df_dict"] = fx.hn3_dn3_pa_coordinates_df_dict.__wrapped__(
        {fx.hn3_iso_name.__wrapped__(): hn3_pa_df},
        {fx.dn3_iso_name.__wrapped__(): dn3_pa_df},
    )

    return data


def build_pyridazine_pheavy_data():
    atom_symbols = ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]
    isotopologue_names = fx.pyridazine_pheavy_isotopologue_names.__wrapped__(
        fx.pyridazine_isotopologue_names.__wrapped__(
            fx.pyridazine_iso_name.__wrapped__()
        ),
        fx.pheavy_isotopologue_names.__wrapped__(fx.pheavy_iso_name.__wrapped__()),
    )

    pa_axis_labels = fx.pa_axis_labels.__wrapped__()
    dipole_index_labels = fx.dipole_index_labels.__wrapped__()
    com_column_labels = fx.com_column_labels.__wrapped__()
    com_axis_labels = fx.com_axis_labels.__wrapped__()
    pa_column_labels = fx.pa_column_labels.__wrapped__()
    evec_column_labels = fx.evec_column_labels.__wrapped__()

    pyr_atom_num = fx.pyridazine_atom_numbering.__wrapped__(
        fx.pyridazine_symbols.__wrapped__(), fx.pyridazine_mass_numbers.__wrapped__()
    )
    pheavy_atom_num = fx.pheavy_atom_numbering.__wrapped__(pyr_atom_num)

    pyr_atom_masses_df = fx.pyridazine_atom_masses_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_mol_masses.__wrapped__(),
        fx.pyridazine_symbols.__wrapped__(),
    )
    pheavy_atom_masses_df = fx.pheavy_atom_masses_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_mol_masses.__wrapped__(),
        fx.pheavy_symbols.__wrapped__(),
    )

    pyr_rot_df = fx.pyridazine_rotational_constants_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_rot_consts.__wrapped__(),
        pa_axis_labels,
    )
    pheavy_rot_df = fx.pheavy_rotational_constants_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_rot_consts.__wrapped__(),
        pa_axis_labels,
    )

    pyr_dip_df = fx.pyridazine_dipole_components_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_pa_dipole.__wrapped__(),
        dipole_index_labels,
    )
    pheavy_dip_df = fx.pheavy_dipole_components_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_pa_dipole.__wrapped__(),
        dipole_index_labels,
    )

    pyr_com_df = fx.pyridazine_com_coordinates_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_COM_coords.__wrapped__(),
        com_column_labels,
        pyr_atom_num,
    )
    pheavy_com_df = fx.pheavy_com_coordinates_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_COM_coords.__wrapped__(),
        com_column_labels,
        pheavy_atom_num,
    )

    pyr_ci_df = fx.pyridazine_com_inertias_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_COM_inertia.__wrapped__(),
        com_column_labels,
        com_axis_labels,
    )
    pheavy_ci_df = fx.pheavy_com_inertias_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_COM_inertia.__wrapped__(),
        com_column_labels,
        com_axis_labels,
    )

    pyr_ev_df = fx.pyridazine_eigenvectors_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_evecs.__wrapped__(),
        com_axis_labels,
        evec_column_labels,
    )
    pheavy_ev_df = fx.pheavy_eigenvectors_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_evecs.__wrapped__(),
        com_axis_labels,
        evec_column_labels,
    )

    pyr_pi_df = fx.pyridazine_pa_inertias_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_pa_inertia.__wrapped__(),
        pa_axis_labels,
        pa_column_labels,
    )
    pheavy_pi_df = fx.pheavy_pa_inertias_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_pa_inertia.__wrapped__(),
        pa_axis_labels,
        pa_column_labels,
    )

    pyr_pa_df = fx.pyridazine_pa_coordinates_df.__wrapped__(
        fx.pyridazine_iso_name.__wrapped__(),
        fx.pyridazine_pa_coords.__wrapped__(),
        pa_column_labels,
        pyr_atom_num,
    )
    pheavy_pa_df = fx.pheavy_pa_coordinates_df.__wrapped__(
        fx.pheavy_iso_name.__wrapped__(),
        fx.pheavy_pa_coords.__wrapped__(),
        pa_column_labels,
        pheavy_atom_num,
    )

    return {
        "atom_symbols": atom_symbols,
        "isotopologue_names": isotopologue_names,
        "atom_masses_df": fx.pyridazine_pheavy_atom_masses_df.__wrapped__(
            pyr_atom_masses_df, pheavy_atom_masses_df
        ),
        "rotational_constants_df": fx.pyridazine_pheavy_rotational_constants_df.__wrapped__(
            pyr_rot_df, pheavy_rot_df
        ),
        "dipole_components_df": fx.pyridazine_pheavy_dipole_components_df.__wrapped__(
            pyr_dip_df, pheavy_dip_df
        ),
        "com_coordinates_df_dict": fx.pyridazine_pheavy_com_coordinates_df_dict.__wrapped__(
            {fx.pyridazine_iso_name.__wrapped__(): pyr_com_df},
            {fx.pheavy_iso_name.__wrapped__(): pheavy_com_df},
        ),
        "com_inertias_df_dict": fx.pyridazine_pheavy_com_inertias_df_dict.__wrapped__(
            {fx.pyridazine_iso_name.__wrapped__(): pyr_ci_df},
            {fx.pheavy_iso_name.__wrapped__(): pheavy_ci_df},
        ),
        "eigenvectors_df_dict": fx.pyridazine_pheavy_eigenvectors_df_dict.__wrapped__(
            {fx.pyridazine_iso_name.__wrapped__(): pyr_ev_df},
            {fx.pheavy_iso_name.__wrapped__(): pheavy_ev_df},
        ),
        "eigenvalues": {
            fx.pyridazine_iso_name.__wrapped__(): fx.pyridazine_evals.__wrapped__(),
            fx.pheavy_iso_name.__wrapped__(): fx.pheavy_evals.__wrapped__(),
        },
        "pa_inertias_df_dict": fx.pyridazine_pheavy_pa_inertias_df_dict.__wrapped__(
            {fx.pyridazine_iso_name.__wrapped__(): pyr_pi_df},
            {fx.pheavy_iso_name.__wrapped__(): pheavy_pi_df},
        ),
        "pa_coordinates_df_dict": fx.pyridazine_pheavy_pa_coordinates_df_dict.__wrapped__(
            {fx.pyridazine_iso_name.__wrapped__(): pyr_pa_df},
            {fx.pheavy_iso_name.__wrapped__(): pheavy_pa_df},
        ),
    }


def emit_pair(pair_name: str, data: dict) -> None:
    out_dir = Path("tests/golden/writer") / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = INPUT_PATH.read_text(encoding="utf-8")

    # Keep full_output.out as the only source-of-truth golden artifact.
    for section_name in [
        "preamble.txt",
        "input.txt",
        "atomic_masses.txt",
        "com_coordinates.txt",
        "com_inertias.txt",
        "eigens.txt",
        "pa_inertias.txt",
        "rotational_constants.txt",
        "dipole_components.txt",
        "results.txt",
    ]:
        section_path = out_dir / section_name
        if section_path.exists():
            section_path.unlink()

    output_path = out_dir / "full_output.out"
    generate_output_file(
        num_of_decimals=DECIMALS,
        csv_output_name=CSV_NAME,
        input_file=input_file,
        atom_masses_df=data["atom_masses_df"],
        rotational_constants_df=data["rotational_constants_df"],
        dipole_components_df=data["dipole_components_df"],
        isotopologue_names=data["isotopologue_names"],
        com_coordinates_df_dict=data["com_coordinates_df_dict"],
        atom_symbols=data["atom_symbols"],
        com_inertias_df_dict=data["com_inertias_df_dict"],
        eigenvectors_df_dict=data["eigenvectors_df_dict"],
        eigenvalues=data["eigenvalues"],
        pa_inertias_df_dict=data["pa_inertias_df_dict"],
        pa_coordinates_df_dict=data["pa_coordinates_df_dict"],
        text_output_path=str(output_path),
    )


def main() -> None:
    emit_pair("hn3_dn3", build_hn3_dn3_data())
    emit_pair("pyridazine_pheavy", build_pyridazine_pheavy_data())


if __name__ == "__main__":
    main()
