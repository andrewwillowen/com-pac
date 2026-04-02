"""
Sharing fixtures across module tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import re

module_fixture = pytest.fixture(scope="module")


# HN3 fixtures
@module_fixture
def hn3_symbols():
    return ["H", "N", "N", "N"]


@module_fixture
def hn3_mass_numbers():
    return [1, 14, 14, 14]


@module_fixture
def hn3_atom_numbering(hn3_symbols, hn3_mass_numbers):
    return [f"{sym}{num}" for sym, num in zip(hn3_symbols, hn3_mass_numbers)]


@module_fixture
def hn3_n_atoms():
    return 4


@module_fixture
def hn3_inputs(hn3_symbols, hn3_mass_numbers, hn3_n_atoms):
    return hn3_symbols, hn3_mass_numbers, hn3_n_atoms


@module_fixture
def hn3_coords():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, -1.0]]
    )


@module_fixture
def hn3_mol_masses():
    return np.array([1.00782503, 14.003074, 14.003074, 14.003074])


@module_fixture
def hn3_COM_coords():
    return np.array(
        [
            [-1.95314299, -0.32552383, 0.32552383],
            [-0.95314299, -0.32552383, 0.32552383],
            [0.04685701, 0.67447617, 0.32552383],
            [1.04685701, -0.32552383, -0.67447617],
        ]
    )


@module_fixture
def hn3_COM_inertia():
    return np.array(
        [
            [18.88947938, -0.65614213, 14.65921614],
            [-0.65614213, 41.3877405, -4.55833431],
            [14.65921614, -4.55833431, 41.3877405],
        ]
    )


@module_fixture
def hn3_evecs():
    return np.array(
        [
            [0.89325355, -0.23100136, -0.38566367],
            [-0.04869846, -0.9025551, 0.42781159],
            [-0.44690777, -0.36336299, -0.81745996],
        ]
    )


@module_fixture
def hn3_evals():
    return np.array([11.59103153, 39.3846498, 50.68927905])


@module_fixture
def hn3_pa_coords():
    return np.array(
        [
            [-1.87427853, 0.62669856, 0.34789073],
            [-0.98102498, 0.39569721, -0.03777294],
            [-0.13646989, -0.73785925, 0.00437498],
            [1.25238989, 0.29705748, 0.00835968],
        ]
    )


@module_fixture
def hn3_pa_inertia():
    return np.array(
        [
            [1.15910315e01, 1.77635684e-15, -2.27595720e-15],
            [1.77635684e-15, 3.93846498e01, -2.56045185e-15],
            [-2.27595720e-15, -2.56045185e-15, 5.06892790e01],
        ]
    )


@module_fixture
def hn3_COM_value():
    return np.array([1.95314299, 0.32552383, -0.32552383])


@module_fixture
def hn3_dipole():
    return np.array([1.5, 0.5, 0])


@module_fixture
def hn3_pa_dipole():
    return np.array([[1.31553109, 0.79777959, 0.36458971]])


@module_fixture
def hn3_pa_dipole_list(hn3_pa_dipole):
    return hn3_pa_dipole.flatten().tolist()


@module_fixture
def hn3_rot_consts():
    return [
        np.float64(43600.86531487504),
        np.float64(12831.877575816354),
        np.float64(9970.135974936498),
    ]


@module_fixture
def hn3_iso_name():
    return "hn3"


@module_fixture
def hn3_atom_masses(hn3_iso_name, hn3_mol_masses):
    return {hn3_iso_name: hn3_mol_masses}


@module_fixture
def hn3_atom_masses_df(hn3_iso_name, hn3_mol_masses, hn3_symbols):
    return pd.DataFrame(
        np.array([[mass] for mass in hn3_mol_masses] + [[np.sum(hn3_mol_masses)]]),
        index=pd.Index(hn3_symbols + ["Total"], dtype="object", name="Atom"),
        columns=pd.Index([hn3_iso_name], dtype="object"),
    )


@module_fixture
def pa_axis_labels():
    return ["A", "B", "C"]


@module_fixture
def hn3_rotational_constants_df(hn3_iso_name, hn3_rot_consts, pa_axis_labels):
    return pd.DataFrame(
        np.array(hn3_rot_consts).reshape(3, 1),
        index=pd.Index(pa_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index([hn3_iso_name], dtype="object"),
    )


@module_fixture
def dipole_index_labels():
    return ["mu_A", "mu_B", "mu_C"]


@module_fixture
def hn3_dipole_components_df(hn3_iso_name, hn3_pa_dipole, dipole_index_labels):
    return pd.DataFrame(
        np.array(hn3_pa_dipole).T,
        index=pd.Index(dipole_index_labels, dtype="object"),
        columns=pd.Index([hn3_iso_name], dtype="object"),
    )


@module_fixture
def com_column_labels():
    return ["x", "y", "z"]


@module_fixture
def com_axis_labels():
    return ["x", "y", "z"]


@module_fixture
def pa_column_labels():
    return ["a", "b", "c"]


@module_fixture
def hn3_com_coordinates_df(
    hn3_iso_name, hn3_COM_coords, com_column_labels, hn3_atom_numbering
):
    return pd.DataFrame(
        hn3_COM_coords,
        index=pd.Index(hn3_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def hn3_pa_coordinates_df(
    hn3_iso_name, hn3_pa_coords, pa_column_labels, hn3_atom_numbering
):
    return pd.DataFrame(
        hn3_pa_coords,
        index=pd.Index(hn3_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def hn3_com_inertias_df(
    hn3_iso_name, hn3_COM_inertia, com_column_labels, com_axis_labels
):
    return pd.DataFrame(
        hn3_COM_inertia,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def evec_column_labels():
    return ["1", "2", "3"]


@module_fixture
def hn3_eigenvectors_df(hn3_iso_name, hn3_evecs, com_axis_labels, evec_column_labels):
    return pd.DataFrame(
        hn3_evecs,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(evec_column_labels, dtype="object"),
    )


@module_fixture
def hn3_pa_inertias_df(hn3_iso_name, hn3_pa_inertia, pa_axis_labels, pa_column_labels):
    return pd.DataFrame(
        hn3_pa_inertia,
        index=pd.Index(pa_column_labels, dtype="object", name="Axis"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def hn3_COM_values_dict(hn3_iso_name, hn3_COM_value):
    return {hn3_iso_name: hn3_COM_value}


@module_fixture
def hn3_com_values_df(hn3_iso_name, hn3_COM_value):
    return pd.DataFrame(
        hn3_COM_value.reshape(3, 1),
        index=pd.Index(["x", "y", "z"], dtype="object"),
        columns=pd.Index([hn3_iso_name], dtype="object"),
    )


@module_fixture
def hn3_isotopologue_names(hn3_iso_name):
    return [hn3_iso_name]


@module_fixture
def hn3_com_coordinates_df_dict(hn3_iso_name, hn3_com_coordinates_df):
    return {hn3_iso_name: hn3_com_coordinates_df}


@module_fixture
def hn3_com_inertias_df_dict(hn3_iso_name, hn3_com_inertias_df):
    return {hn3_iso_name: hn3_com_inertias_df}


@module_fixture
def hn3_eigenvectors_df_dict(hn3_iso_name, hn3_eigenvectors_df):
    return {hn3_iso_name: hn3_eigenvectors_df}


@module_fixture
def hn3_pa_inertias_df_dict(hn3_iso_name, hn3_pa_inertias_df):
    return {hn3_iso_name: hn3_pa_inertias_df}


@module_fixture
def hn3_pa_coordinates_df_dict(hn3_iso_name, hn3_pa_coordinates_df):
    return {hn3_iso_name: hn3_pa_coordinates_df}


@module_fixture
def hn3_rot_consts_dict(hn3_iso_name, hn3_rot_consts):
    return {hn3_iso_name: hn3_rot_consts}


@module_fixture
def hn3_pa_dipole_dict(hn3_iso_name, hn3_pa_dipole_list):
    return {hn3_iso_name: hn3_pa_dipole_list}


@module_fixture
def hn3_COM_coords_dict(hn3_iso_name, hn3_COM_coords):
    return {hn3_iso_name: hn3_COM_coords}


@module_fixture
def hn3_COM_inertia_dict(hn3_iso_name, hn3_COM_inertia):
    return {hn3_iso_name: hn3_COM_inertia}


@module_fixture
def hn3_evecs_dict(hn3_iso_name, hn3_evecs):
    return {hn3_iso_name: hn3_evecs}


@module_fixture
def hn3_pa_inertia_dict(hn3_iso_name, hn3_pa_inertia):
    return {hn3_iso_name: hn3_pa_inertia}


@module_fixture
def hn3_pa_coords_dict(hn3_iso_name, hn3_pa_coords):
    return {hn3_iso_name: hn3_pa_coords}


# DN3 fixtures
@module_fixture
def dn3_symbols():
    return ["H", "N", "N", "N"]


@module_fixture
def dn3_mass_numbers():
    return [2, 14, 14, 14]


@module_fixture
def dn3_n_atoms():
    return 4


@module_fixture
def dn3_inputs(dn3_symbols, dn3_mass_numbers, dn3_n_atoms):
    return dn3_symbols, dn3_mass_numbers, dn3_n_atoms


@module_fixture
def dn3_coords():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, -1.0]]
    )


@module_fixture
def dn3_mol_masses():
    return np.array([2.01410178, 14.003074, 14.003074, 14.003074])


@module_fixture
def dn3_COM_coords():
    return np.array(
        [
            [-1.90849842, -0.31808307, 0.31808307],
            [-0.90849842, -0.31808307, 0.31808307],
            [0.09150158, 0.68191693, 0.31808307],
            [1.09150158, -0.31808307, -0.68191693],
        ]
    )


@module_fixture
def dn3_COM_inertia():
    return np.array(
        [
            [19.09786646, -1.28130336, 15.28437736],
            [-1.28130336, 45.24290137, -4.45414078],
            [15.28437736, -4.45414078, 45.24290137],
        ]
    )


@module_fixture
def dn3_evecs():
    return np.array(
        [
            [0.90705872, -0.20349833, -0.36855517],
            [-0.02140286, -0.89657661, 0.44237121],
            [-0.42045975, -0.39336853, -0.81760308],
        ]
    )


@module_fixture
def dn3_evals():
    return np.array([12.04315016, 42.99784848, 54.54267056])


@module_fixture
def dn3_pa_coords():
    return np.array(
        [
            [-1.85805337, 0.54843822, 0.30261046],
            [-0.95099465, 0.34493988, -0.0659447],
            [-0.06533879, -0.75513506, 0.00787134],
            [1.28358253, 0.33131175, 0.01454804],
        ]
    )


@module_fixture
def dn3_pa_inertia():
    return np.array(
        [
            [1.20431502e01, 3.55271368e-15, -5.55111512e-15],
            [3.55271368e-15, 4.29978485e01, -2.25236496e-14],
            [-5.55111512e-15, -2.25236496e-14, 5.45426706e01],
        ]
    )


@module_fixture
def dn3_COM_value():
    return np.array([1.90849842, 0.31808307, -0.31808307])


@module_fixture
def dn3_dipole():
    return np.array([1.5, 0.5, 0])


@module_fixture
def dn3_pa_dipole():
    return np.array([1.34988665, 0.7535358, 0.33164715])


@module_fixture
def dn3_rot_consts():
    return [
        np.float64(41964.020865451035),
        np.float64(11753.588201862513),
        np.float64(9265.754672647625),
    ]


@module_fixture
def dn3_iso_name():
    return "dn3"


@module_fixture
def dn3_atom_numbering(hn3_atom_numbering):
    return hn3_atom_numbering


@module_fixture
def dn3_atom_masses(dn3_iso_name, dn3_mol_masses):
    return {dn3_iso_name: dn3_mol_masses}


@module_fixture
def dn3_atom_masses_df(dn3_iso_name, dn3_mol_masses, dn3_symbols):
    return pd.DataFrame(
        np.array([[mass] for mass in dn3_mol_masses] + [[np.sum(dn3_mol_masses)]]),
        index=pd.Index(dn3_symbols + ["Total"], dtype="object", name="Atom"),
        columns=pd.Index([dn3_iso_name], dtype="object"),
    )


@module_fixture
def dn3_rotational_constants_df(dn3_iso_name, dn3_rot_consts, pa_axis_labels):
    return pd.DataFrame(
        np.array(dn3_rot_consts).reshape(3, 1),
        index=pd.Index(pa_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index([dn3_iso_name], dtype="object"),
    )


@module_fixture
def dn3_dipole_components_df(dn3_iso_name, dn3_pa_dipole, dipole_index_labels):
    return pd.DataFrame(
        np.array([dn3_pa_dipole]).T,
        index=pd.Index(dipole_index_labels, dtype="object"),
        columns=pd.Index([dn3_iso_name], dtype="object"),
    )


@module_fixture
def dn3_com_coordinates_df(
    dn3_iso_name, dn3_COM_coords, com_column_labels, dn3_atom_numbering
):
    return pd.DataFrame(
        dn3_COM_coords,
        index=pd.Index(dn3_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def dn3_pa_coordinates_df(
    dn3_iso_name, dn3_pa_coords, pa_column_labels, dn3_atom_numbering
):
    return pd.DataFrame(
        dn3_pa_coords,
        index=pd.Index(dn3_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def dn3_com_inertias_df(
    dn3_iso_name, dn3_COM_inertia, com_column_labels, com_axis_labels
):
    return pd.DataFrame(
        dn3_COM_inertia,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def dn3_eigenvectors_df(dn3_iso_name, dn3_evecs, com_axis_labels, evec_column_labels):
    return pd.DataFrame(
        dn3_evecs,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(evec_column_labels, dtype="object"),
    )


@module_fixture
def dn3_pa_inertias_df(dn3_iso_name, dn3_pa_inertia, pa_axis_labels, pa_column_labels):
    return pd.DataFrame(
        dn3_pa_inertia,
        index=pd.Index(pa_column_labels, dtype="object", name="Axis"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def dn3_COM_values_dict(dn3_iso_name, dn3_COM_value):
    return {dn3_iso_name: dn3_COM_value}


@module_fixture
def dn3_com_values_df(dn3_iso_name, dn3_COM_value):
    return pd.DataFrame(
        dn3_COM_value.reshape(3, 1),
        index=pd.Index(["x", "y", "z"], dtype="object"),
        columns=pd.Index([dn3_iso_name], dtype="object"),
    )


@module_fixture
def dn3_isotopologue_names(dn3_iso_name):
    return [dn3_iso_name]


@module_fixture
def dn3_com_coordinates_df_dict(dn3_iso_name, dn3_com_coordinates_df):
    return {dn3_iso_name: dn3_com_coordinates_df}


@module_fixture
def dn3_com_inertias_df_dict(dn3_iso_name, dn3_com_inertias_df):
    return {dn3_iso_name: dn3_com_inertias_df}


@module_fixture
def dn3_eigenvectors_df_dict(dn3_iso_name, dn3_eigenvectors_df):
    return {dn3_iso_name: dn3_eigenvectors_df}


@module_fixture
def dn3_pa_inertias_df_dict(dn3_iso_name, dn3_pa_inertias_df):
    return {dn3_iso_name: dn3_pa_inertias_df}


@module_fixture
def dn3_pa_coordinates_df_dict(dn3_iso_name, dn3_pa_coordinates_df):
    return {dn3_iso_name: dn3_pa_coordinates_df}


@module_fixture
def dn3_rot_consts_dict(dn3_iso_name, dn3_rot_consts):
    return {dn3_iso_name: dn3_rot_consts}


@module_fixture
def dn3_pa_dipole_list(dn3_pa_dipole):
    return dn3_pa_dipole.flatten().tolist()


@module_fixture
def dn3_pa_dipole_dict(dn3_iso_name, dn3_pa_dipole_list):
    return {dn3_iso_name: dn3_pa_dipole_list}


@module_fixture
def dn3_COM_coords_dict(dn3_iso_name, dn3_COM_coords):
    return {dn3_iso_name: dn3_COM_coords}


@module_fixture
def dn3_COM_inertia_dict(dn3_iso_name, dn3_COM_inertia):
    return {dn3_iso_name: dn3_COM_inertia}


@module_fixture
def dn3_evecs_dict(dn3_iso_name, dn3_evecs):
    return {dn3_iso_name: dn3_evecs}


@module_fixture
def dn3_pa_inertia_dict(dn3_iso_name, dn3_pa_inertia):
    return {dn3_iso_name: dn3_pa_inertia}


@module_fixture
def dn3_pa_coords_dict(dn3_iso_name, dn3_pa_coords):
    return {dn3_iso_name: dn3_pa_coords}


# Consolidated HN3 and DN3 fixtures
@module_fixture
def hn3_dn3_isotopologue_names(hn3_isotopologue_names, dn3_isotopologue_names):
    return hn3_isotopologue_names + dn3_isotopologue_names


@module_fixture
def hn3_dn3_atom_masses(hn3_atom_masses, dn3_atom_masses):
    return {**hn3_atom_masses, **dn3_atom_masses}


@module_fixture
def hn3_dn3_rot_consts_dict(hn3_rot_consts_dict, dn3_rot_consts_dict):
    return {**hn3_rot_consts_dict, **dn3_rot_consts_dict}


@module_fixture
def hn3_dn3_pa_dipole_dict(hn3_pa_dipole_dict, dn3_pa_dipole_dict):
    return {**hn3_pa_dipole_dict, **dn3_pa_dipole_dict}


@module_fixture
def hn3_dn3_COM_coords_dict(hn3_COM_coords_dict, dn3_COM_coords_dict):
    return {**hn3_COM_coords_dict, **dn3_COM_coords_dict}


@module_fixture
def hn3_dn3_COM_inertia_dict(hn3_COM_inertia_dict, dn3_COM_inertia_dict):
    return {**hn3_COM_inertia_dict, **dn3_COM_inertia_dict}


@module_fixture
def hn3_dn3_evecs_dict(hn3_evecs_dict, dn3_evecs_dict):
    return {**hn3_evecs_dict, **dn3_evecs_dict}


@module_fixture
def hn3_dn3_pa_inertia_dict(hn3_pa_inertia_dict, dn3_pa_inertia_dict):
    return {**hn3_pa_inertia_dict, **dn3_pa_inertia_dict}


@module_fixture
def hn3_dn3_pa_coords_dict(hn3_pa_coords_dict, dn3_pa_coords_dict):
    return {**hn3_pa_coords_dict, **dn3_pa_coords_dict}


@module_fixture
def hn3_dn3_atom_masses_df(hn3_atom_masses_df, dn3_atom_masses_df):
    return pd.concat([hn3_atom_masses_df, dn3_atom_masses_df], axis=1)


@module_fixture
def hn3_dn3_rotational_constants_df(
    hn3_rotational_constants_df, dn3_rotational_constants_df
):
    return pd.concat([hn3_rotational_constants_df, dn3_rotational_constants_df], axis=1)


@module_fixture
def hn3_dn3_dipole_components_df(hn3_dipole_components_df, dn3_dipole_components_df):
    return pd.concat([hn3_dipole_components_df, dn3_dipole_components_df], axis=1)


@module_fixture
def hn3_dn3_com_coordinates_df_dict(
    hn3_com_coordinates_df_dict, dn3_com_coordinates_df_dict
):
    return {**hn3_com_coordinates_df_dict, **dn3_com_coordinates_df_dict}


@module_fixture
def hn3_dn3_com_inertias_df_dict(hn3_com_inertias_df_dict, dn3_com_inertias_df_dict):
    return {**hn3_com_inertias_df_dict, **dn3_com_inertias_df_dict}


@module_fixture
def hn3_dn3_eigenvectors_df_dict(hn3_eigenvectors_df_dict, dn3_eigenvectors_df_dict):
    return {**hn3_eigenvectors_df_dict, **dn3_eigenvectors_df_dict}


@module_fixture
def hn3_dn3_pa_inertias_df_dict(hn3_pa_inertias_df_dict, dn3_pa_inertias_df_dict):
    return {**hn3_pa_inertias_df_dict, **dn3_pa_inertias_df_dict}


@module_fixture
def hn3_dn3_pa_coordinates_df_dict(
    hn3_pa_coordinates_df_dict, dn3_pa_coordinates_df_dict
):
    return {**hn3_pa_coordinates_df_dict, **dn3_pa_coordinates_df_dict}


@module_fixture
def hn3_dn3_COM_values_dict(hn3_COM_values_dict, dn3_COM_values_dict):
    return {**hn3_COM_values_dict, **dn3_COM_values_dict}


@module_fixture
def hn3_dn3_com_values_df(hn3_com_values_df, dn3_com_values_df):
    return pd.concat([hn3_com_values_df, dn3_com_values_df], axis=1)


# Pyridazine fixtures
@module_fixture
def pyridazine_symbols():
    return ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]


@module_fixture
def pyridazine_mass_numbers():
    return [14, 14, 12, 12, 12, 12, 1, 1, 1, 1]


@module_fixture
def pyridazine_n_atoms():
    return 10


@module_fixture
def pyridazine_inputs(pyridazine_symbols, pyridazine_mass_numbers, pyridazine_n_atoms):
    return pyridazine_symbols, pyridazine_mass_numbers, pyridazine_n_atoms


@module_fixture
def pyridazine_coords():
    return np.array(
        [
            [4.97560498, -1.52711508, -4.6439251],
            [-2.24354171, 1.46618232, 3.7042745],
            [-3.36868809, 3.44637206, 2.35270223],
            [-2.74559909, -1.83363707, 4.32503685],
            [3.97839912, 3.50814032, 1.35067896],
            [2.83399728, 4.29457292, -2.56178475],
            [-3.47524305, -1.79860119, -2.91935792],
            [-3.1576719, 1.45821783, 3.66951168],
            [2.75273824, -2.46957419, -2.42323765],
            [1.23559648, -4.10342235, 0.0342625],
        ]
    )


@module_fixture
def pyridazine_mol_masses():
    return np.array(
        [
            14.003074,
            14.003074,
            12.0,
            12.0,
            12.0,
            12.0,
            1.00782503,
            1.00782503,
            1.00782503,
            1.00782503,
        ]
    )


@module_fixture
def pyridazine_COM_coords():
    return np.array(
        [
            [4.42624563, -2.84105821, -5.27850252],
            [-2.79290106, 0.15223919, 3.06969708],
            [-3.91804744, 2.13242893, 1.71812481],
            [-3.29495844, -3.1475802, 3.69045943],
            [3.42903977, 2.19419719, 0.71610154],
            [2.28463793, 2.98062979, -3.19636217],
            [-4.0246024, -3.11254432, -3.55393534],
            [-3.70703125, 0.1442747, 3.03493426],
            [2.20337889, -3.78351732, -3.05781507],
            [0.68623713, -5.41736548, -0.60031492],
        ]
    )


@module_fixture
def pyridazine_COM_inertia():
    return np.array(
        [
            [1386.50311554, -14.08965667, 736.21259661],
            [-14.08965667, 1818.86541187, -52.17091434],
            [736.21259661, -52.17091434, 1442.32067778],
        ]
    )


@module_fixture
def pyridazine_evecs():
    return np.array(
        [
            [0.71967888, -0.11247809, -0.68513575],
            [-0.02282647, -0.99009027, 0.13856485],
            [-0.69393174, -0.08408296, -0.71511453],
        ]
    )


@module_fixture
def pyridazine_evals():
    return np.array([677.07605434, 1812.83418123, 2157.77896961])


@module_fixture
def pyridazine_pa_coords():
    return np.array(
        [
            [6.91324729, 2.75888055, 0.34848395],
            [-4.14362723, -0.09469959, -0.26057364],
            [-4.06067317, -1.81506765, 1.75121803],
            [-4.86039078, 3.17669439, -0.81775134],
            [1.92079612, -2.61835707, -2.55741373],
            [3.79422557, -2.93930465, 1.13348844],
            [-0.35918439, 3.83320484, 4.86758053],
            [-4.7772026, 0.01892856, 0.38948543],
            [3.79400455, 3.75530197, 0.15281184],
            [1.0341073, 5.33697044, -0.7915281],
        ]
    )


@module_fixture
def pyridazine_pa_inertia():
    return np.array(
        [
            [6.77076054e02, 1.34114941e-13, 2.88657986e-14],
            [1.34114941e-13, 1.81283418e03, 1.72306613e-13],
            [2.88657986e-14, 1.72306613e-13, 2.15777897e03],
        ]
    )


@module_fixture
def pyridazine_COM_value():
    return np.array([0.54935935, 1.31394313, 0.63457742])


@module_fixture
def pyridazine_dipole():
    return np.array([0.123, 1.943, 0.923])


@module_fixture
def pyridazine_pa_dipole():
    return np.array([0.59633032, 2.01518877, 0.4750909])


@module_fixture
def pyridazine_rot_consts():
    return [
        np.float64(746.4139388190787),
        np.float64(278.7783956374336),
        np.float64(234.21259161282072),
    ]


@module_fixture
def pyridazine_iso_name():
    return "pyridazine"


@module_fixture
def pyridazine_atom_numbering(pyridazine_symbols, pyridazine_mass_numbers):
    return [
        f"{sym}{num}" for sym, num in zip(pyridazine_symbols, pyridazine_mass_numbers)
    ]


@module_fixture
def pyridazine_atom_masses(pyridazine_iso_name, pyridazine_mol_masses):
    return {pyridazine_iso_name: pyridazine_mol_masses}


@module_fixture
def pyridazine_atom_masses_df(
    pyridazine_iso_name, pyridazine_mol_masses, pyridazine_symbols
):
    return pd.DataFrame(
        np.array(
            [[mass] for mass in pyridazine_mol_masses]
            + [[np.sum(pyridazine_mol_masses)]]
        ),
        index=pd.Index(pyridazine_symbols + ["Total"], dtype="object", name="Atom"),
        columns=pd.Index([pyridazine_iso_name], dtype="object"),
    )


@module_fixture
def pyridazine_rotational_constants_df(
    pyridazine_iso_name, pyridazine_rot_consts, pa_axis_labels
):
    return pd.DataFrame(
        np.array(pyridazine_rot_consts).reshape(3, 1),
        index=pd.Index(pa_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index([pyridazine_iso_name], dtype="object"),
    )


@module_fixture
def pyridazine_dipole_components_df(
    pyridazine_iso_name, pyridazine_pa_dipole, dipole_index_labels
):
    return pd.DataFrame(
        np.array([pyridazine_pa_dipole]).T,
        index=pd.Index(dipole_index_labels, dtype="object"),
        columns=pd.Index([pyridazine_iso_name], dtype="object"),
    )


@module_fixture
def pyridazine_com_coordinates_df(
    pyridazine_iso_name,
    pyridazine_COM_coords,
    com_column_labels,
    pyridazine_atom_numbering,
):
    return pd.DataFrame(
        pyridazine_COM_coords,
        index=pd.Index(pyridazine_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def pyridazine_pa_coordinates_df(
    pyridazine_iso_name,
    pyridazine_pa_coords,
    pa_column_labels,
    pyridazine_atom_numbering,
):
    return pd.DataFrame(
        pyridazine_pa_coords,
        index=pd.Index(pyridazine_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def pyridazine_com_inertias_df(
    pyridazine_iso_name, pyridazine_COM_inertia, com_column_labels, com_axis_labels
):
    return pd.DataFrame(
        pyridazine_COM_inertia,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def pyridazine_eigenvectors_df(
    pyridazine_iso_name, pyridazine_evecs, com_axis_labels, evec_column_labels
):
    return pd.DataFrame(
        pyridazine_evecs,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(evec_column_labels, dtype="object"),
    )


@module_fixture
def pyridazine_pa_inertias_df(
    pyridazine_iso_name, pyridazine_pa_inertia, pa_axis_labels, pa_column_labels
):
    return pd.DataFrame(
        pyridazine_pa_inertia,
        index=pd.Index(pa_column_labels, dtype="object", name="Axis"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def pyridazine_COM_values_dict(pyridazine_iso_name, pyridazine_COM_value):
    return {pyridazine_iso_name: pyridazine_COM_value}


@module_fixture
def pyridazine_com_values_df(pyridazine_iso_name, pyridazine_COM_value):
    return pd.DataFrame(
        pyridazine_COM_value.reshape(3, 1),
        index=pd.Index(["x", "y", "z"], dtype="object"),
        columns=pd.Index([pyridazine_iso_name], dtype="object"),
    )


@module_fixture
def pyridazine_isotopologue_names(pyridazine_iso_name):
    return [pyridazine_iso_name]


@module_fixture
def pyridazine_com_coordinates_df_dict(
    pyridazine_iso_name, pyridazine_com_coordinates_df
):
    return {pyridazine_iso_name: pyridazine_com_coordinates_df}


@module_fixture
def pyridazine_com_inertias_df_dict(pyridazine_iso_name, pyridazine_com_inertias_df):
    return {pyridazine_iso_name: pyridazine_com_inertias_df}


@module_fixture
def pyridazine_eigenvectors_df_dict(pyridazine_iso_name, pyridazine_eigenvectors_df):
    return {pyridazine_iso_name: pyridazine_eigenvectors_df}


@module_fixture
def pyridazine_pa_inertias_df_dict(pyridazine_iso_name, pyridazine_pa_inertias_df):
    return {pyridazine_iso_name: pyridazine_pa_inertias_df}


@module_fixture
def pyridazine_pa_coordinates_df_dict(
    pyridazine_iso_name, pyridazine_pa_coordinates_df
):
    return {pyridazine_iso_name: pyridazine_pa_coordinates_df}


@module_fixture
def pyridazine_rot_consts_dict(pyridazine_iso_name, pyridazine_rot_consts):
    return {pyridazine_iso_name: pyridazine_rot_consts}


@module_fixture
def pyridazine_pa_dipole_list(pyridazine_pa_dipole):
    return pyridazine_pa_dipole.flatten().tolist()


@module_fixture
def pyridazine_pa_dipole_dict(pyridazine_iso_name, pyridazine_pa_dipole_list):
    return {pyridazine_iso_name: pyridazine_pa_dipole_list}


@module_fixture
def pyridazine_COM_coords_dict(pyridazine_iso_name, pyridazine_COM_coords):
    return {pyridazine_iso_name: pyridazine_COM_coords}


@module_fixture
def pyridazine_COM_inertia_dict(pyridazine_iso_name, pyridazine_COM_inertia):
    return {pyridazine_iso_name: pyridazine_COM_inertia}


@module_fixture
def pyridazine_evecs_dict(pyridazine_iso_name, pyridazine_evecs):
    return {pyridazine_iso_name: pyridazine_evecs}


@module_fixture
def pyridazine_pa_inertia_dict(pyridazine_iso_name, pyridazine_pa_inertia):
    return {pyridazine_iso_name: pyridazine_pa_inertia}


@module_fixture
def pyridazine_pa_coords_dict(pyridazine_iso_name, pyridazine_pa_coords):
    return {pyridazine_iso_name: pyridazine_pa_coords}


# Pyridazine "heavy" (4-D, 4-C13) fixtures
@module_fixture
def pheavy_symbols():
    return ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]


@module_fixture
def pheavy_mass_numbers():
    return [14, 14, 12, 13, 12, 12, 1, 2, 1, 1]


@module_fixture
def pheavy_n_atoms():
    return 10


@module_fixture
def pheavy_inputs(pheavy_symbols, pheavy_mass_numbers, pheavy_n_atoms):
    return (
        pheavy_symbols,
        pheavy_mass_numbers,
        pheavy_n_atoms,
    )


@module_fixture
def pheavy_coords():
    return np.array(
        [
            [4.97560498, -1.52711508, -4.6439251],
            [-2.24354171, 1.46618232, 3.7042745],
            [-3.36868809, 3.44637206, 2.35270223],
            [-2.74559909, -1.83363707, 4.32503685],
            [3.97839912, 3.50814032, 1.35067896],
            [2.83399728, 4.29457292, -2.56178475],
            [-3.47524305, -1.79860119, -2.91935792],
            [-3.1576719, 1.45821783, 3.66951168],
            [2.75273824, -2.46957419, -2.42323765],
            [1.23559648, -4.10342235, 0.0342625],
        ]
    )


@module_fixture
def pheavy_mol_masses():
    return np.array(
        [
            14.003074,
            14.003074,
            12.0,
            13.00335484,
            12.0,
            12.0,
            1.00782503,
            2.01410178,
            1.00782503,
            1.00782503,
        ]
    )


@module_fixture
def pheavy_COM_coords():
    return np.array(
        [
            [4.51200507, -2.80433589, -5.36085553],
            [-2.70714162, 0.18896151, 2.98734407],
            [-3.832288, 2.16915125, 1.6357718],
            [-3.209199, -3.11085788, 3.60810642],
            [3.51479921, 2.23091951, 0.63374853],
            [2.37039737, 3.01735211, -3.27871518],
            [-3.93884296, -3.075822, -3.63628835],
            [-3.62127181, 0.18099702, 2.95258125],
            [2.28913833, -3.746795, -3.14016808],
            [0.77199657, -5.38064316, -0.68266793],
        ]
    )


@module_fixture
def pheavy_COM_inertia():
    return np.array(
        [
            [1418.73129399, -23.69901859, 759.15505338],
            [-23.69901859, 1865.3608688, -41.20466634],
            [759.15505338, -41.20466634, 1476.28955893],
        ]
    )


@module_fixture
def pheavy_evecs():
    return np.array(
        [
            [0.72013525, -0.09817077, -0.6868535],
            [-0.00978195, -0.99127777, 0.13142566],
            [-0.69376476, -0.08792548, -0.71481436],
        ]
    )


@module_fixture
def pheavy_evals():
    return np.array([687.6975053, 1859.35902817, 2213.32518825])


@module_fixture
def pheavy_pa_coords():
    return np.array(
        [
            [6.9958584, 2.80828461, 0.36436837],
            [-4.02387055, -0.18421483, -0.25115237],
            [-3.91582503, -1.91783876, 1.74802937],
            [-4.78380413, 3.08152929, -0.78372329],
            [2.06963566, -2.61223411, -2.5739646],
            [3.95214814, -2.93545518, 1.11211448],
            [-0.2836834, 3.75539563, 4.90043725],
            [-4.65797279, -0.08352241, 0.40052337],
            [3.86367811, 3.7654889, 0.17990956],
            [1.08218609, 5.31794834, -0.74942229],
        ]
    )


@module_fixture
def pheavy_pa_inertia():
    return np.array(
        [
            [6.87697505e02, -8.10018719e-13, -1.16895382e-12],
            [-8.10018719e-13, 1.85935903e03, 2.87769808e-13],
            [-1.16895382e-12, 2.87769808e-13, 2.21332519e03],
        ]
    )


@module_fixture
def pheavy_COM_value():
    return np.array([0.46359991, 1.27722081, 0.71693043])


@module_fixture
def pheavy_dipole():
    return np.array([0.123, 1.943, 0.923])


@module_fixture
def pheavy_pa_dipole():
    return np.array([0.57077457, 2.01928293, 0.48889658])


@module_fixture
def pheavy_rot_consts():
    return [
        np.float64(734.8856157032798),
        np.float64(271.80280781888536),
        np.float64(228.33472789399997),
    ]


@module_fixture
def pheavy_iso_name():
    return "pheavy"


@module_fixture
def pheavy_atom_numbering(pyridazine_atom_numbering):
    return pyridazine_atom_numbering


@module_fixture
def pheavy_atom_masses(pheavy_iso_name, pheavy_mol_masses):
    return {pheavy_iso_name: pheavy_mol_masses}


@module_fixture
def pheavy_atom_masses_df(pheavy_iso_name, pheavy_mol_masses, pheavy_symbols):
    return pd.DataFrame(
        np.array(
            [[mass] for mass in pheavy_mol_masses] + [[np.sum(pheavy_mol_masses)]]
        ),
        index=pd.Index(pheavy_symbols + ["Total"], dtype="object", name="Atom"),
        columns=pd.Index([pheavy_iso_name], dtype="object"),
    )


@module_fixture
def pheavy_rotational_constants_df(pheavy_iso_name, pheavy_rot_consts, pa_axis_labels):
    return pd.DataFrame(
        np.array(pheavy_rot_consts).reshape(3, 1),
        index=pd.Index(pa_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index([pheavy_iso_name], dtype="object"),
    )


@module_fixture
def pheavy_dipole_components_df(pheavy_iso_name, pheavy_pa_dipole, dipole_index_labels):
    return pd.DataFrame(
        np.array([pheavy_pa_dipole]).T,
        index=pd.Index(dipole_index_labels, dtype="object"),
        columns=pd.Index([pheavy_iso_name], dtype="object"),
    )


@module_fixture
def pheavy_com_coordinates_df(
    pheavy_iso_name, pheavy_COM_coords, com_column_labels, pheavy_atom_numbering
):
    return pd.DataFrame(
        pheavy_COM_coords,
        index=pd.Index(pheavy_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def pheavy_pa_coordinates_df(
    pheavy_iso_name, pheavy_pa_coords, pa_column_labels, pheavy_atom_numbering
):
    return pd.DataFrame(
        pheavy_pa_coords,
        index=pd.Index(pheavy_atom_numbering, dtype="object", name="Atom"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def pheavy_com_inertias_df(
    pheavy_iso_name, pheavy_COM_inertia, com_column_labels, com_axis_labels
):
    return pd.DataFrame(
        pheavy_COM_inertia,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(com_column_labels, dtype="object"),
    )


@module_fixture
def pheavy_eigenvectors_df(
    pheavy_iso_name, pheavy_evecs, com_axis_labels, evec_column_labels
):
    return pd.DataFrame(
        pheavy_evecs,
        index=pd.Index(com_axis_labels, dtype="object", name="Axis"),
        columns=pd.Index(evec_column_labels, dtype="object"),
    )


@module_fixture
def pheavy_pa_inertias_df(
    pheavy_iso_name, pheavy_pa_inertia, pa_axis_labels, pa_column_labels
):
    return pd.DataFrame(
        pheavy_pa_inertia,
        index=pd.Index(pa_column_labels, dtype="object", name="Axis"),
        columns=pd.Index(pa_column_labels, dtype="object"),
    )


@module_fixture
def pheavy_COM_values_dict(pheavy_iso_name, pheavy_COM_value):
    return {pheavy_iso_name: pheavy_COM_value}


@module_fixture
def pheavy_com_values_df(pheavy_iso_name, pheavy_COM_value):
    return pd.DataFrame(
        pheavy_COM_value.reshape(3, 1),
        index=pd.Index(["x", "y", "z"], dtype="object"),
        columns=pd.Index([pheavy_iso_name], dtype="object"),
    )


@module_fixture
def pheavy_isotopologue_names(pheavy_iso_name):
    return [pheavy_iso_name]


@module_fixture
def pheavy_com_coordinates_df_dict(pheavy_iso_name, pheavy_com_coordinates_df):
    return {pheavy_iso_name: pheavy_com_coordinates_df}


@module_fixture
def pheavy_com_inertias_df_dict(pheavy_iso_name, pheavy_com_inertias_df):
    return {pheavy_iso_name: pheavy_com_inertias_df}


@module_fixture
def pheavy_eigenvectors_df_dict(pheavy_iso_name, pheavy_eigenvectors_df):
    return {pheavy_iso_name: pheavy_eigenvectors_df}


@module_fixture
def pheavy_pa_inertias_df_dict(pheavy_iso_name, pheavy_pa_inertias_df):
    return {pheavy_iso_name: pheavy_pa_inertias_df}


@module_fixture
def pheavy_pa_coordinates_df_dict(pheavy_iso_name, pheavy_pa_coordinates_df):
    return {pheavy_iso_name: pheavy_pa_coordinates_df}


@module_fixture
def pheavy_rot_consts_dict(pheavy_iso_name, pheavy_rot_consts):
    return {pheavy_iso_name: pheavy_rot_consts}


@module_fixture
def pheavy_pa_dipole_list(pheavy_pa_dipole):
    return pheavy_pa_dipole.flatten().tolist()


@module_fixture
def pheavy_pa_dipole_dict(pheavy_iso_name, pheavy_pa_dipole_list):
    return {pheavy_iso_name: pheavy_pa_dipole_list}


@module_fixture
def pheavy_COM_coords_dict(pheavy_iso_name, pheavy_COM_coords):
    return {pheavy_iso_name: pheavy_COM_coords}


@module_fixture
def pheavy_COM_inertia_dict(pheavy_iso_name, pheavy_COM_inertia):
    return {pheavy_iso_name: pheavy_COM_inertia}


@module_fixture
def pheavy_evecs_dict(pheavy_iso_name, pheavy_evecs):
    return {pheavy_iso_name: pheavy_evecs}


@module_fixture
def pheavy_pa_inertia_dict(pheavy_iso_name, pheavy_pa_inertia):
    return {pheavy_iso_name: pheavy_pa_inertia}


@module_fixture
def pheavy_pa_coords_dict(pheavy_iso_name, pheavy_pa_coords):
    return {pheavy_iso_name: pheavy_pa_coords}


# Consolidated Pyridazine and Pheavy fixtures
@module_fixture
def pyridazine_pheavy_isotopologue_names(
    pyridazine_isotopologue_names, pheavy_isotopologue_names
):
    return pyridazine_isotopologue_names + pheavy_isotopologue_names


@module_fixture
def pyridazine_pheavy_atom_masses(pyridazine_atom_masses, pheavy_atom_masses):
    return {**pyridazine_atom_masses, **pheavy_atom_masses}


@module_fixture
def pyridazine_pheavy_rot_consts_dict(
    pyridazine_rot_consts_dict, pheavy_rot_consts_dict
):
    return {**pyridazine_rot_consts_dict, **pheavy_rot_consts_dict}


@module_fixture
def pyridazine_pheavy_pa_dipole_dict(pyridazine_pa_dipole_dict, pheavy_pa_dipole_dict):
    return {**pyridazine_pa_dipole_dict, **pheavy_pa_dipole_dict}


@module_fixture
def pyridazine_pheavy_COM_coords_dict(
    pyridazine_COM_coords_dict, pheavy_COM_coords_dict
):
    return {**pyridazine_COM_coords_dict, **pheavy_COM_coords_dict}


@module_fixture
def pyridazine_pheavy_COM_inertia_dict(
    pyridazine_COM_inertia_dict, pheavy_COM_inertia_dict
):
    return {**pyridazine_COM_inertia_dict, **pheavy_COM_inertia_dict}


@module_fixture
def pyridazine_pheavy_evecs_dict(pyridazine_evecs_dict, pheavy_evecs_dict):
    return {**pyridazine_evecs_dict, **pheavy_evecs_dict}


@module_fixture
def pyridazine_pheavy_pa_inertia_dict(
    pyridazine_pa_inertia_dict, pheavy_pa_inertia_dict
):
    return {**pyridazine_pa_inertia_dict, **pheavy_pa_inertia_dict}


@module_fixture
def pyridazine_pheavy_pa_coords_dict(pyridazine_pa_coords_dict, pheavy_pa_coords_dict):
    return {**pyridazine_pa_coords_dict, **pheavy_pa_coords_dict}


@module_fixture
def pyridazine_pheavy_atom_masses_df(pyridazine_atom_masses_df, pheavy_atom_masses_df):
    return pd.concat([pyridazine_atom_masses_df, pheavy_atom_masses_df], axis=1)


@module_fixture
def pyridazine_pheavy_rotational_constants_df(
    pyridazine_rotational_constants_df, pheavy_rotational_constants_df
):
    return pd.concat(
        [pyridazine_rotational_constants_df, pheavy_rotational_constants_df], axis=1
    )


@module_fixture
def pyridazine_pheavy_dipole_components_df(
    pyridazine_dipole_components_df, pheavy_dipole_components_df
):
    return pd.concat(
        [pyridazine_dipole_components_df, pheavy_dipole_components_df], axis=1
    )


@module_fixture
def pyridazine_pheavy_com_coordinates_df_dict(
    pyridazine_com_coordinates_df_dict, pheavy_com_coordinates_df_dict
):
    return {**pyridazine_com_coordinates_df_dict, **pheavy_com_coordinates_df_dict}


@module_fixture
def pyridazine_pheavy_com_inertias_df_dict(
    pyridazine_com_inertias_df_dict, pheavy_com_inertias_df_dict
):
    return {**pyridazine_com_inertias_df_dict, **pheavy_com_inertias_df_dict}


@module_fixture
def pyridazine_pheavy_eigenvectors_df_dict(
    pyridazine_eigenvectors_df_dict, pheavy_eigenvectors_df_dict
):
    return {**pyridazine_eigenvectors_df_dict, **pheavy_eigenvectors_df_dict}


@module_fixture
def pyridazine_pheavy_pa_inertias_df_dict(
    pyridazine_pa_inertias_df_dict, pheavy_pa_inertias_df_dict
):
    return {**pyridazine_pa_inertias_df_dict, **pheavy_pa_inertias_df_dict}


@module_fixture
def pyridazine_pheavy_pa_coordinates_df_dict(
    pyridazine_pa_coordinates_df_dict, pheavy_pa_coordinates_df_dict
):
    return {**pyridazine_pa_coordinates_df_dict, **pheavy_pa_coordinates_df_dict}


@module_fixture
def pyridazine_pheavy_COM_values_dict(
    pyridazine_COM_values_dict, pheavy_COM_values_dict
):
    return {**pyridazine_COM_values_dict, **pheavy_COM_values_dict}


@module_fixture
def pyridazine_pheavy_com_values_df(pyridazine_com_values_df, pheavy_com_values_df):
    return pd.concat([pyridazine_com_values_df, pheavy_com_values_df], axis=1)


# Expected output fixtures for writer tests
@module_fixture
def expected_text_output():
    """Expected text output from generate_output_file"""
    with open("docs/example/com-pac/test_pac.out", "r") as f:
        return f.read()


@module_fixture
def expected_csv_output():
    """Expected CSV output from generate_csv_output"""
    with open("docs/example/com-pac/test_pac.csv", "r") as f:
        return f.read()


@module_fixture
def test_input_file():
    """Test input file content"""
    with open("docs/example/com-pac/test.txt", "r") as f:
        return f.read()


def _read_writer_golden_text(pair_name, filename):
    golden_root = Path(__file__).resolve().parent.parent / "golden" / "writer"
    return (golden_root / pair_name / filename).read_text(encoding="utf-8")


def _writer_sections_from_full_output(full_output):
    # Section fixtures are parsed from the canonical full output to keep one
    # golden source of truth per pair and avoid per-section fixture drift.
    header_pattern = re.compile(r"(?m)^# [=]+ #\n#  (?P<title>.+?)  #\n# [=]+ #\n")
    title_to_key = {
        "Raw Input": "input",
        "Atomic Masses": "atomic_masses",
        "COM Values": "com_values",
        "COM Coordinates": "com_coordinates",
        "COM Inertia Matrix": "com_inertias",
        "Eigenvectors & Eigenvalues": "eigens",
        "Principal Axes Inertia Matrix": "pa_inertias",
        "Rotational Constants": "rotational_constants",
        "Dipole Components": "dipole_components",
        "Principal Axes Coordinates": "results",
    }

    matches = list(header_pattern.finditer(full_output))
    if not matches:
        raise ValueError("No writer section headers found in golden output")

    parsed = {"preamble": full_output[: matches[0].start()].rstrip("\n")}
    for idx, match in enumerate(matches):
        section_title = match.group("title")
        section_key = title_to_key.get(section_title)
        if section_key is None:
            raise KeyError(
                f"Unexpected writer section title in golden output: {section_title}"
            )

        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_output)
        parsed[section_key] = full_output[start:end].rstrip("\n")

    return parsed


@module_fixture
def hn3_dn3_writer_sections_expected():
    return _writer_sections_from_full_output(
        _read_writer_golden_text("hn3_dn3", "full_output.out")
    )


@module_fixture
def pyridazine_pheavy_writer_sections_expected():
    return _writer_sections_from_full_output(
        _read_writer_golden_text("pyridazine_pheavy", "full_output.out")
    )


@module_fixture
def hn3_dn3_generate_output_expected():
    return _read_writer_golden_text("hn3_dn3", "full_output.out")


@module_fixture
def pyridazine_pheavy_generate_output_expected():
    return _read_writer_golden_text("pyridazine_pheavy", "full_output.out")


@module_fixture
def hn3_dn3_generate_csv_expected(
    hn3_dn3_pa_coordinates_df_dict,
    hn3_dn3_rotational_constants_df,
    hn3_dn3_dipole_components_df,
    hn3_dn3_atom_masses_df,
):
    """Generate expected CSV output for hn3_dn3 pair"""
    import tempfile
    from com_pac.writer import generate_csv_output

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv"
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        generate_csv_output(
            pa_coordinates_df_dict=hn3_dn3_pa_coordinates_df_dict,
            rotational_constants_df=hn3_dn3_rotational_constants_df,
            dipole_components_df=hn3_dn3_dipole_components_df,
            atom_masses_df=hn3_dn3_atom_masses_df,
            csv_output_path=tmp_path,
        )

        with open(tmp_path, "r") as f:
            return f.read()
    finally:
        import os

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@module_fixture
def pyridazine_pheavy_generate_csv_expected(
    pyridazine_pheavy_pa_coordinates_df_dict,
    pyridazine_pheavy_rotational_constants_df,
    pyridazine_pheavy_dipole_components_df,
    pyridazine_pheavy_atom_masses_df,
):
    """Generate expected CSV output for pyridazine_pheavy pair"""
    import tempfile
    from com_pac.writer import generate_csv_output

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv"
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        generate_csv_output(
            pa_coordinates_df_dict=pyridazine_pheavy_pa_coordinates_df_dict,
            rotational_constants_df=pyridazine_pheavy_rotational_constants_df,
            dipole_components_df=pyridazine_pheavy_dipole_components_df,
            atom_masses_df=pyridazine_pheavy_atom_masses_df,
            csv_output_path=tmp_path,
        )

        with open(tmp_path, "r") as f:
            return f.read()
    finally:
        import os

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
