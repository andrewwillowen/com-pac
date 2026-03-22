#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #

import pandas as pd


def get_atom_masses_df(atom_masses, atom_symbols):
    atom_masses_df = pd.DataFrame.from_dict(atom_masses)
    atom_masses_df["Atom"] = atom_symbols
    atom_masses_df = atom_masses_df.set_index("Atom")
    atom_masses_df.loc["Total"] = [
        atom_masses_df[i].sum() for i in atom_masses_df.columns
    ]

    return atom_masses_df


def get_rotational_constants_df(rotational_constants):
    rotational_constants_df = pd.DataFrame.from_dict(rotational_constants)
    rotational_constants_df["Axis"] = ["A", "B", "C"]
    rotational_constants_df = rotational_constants_df.set_index("Axis")

    return rotational_constants_df


def get_dipole_components_df(dipoles):
    dipole_components_df = pd.DataFrame.from_dict(dipoles)
    dipole_components_df.index = ["mu_A", "mu_B", "mu_C"]
    return dipole_components_df


def get_atom_indexed_df(coordinates, atom_numbering, column_labels):
    coordinates_df = pd.DataFrame(data=coordinates, columns=column_labels)
    coordinates_df["Atom"] = atom_numbering
    coordinates_df = coordinates_df.set_index("Atom")
    return coordinates_df


def get_axis_indexed_df(data, column_labels, axis_labels):
    data_df = pd.DataFrame(data=data, columns=column_labels)
    data_df["Axis"] = axis_labels
    data_df = data_df.set_index("Axis")
    return data_df


def get_dataframes(
    atom_masses,
    atom_symbols,
    rotational_constants,
    pa_dipoles,
    isotopologue_names,
    com_coordinates,
    atom_numbering,
    com_inertias,
    eigenvectors,
    pa_inertias,
    pa_coordinates,
):
    # Atomic masses
    atom_masses_df = get_atom_masses_df(atom_masses, atom_symbols)

    # Rotational constants
    rotational_constants_df = get_rotational_constants_df(rotational_constants)

    # Dipole moments
    dipole_components_df = get_dipole_components_df(pa_dipoles)

    com_coordinates_df_dict = {}
    com_inertias_df_dict = {}
    eigenvectors_df_dict = {}
    pa_inertias_df_dict = {}
    pa_coordinates_df_dict = {}
    for iso in isotopologue_names:
        # Atom indexed
        com_coordinates_df_dict[iso] = get_atom_indexed_df(
            coordinates=com_coordinates[iso],
            atom_numbering=atom_numbering,
            column_labels=["x", "y", "z"],
        )

        pa_coordinates_df_dict[iso] = get_atom_indexed_df(
            coordinates=pa_coordinates[iso],
            atom_numbering=atom_numbering,
            column_labels=["a", "b", "c"],
        )

        # Axis indexed
        com_inertias_df_dict[iso] = get_axis_indexed_df(
            data=com_inertias[iso],
            column_labels=["x", "y", "z"],
            axis_labels=["x", "y", "z"],
        )

        eigenvectors_df_dict[iso] = get_axis_indexed_df(
            data=eigenvectors[iso],
            column_labels=["1", "2", "3"],
            axis_labels=["x", "y", "z"],
        )

        pa_inertias_df_dict[iso] = get_axis_indexed_df(
            data=pa_inertias[iso],
            column_labels=["a", "b", "c"],
            axis_labels=["a", "b", "c"],
        )

    return (
        atom_masses_df,
        rotational_constants_df,
        dipole_components_df,
        com_coordinates_df_dict,
        com_inertias_df_dict,
        eigenvectors_df_dict,
        pa_inertias_df_dict,
        pa_coordinates_df_dict,
    )
