#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #

import pandas as pd


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
    atom_masses_df = pd.DataFrame.from_dict(atom_masses)
    atom_masses_df["Atom"] = atom_symbols
    atom_masses_df = atom_masses_df.set_index("Atom")
    atom_masses_df.loc["Total"] = [
        atom_masses_df[i].sum() for i in atom_masses_df.columns
    ]

    # Rotational constants
    rotational_constants_df = pd.DataFrame.from_dict(rotational_constants)
    rotational_constants_df["Axis"] = ["A", "B", "C"]
    rotational_constants_df = rotational_constants_df.set_index("Axis")

    # Dipole moments
    dipole_components_df = pd.DataFrame.from_dict(pa_dipoles)
    dipole_components_df.index = ["mu_A", "mu_B", "mu_C"]

    com_coordinates_df_dict = {}
    com_inertias_df_dict = {}
    eigenvectors_df_dict = {}
    pa_inertias_df_dict = {}
    pa_coordinates_df_dict = {}
    for iso in isotopologue_names:
        com_coordinates_df = pd.DataFrame(
            columns=["x", "y", "z"], data=com_coordinates[iso]
        )
        com_coordinates_df["Atom"] = atom_numbering
        com_coordinates_df = com_coordinates_df.set_index("Atom")
        com_coordinates_df_dict[iso] = com_coordinates_df

        com_inertias_df = pd.DataFrame(columns=["x", "y", "z"], data=com_inertias[iso])
        com_inertias_df["Axis"] = ["x", "y", "z"]
        com_inertias_df = com_inertias_df.set_index("Axis")
        com_inertias_df_dict[iso] = com_inertias_df

        eigenvector_df = pd.DataFrame(columns=["1", "2", "3"], data=eigenvectors[iso])
        eigenvector_df["Axis"] = ["x", "y", "z"]
        eigenvector_df = eigenvector_df.set_index("Axis")
        eigenvectors_df_dict[iso] = eigenvector_df

        pa_inertias_df = pd.DataFrame(columns=["a", "b", "c"], data=pa_inertias[iso])
        pa_inertias_df["Axis"] = ["a", "b", "c"]
        pa_inertias_df = pa_inertias_df.set_index("Axis")
        pa_inertias_df_dict[iso] = pa_inertias_df

        pa_coordinates_df = pd.DataFrame(
            columns=["a", "b", "c"], data=pa_coordinates[iso]
        )
        pa_coordinates_df["Atom"] = atom_numbering
        pa_coordinates_df = pa_coordinates_df.set_index("Atom")
        pa_coordinates_df_dict[iso] = pa_coordinates_df

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
