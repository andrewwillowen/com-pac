"""
Unit tests for functions in dataframes.py
"""

from com_pac.dataframes import (
    get_atom_masses_df,
    get_rotational_constants_df,
    get_dipole_components_df,
    get_atom_indexed_df,
    get_axis_indexed_df,
    get_dataframes,
)

import pytest
import numpy as np
import pandas as pd


def assert_equal_df_float(df1, df2, **kwargs):
    """
    Shorthand for pandas test.
    Useful kwargs include
        rtol: float
            relative tolerance
        atol: float
            relative tolerance
    """
    pd.testing.assert_frame_equal(df1, df2, check_exact=False, **kwargs)


class Test_get_atoms_masses_df:
    @pytest.mark.parametrize(
        "f_atom_masses,f_atom_symbols,f_atom_masses_df",
        [
            ("hn3_atom_masses", "hn3_symbols", "hn3_atom_masses_df"),
        ],
    )
    def test_expected_results_one_iso(
        self, f_atom_masses, f_atom_symbols, f_atom_masses_df, request
    ):
        atom_masses = request.getfixturevalue(f_atom_masses)
        atom_symbols = request.getfixturevalue(f_atom_symbols)
        atom_masses_df = request.getfixturevalue(f_atom_masses_df)

        result = get_atom_masses_df(atom_masses, atom_symbols)

        assert_equal_df_float(result, atom_masses_df)


class Test_get_rotational_constants_df:
    @pytest.mark.parametrize(
        "f_iso_name,f_rotational_constants,f_rotational_constants_df",
        [
            ("hn3_iso_name", "hn3_rot_consts", "hn3_rotational_constants_df"),
        ],
    )
    def test_expected_results_one_iso(
        self, f_iso_name, f_rotational_constants, f_rotational_constants_df, request
    ):
        iso_name = request.getfixturevalue(f_iso_name)
        rotational_constants = request.getfixturevalue(f_rotational_constants)
        rotational_constants_df = request.getfixturevalue(f_rotational_constants_df)

        rc_dict = {iso_name: rotational_constants}
        result = get_rotational_constants_df(rc_dict)

        assert_equal_df_float(result, rotational_constants_df)


class Test_get_dipole_components_df:
    @pytest.mark.parametrize(
        "f_iso_name,f_dipoles,f_dipole_components_df",
        [
            ("hn3_iso_name", "hn3_pa_dipole_list", "hn3_dipole_components_df"),
        ],
    )
    def test_expected_results_one_iso(
        self, f_iso_name, f_dipoles, f_dipole_components_df, request
    ):
        iso_name = request.getfixturevalue(f_iso_name)
        dipoles = request.getfixturevalue(f_dipoles)
        dipole_components_df = request.getfixturevalue(f_dipole_components_df)

        dip_dict = {iso_name: dipoles}
        result = get_dipole_components_df(dip_dict)

        assert_equal_df_float(result, dipole_components_df)


class Test_get_atom_indexed_df:
    @pytest.mark.parametrize(
        "f_coordinates,f_atom_numbering,f_column_labels,f_coordinates_df",
        [
            (
                "hn3_COM_coords",
                "hn3_atom_numbering",
                "com_column_labels",
                "hn3_com_coordinates_df",
            ),
            (
                "hn3_pa_coords",
                "hn3_atom_numbering",
                "pa_column_labels",
                "hn3_pa_coordinates_df",
            ),
        ],
    )
    def test_expected_results_one_iso(
        self,
        f_coordinates,
        f_atom_numbering,
        f_column_labels,
        f_coordinates_df,
        request,
    ):
        coordinates = request.getfixturevalue(f_coordinates)
        atom_numbering = request.getfixturevalue(f_atom_numbering)
        column_labels = request.getfixturevalue(f_column_labels)
        coordinates_df = request.getfixturevalue(f_coordinates_df)

        result = get_atom_indexed_df(coordinates, atom_numbering, column_labels)

        assert_equal_df_float(result, coordinates_df)


class Test_get_axis_indexed_df:
    @pytest.mark.parametrize(
        "f_data,f_column_labels,f_axis_labels,f_data_df",
        [
            (
                "hn3_COM_inertia",
                "com_column_labels",
                "com_axis_labels",
                "hn3_com_inertias_df",
            ),
            (
                "hn3_evecs",
                "evec_column_labels",
                "com_axis_labels",
                "hn3_eigenvectors_df",
            ),
            (
                "hn3_pa_inertia",
                "pa_column_labels",
                "pa_column_labels",
                "hn3_pa_inertias_df",
            ),
        ],
    )
    def test_expected_results_one_iso(
        self, f_data, f_column_labels, f_axis_labels, f_data_df, request
    ):
        data = request.getfixturevalue(f_data)
        column_labels = request.getfixturevalue(f_column_labels)
        axis_labels = request.getfixturevalue(f_axis_labels)
        data_df = request.getfixturevalue(f_data_df)

        result = get_axis_indexed_df(data, column_labels, axis_labels)

        assert_equal_df_float(result, data_df)


class Test_get_dataframes:
    @pytest.mark.parametrize(
        "f_atom_masses,f_atom_symbols,f_rotational_constants,f_pa_dipoles,f_isotopologue_names,f_com_coordinates,f_atom_numbering,f_com_inertias,f_eigenvectors,f_pa_inertias,f_pa_coordinates,f_atom_masses_df,f_rotational_constants_df,f_dipole_components_df,f_com_coordinates_df_dict,f_com_inertias_df_dict,f_eigenvectors_df_dict,f_pa_inertias_df_dict,f_pa_coordinates_df_dict",
        [
            (
                "hn3_atom_masses",
                "hn3_symbols",
                "hn3_rot_consts_dict",  # should be a dict
                "hn3_pa_dipole_dict",  # should be a dict
                "hn3_isotopologue_names",
                "hn3_COM_coords_dict",  # should be a dict
                "hn3_atom_numbering",
                "hn3_COM_inertia_dict",  # should be a dict
                "hn3_evecs_dict",  # should be a dict
                "hn3_pa_inertia_dict",  # should be a dict
                "hn3_pa_coords_dict",  # should be a dict
                "hn3_atom_masses_df",
                "hn3_rotational_constants_df",
                "hn3_dipole_components_df",
                "hn3_com_coordinates_df_dict",
                "hn3_com_inertias_df_dict",
                "hn3_eigenvectors_df_dict",
                "hn3_pa_inertias_df_dict",
                "hn3_pa_coordinates_df_dict",
            ),
        ],
    )
    def test_expected_results_one_iso(
        self,
        f_atom_masses,
        f_atom_symbols,
        f_rotational_constants,
        f_pa_dipoles,
        f_isotopologue_names,
        f_com_coordinates,
        f_atom_numbering,
        f_com_inertias,
        f_eigenvectors,
        f_pa_inertias,
        f_pa_coordinates,
        f_atom_masses_df,
        f_rotational_constants_df,
        f_dipole_components_df,
        f_com_coordinates_df_dict,
        f_com_inertias_df_dict,
        f_eigenvectors_df_dict,
        f_pa_inertias_df_dict,
        f_pa_coordinates_df_dict,
        request,
    ):
        atom_masses = request.getfixturevalue(f_atom_masses)
        atom_symbols = request.getfixturevalue(f_atom_symbols)
        rotational_constants = request.getfixturevalue(f_rotational_constants)
        pa_dipoles = request.getfixturevalue(f_pa_dipoles)
        isotopologue_names = request.getfixturevalue(f_isotopologue_names)
        com_coordinates = request.getfixturevalue(f_com_coordinates)
        atom_numbering = request.getfixturevalue(f_atom_numbering)
        com_inertias = request.getfixturevalue(f_com_inertias)
        eigenvectors = request.getfixturevalue(f_eigenvectors)
        pa_inertias = request.getfixturevalue(f_pa_inertias)
        pa_coordinates = request.getfixturevalue(f_pa_coordinates)

        atom_masses_df = request.getfixturevalue(f_atom_masses_df)
        rotational_constants_df = request.getfixturevalue(f_rotational_constants_df)
        dipole_components_df = request.getfixturevalue(f_dipole_components_df)
        com_coordinates_df_dict = request.getfixturevalue(f_com_coordinates_df_dict)
        com_inertias_df_dict = request.getfixturevalue(f_com_inertias_df_dict)
        eigenvectors_df_dict = request.getfixturevalue(f_eigenvectors_df_dict)
        pa_inertias_df_dict = request.getfixturevalue(f_pa_inertias_df_dict)
        pa_coordinates_df_dict = request.getfixturevalue(f_pa_coordinates_df_dict)

        (
            result_atom_masses_df,
            result_rotational_constants_df,
            result_dipole_components_df,
            result_com_coordinates_df_dict,
            result_com_inertias_df_dict,
            result_eigenvectors_df_dict,
            result_pa_inertias_df_dict,
            result_pa_coordinates_df_dict,
        ) = get_dataframes(
            atom_masses=atom_masses,
            atom_symbols=atom_symbols,
            rotational_constants=rotational_constants,
            pa_dipoles=pa_dipoles,
            isotopologue_names=isotopologue_names,
            com_coordinates=com_coordinates,
            atom_numbering=atom_numbering,
            com_inertias=com_inertias,
            eigenvectors=eigenvectors,
            pa_inertias=pa_inertias,
            pa_coordinates=pa_coordinates,
        )

        # compare result_atom_masses_df to atom_masses_df
        assert_equal_df_float(result_atom_masses_df, atom_masses_df)
        # compare result_rotational_constants_df to rotational_constants_df
        assert_equal_df_float(result_rotational_constants_df, rotational_constants_df)
        # compare result_dipole_components_df to dipole_components_df
        assert_equal_df_float(result_dipole_components_df, dipole_components_df)

        for iso in isotopologue_names:
            # compare result_com_coordinates_df_dict to com_coordinates_df_dict
            assert_equal_df_float(
                result_com_coordinates_df_dict[iso],
                com_coordinates_df_dict[iso],
            )
            # compare result_com_inertias_df_dict to com_inertias_df_dict
            assert_equal_df_float(
                result_com_inertias_df_dict[iso],
                com_inertias_df_dict[iso],
            )
            # compare result_eigenvectors_df_dict to eigenvectors_df_dict
            assert_equal_df_float(
                result_eigenvectors_df_dict[iso],
                eigenvectors_df_dict[iso],
            )
            # compare result_pa_inertias_df_dict to pa_inertias_df_dict
            assert_equal_df_float(
                result_pa_inertias_df_dict[iso],
                pa_inertias_df_dict[iso],
            )
            # compare result_pa_coordinates_df_dict to pa_coordinates_df_dict
            assert_equal_df_float(
                result_pa_coordinates_df_dict[iso],
                pa_coordinates_df_dict[iso],
            )
