"""
Unit tests for functions in dataframes.py
"""

from com_pac.dataframes import (
    get_atom_masses_df,
    get_rotational_constants_df,
    get_dipole_components_df,
    get_COM_values_df,
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
            ("dn3_atom_masses", "dn3_symbols", "dn3_atom_masses_df"),
            (
                "pyridazine_atom_masses",
                "pyridazine_symbols",
                "pyridazine_atom_masses_df",
            ),
            ("pheavy_atom_masses", "pheavy_symbols", "pheavy_atom_masses_df"),
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
            ("dn3_iso_name", "dn3_rot_consts", "dn3_rotational_constants_df"),
            (
                "pyridazine_iso_name",
                "pyridazine_rot_consts",
                "pyridazine_rotational_constants_df",
            ),
            ("pheavy_iso_name", "pheavy_rot_consts", "pheavy_rotational_constants_df"),
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
            ("dn3_iso_name", "dn3_pa_dipole_list", "dn3_dipole_components_df"),
            (
                "pyridazine_iso_name",
                "pyridazine_pa_dipole_list",
                "pyridazine_dipole_components_df",
            ),
            ("pheavy_iso_name", "pheavy_pa_dipole_list", "pheavy_dipole_components_df"),
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


class Test_get_COM_values_df:
    @pytest.mark.parametrize(
        "f_iso_name,f_com_value,f_com_values_df",
        [
            ("hn3_iso_name", "hn3_COM_value", "hn3_com_values_df"),
            ("dn3_iso_name", "dn3_COM_value", "dn3_com_values_df"),
            (
                "pyridazine_iso_name",
                "pyridazine_COM_value",
                "pyridazine_com_values_df",
            ),
            ("pheavy_iso_name", "pheavy_COM_value", "pheavy_com_values_df"),
        ],
    )
    def test_expected_results_one_iso(
        self, f_iso_name, f_com_value, f_com_values_df, request
    ):
        iso_name = request.getfixturevalue(f_iso_name)
        com_value = request.getfixturevalue(f_com_value)
        com_values_df = request.getfixturevalue(f_com_values_df)

        com_values_dict = {iso_name: com_value}
        result = get_COM_values_df(com_values_dict)

        assert_equal_df_float(result, com_values_df)


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
            (
                "dn3_COM_coords",
                "dn3_atom_numbering",
                "com_column_labels",
                "dn3_com_coordinates_df",
            ),
            (
                "dn3_pa_coords",
                "dn3_atom_numbering",
                "pa_column_labels",
                "dn3_pa_coordinates_df",
            ),
            (
                "pyridazine_COM_coords",
                "pyridazine_atom_numbering",
                "com_column_labels",
                "pyridazine_com_coordinates_df",
            ),
            (
                "pyridazine_pa_coords",
                "pyridazine_atom_numbering",
                "pa_column_labels",
                "pyridazine_pa_coordinates_df",
            ),
            (
                "pheavy_COM_coords",
                "pheavy_atom_numbering",
                "com_column_labels",
                "pheavy_com_coordinates_df",
            ),
            (
                "pheavy_pa_coords",
                "pheavy_atom_numbering",
                "pa_column_labels",
                "pheavy_pa_coordinates_df",
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
            (
                "dn3_COM_inertia",
                "com_column_labels",
                "com_axis_labels",
                "dn3_com_inertias_df",
            ),
            (
                "dn3_evecs",
                "evec_column_labels",
                "com_axis_labels",
                "dn3_eigenvectors_df",
            ),
            (
                "dn3_pa_inertia",
                "pa_column_labels",
                "pa_column_labels",
                "dn3_pa_inertias_df",
            ),
            (
                "pyridazine_COM_inertia",
                "com_column_labels",
                "com_axis_labels",
                "pyridazine_com_inertias_df",
            ),
            (
                "pyridazine_evecs",
                "evec_column_labels",
                "com_axis_labels",
                "pyridazine_eigenvectors_df",
            ),
            (
                "pyridazine_pa_inertia",
                "pa_column_labels",
                "pa_column_labels",
                "pyridazine_pa_inertias_df",
            ),
            (
                "pheavy_COM_inertia",
                "com_column_labels",
                "com_axis_labels",
                "pheavy_com_inertias_df",
            ),
            (
                "pheavy_evecs",
                "evec_column_labels",
                "com_axis_labels",
                "pheavy_eigenvectors_df",
            ),
            (
                "pheavy_pa_inertia",
                "pa_column_labels",
                "pa_column_labels",
                "pheavy_pa_inertias_df",
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
        "f_atom_masses,f_atom_symbols,f_rotational_constants,f_pa_dipoles,f_isotopologue_names,f_com_coordinates,f_atom_numbering,f_com_inertias,f_eigenvectors,f_pa_inertias,f_pa_coordinates,f_COM_values,f_atom_masses_df,f_rotational_constants_df,f_dipole_components_df,f_com_coordinates_df_dict,f_com_inertias_df_dict,f_eigenvectors_df_dict,f_pa_inertias_df_dict,f_pa_coordinates_df_dict,f_com_values_df",
        [
            (
                "hn3_atom_masses",
                "hn3_symbols",
                "hn3_rot_consts_dict",
                "hn3_pa_dipole_dict",
                "hn3_isotopologue_names",
                "hn3_COM_coords_dict",
                "hn3_atom_numbering",
                "hn3_COM_inertia_dict",
                "hn3_evecs_dict",
                "hn3_pa_inertia_dict",
                "hn3_pa_coords_dict",
                "hn3_COM_values_dict",
                "hn3_atom_masses_df",
                "hn3_rotational_constants_df",
                "hn3_dipole_components_df",
                "hn3_com_coordinates_df_dict",
                "hn3_com_inertias_df_dict",
                "hn3_eigenvectors_df_dict",
                "hn3_pa_inertias_df_dict",
                "hn3_pa_coordinates_df_dict",
                "hn3_com_values_df",
            ),
            (
                "dn3_atom_masses",
                "dn3_symbols",
                "dn3_rot_consts_dict",
                "dn3_pa_dipole_dict",
                "dn3_isotopologue_names",
                "dn3_COM_coords_dict",
                "dn3_atom_numbering",
                "dn3_COM_inertia_dict",
                "dn3_evecs_dict",
                "dn3_pa_inertia_dict",
                "dn3_pa_coords_dict",
                "dn3_COM_values_dict",
                "dn3_atom_masses_df",
                "dn3_rotational_constants_df",
                "dn3_dipole_components_df",
                "dn3_com_coordinates_df_dict",
                "dn3_com_inertias_df_dict",
                "dn3_eigenvectors_df_dict",
                "dn3_pa_inertias_df_dict",
                "dn3_pa_coordinates_df_dict",
                "dn3_com_values_df",
            ),
            (
                "pyridazine_atom_masses",
                "pyridazine_symbols",
                "pyridazine_rot_consts_dict",
                "pyridazine_pa_dipole_dict",
                "pyridazine_isotopologue_names",
                "pyridazine_COM_coords_dict",
                "pyridazine_atom_numbering",
                "pyridazine_COM_inertia_dict",
                "pyridazine_evecs_dict",
                "pyridazine_pa_inertia_dict",
                "pyridazine_pa_coords_dict",
                "pyridazine_COM_values_dict",
                "pyridazine_atom_masses_df",
                "pyridazine_rotational_constants_df",
                "pyridazine_dipole_components_df",
                "pyridazine_com_coordinates_df_dict",
                "pyridazine_com_inertias_df_dict",
                "pyridazine_eigenvectors_df_dict",
                "pyridazine_pa_inertias_df_dict",
                "pyridazine_pa_coordinates_df_dict",
                "pyridazine_com_values_df",
            ),
            (
                "pheavy_atom_masses",
                "pheavy_symbols",
                "pheavy_rot_consts_dict",
                "pheavy_pa_dipole_dict",
                "pheavy_isotopologue_names",
                "pheavy_COM_coords_dict",
                "pheavy_atom_numbering",
                "pheavy_COM_inertia_dict",
                "pheavy_evecs_dict",
                "pheavy_pa_inertia_dict",
                "pheavy_pa_coords_dict",
                "pheavy_COM_values_dict",
                "pheavy_atom_masses_df",
                "pheavy_rotational_constants_df",
                "pheavy_dipole_components_df",
                "pheavy_com_coordinates_df_dict",
                "pheavy_com_inertias_df_dict",
                "pheavy_eigenvectors_df_dict",
                "pheavy_pa_inertias_df_dict",
                "pheavy_pa_coordinates_df_dict",
                "pheavy_com_values_df",
            ),
            (
                "hn3_dn3_atom_masses",
                "hn3_symbols",
                "hn3_dn3_rot_consts_dict",
                "hn3_dn3_pa_dipole_dict",
                "hn3_dn3_isotopologue_names",
                "hn3_dn3_COM_coords_dict",
                "hn3_atom_numbering",
                "hn3_dn3_COM_inertia_dict",
                "hn3_dn3_evecs_dict",
                "hn3_dn3_pa_inertia_dict",
                "hn3_dn3_pa_coords_dict",
                "hn3_dn3_COM_values_dict",
                "hn3_dn3_atom_masses_df",
                "hn3_dn3_rotational_constants_df",
                "hn3_dn3_dipole_components_df",
                "hn3_dn3_com_coordinates_df_dict",
                "hn3_dn3_com_inertias_df_dict",
                "hn3_dn3_eigenvectors_df_dict",
                "hn3_dn3_pa_inertias_df_dict",
                "hn3_dn3_pa_coordinates_df_dict",
                "hn3_dn3_com_values_df",
            ),
            (
                "pyridazine_pheavy_atom_masses",
                "pyridazine_symbols",
                "pyridazine_pheavy_rot_consts_dict",
                "pyridazine_pheavy_pa_dipole_dict",
                "pyridazine_pheavy_isotopologue_names",
                "pyridazine_pheavy_COM_coords_dict",
                "pyridazine_atom_numbering",
                "pyridazine_pheavy_COM_inertia_dict",
                "pyridazine_pheavy_evecs_dict",
                "pyridazine_pheavy_pa_inertia_dict",
                "pyridazine_pheavy_pa_coords_dict",
                "pyridazine_pheavy_COM_values_dict",
                "pyridazine_pheavy_atom_masses_df",
                "pyridazine_pheavy_rotational_constants_df",
                "pyridazine_pheavy_dipole_components_df",
                "pyridazine_pheavy_com_coordinates_df_dict",
                "pyridazine_pheavy_com_inertias_df_dict",
                "pyridazine_pheavy_eigenvectors_df_dict",
                "pyridazine_pheavy_pa_inertias_df_dict",
                "pyridazine_pheavy_pa_coordinates_df_dict",
                "pyridazine_pheavy_com_values_df",
            ),
        ],
    )
    def test_expected_results(
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
        f_COM_values,
        f_atom_masses_df,
        f_rotational_constants_df,
        f_dipole_components_df,
        f_com_coordinates_df_dict,
        f_com_inertias_df_dict,
        f_eigenvectors_df_dict,
        f_pa_inertias_df_dict,
        f_pa_coordinates_df_dict,
        f_com_values_df,
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
        COM_values = request.getfixturevalue(f_COM_values)

        atom_masses_df = request.getfixturevalue(f_atom_masses_df)
        rotational_constants_df = request.getfixturevalue(f_rotational_constants_df)
        dipole_components_df = request.getfixturevalue(f_dipole_components_df)
        com_coordinates_df_dict = request.getfixturevalue(f_com_coordinates_df_dict)
        com_inertias_df_dict = request.getfixturevalue(f_com_inertias_df_dict)
        eigenvectors_df_dict = request.getfixturevalue(f_eigenvectors_df_dict)
        pa_inertias_df_dict = request.getfixturevalue(f_pa_inertias_df_dict)
        pa_coordinates_df_dict = request.getfixturevalue(f_pa_coordinates_df_dict)
        com_values_df = request.getfixturevalue(f_com_values_df)

        (
            result_atom_masses_df,
            result_rotational_constants_df,
            result_dipole_components_df,
            result_com_coordinates_df_dict,
            result_com_inertias_df_dict,
            result_eigenvectors_df_dict,
            result_pa_inertias_df_dict,
            result_pa_coordinates_df_dict,
            result_com_values_df,
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
            COM_values=COM_values,
        )

        # compare result_atom_masses_df to atom_masses_df
        assert_equal_df_float(result_atom_masses_df, atom_masses_df)
        # compare result_rotational_constants_df to rotational_constants_df
        assert_equal_df_float(result_rotational_constants_df, rotational_constants_df)
        # compare result_dipole_components_df to dipole_components_df
        assert_equal_df_float(result_dipole_components_df, dipole_components_df)
        # compare result_com_values_df to com_values_df
        assert_equal_df_float(result_com_values_df, com_values_df)

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
