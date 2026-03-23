"""
Unit tests for functions in dataframes.py
"""

from com_pac.dataframes import get_atom_masses_df

import pytest
import numpy as np
import pandas as pd

################################

# Relevant fixtures from test_diagonalize:
# - atom_masses
# - atom_symbols
# - rotational_constants
# - dipoles
# - com_coordinates
# - pa_coordinates
# - com_inertias
# - pa_inertias
# - eigenvectors

# Still needed:
# - atom_numbering
# - the various outputs

################################


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


# fi_atom_masses
# fi_atom_symbols
# fo_atom_masses_df
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


# fi_rotational_constants
# fo_rotational_constants_df
class Test_get_rotational_constants_df:
    pass


# fi_dipoles
# fo_dipole_components_df
class Test_get_dipole_components_df:
    pass


# fi_coordinates
# fi_atom_numbering
# fi_column_labels
# fo_coordinates_df
class Test_get_atom_indexed_df:
    pass


# fi_data
# fi_column_labels
# fi_axis_labels
# fo_data_df
class Test_get_axis_indexed_df:
    pass


# fi_atom_masses
# fi_atom_symbols
# fi_rotational_constants
# fi_pa_dipoles
# fi_isotopologue_names
# fi_com_coordinates
# fi_atom_numbering
# fi_com_inertias
# fi_eigenvectors
# fi_pa_inertias
# fi_pa_coordinates
# fo_atom_masses_df
# fo_rotational_constants_df
# fo_dipole_components_df
# fo_com_coordinates_df_dict
# fo_com_inertias_df_dict
# fo_eigenvectors_df_dict
# fo_pa_inertias_df_dict
# fo_pa_coordinates_df_dict
class Test_get_dataframes:
    pass
