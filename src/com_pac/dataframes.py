#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #

import pandas as pd


def get_atom_masses_df(atom_masses, atom_symbols):
    """Convert atom masses dictionary to DataFrame

    Parameters
    ----------
    atom_masses: dict
        key = isotopologue_name: str
        value = mol_masses: np.array[float] of length n_atoms

    atom_symbols: list[str]
        List of atom symbols of length n_atoms

    Returns
    -------
    atom_masses_df: pd.DataFrame
        RowLabel = Atom
        ColumnLabel = IsotopologueName
        Values = ExactMasses
        FinalRow = Total
    """
    atom_masses_df = pd.DataFrame.from_dict(atom_masses)
    atom_masses_df["Atom"] = atom_symbols
    atom_masses_df = atom_masses_df.set_index("Atom")
    atom_masses_df.loc["Total"] = [
        atom_masses_df[i].sum() for i in atom_masses_df.columns
    ]

    return atom_masses_df


def get_rotational_constants_df(rotational_constants):
    """Convert rotational constants dictionary to DataFrame

    Parameters
    ----------
    rotational_constants: dict
        key = isotopologue_name: str
        value = constant_list: list[float] of length 3 in order of "A", "B", "C" axes

    Returns
    -------
    rotational_constants_df: pd.DataFrame
        RowLabel = Axis
        ColumnLabel = IsotopologueName
        Values = RotationalConstant

    """
    rotational_constants_df = pd.DataFrame.from_dict(rotational_constants)
    rotational_constants_df["Axis"] = ["A", "B", "C"]
    rotational_constants_df = rotational_constants_df.set_index("Axis")

    return rotational_constants_df


def get_COM_values_df(COM_values):
    """Convert COM values dictionary to DataFrame

    Parameters
    ----------
    COM_values: dict
        key = isotopologue_name: str
        value = com_value: np.array[float] of length 3 in order of "x", "y", "z" axes

    Returns
    -------
    com_values_df: pd.DataFrame
        RowLabel = Axis
        ColumnLabel = IsotopologueName
        Values = COMValue
    """
    com_values_df = pd.DataFrame.from_dict(COM_values)
    com_values_df.index = ["x", "y", "z"]
    return com_values_df


def get_dipole_components_df(dipoles):
    """Convert dipole dictionary to DataFrame

    Parameters
    ----------
    dipoles: dict
        key = isotopologue_name: str
        value = dipole_list: np.array[float] of length 3 in order of "A", "B", and "C" axes

    Returns
    -------
    dipole_components_df: pd.DataFrame
        RowLabel = Axis
        ColumnLabel = IsotopologueName
        Values = DipoleComponent
    """
    dipole_components_df = pd.DataFrame.from_dict(dipoles)
    dipole_components_df.index = ["mu_A", "mu_B", "mu_C"]
    return dipole_components_df


def get_atom_indexed_df(coordinates, atom_numbering, column_labels):
    """Convert an atomic coordinates array into a DataFrame

    Parameters
    ----------
    coordinates: np.array[float]
        n_atom rows for atoms, 3 columns for axes
    atom_numbering: list[str]
        List of length n_atoms containing labels of the form "<Symbol><AtomNumber>"
    column_labels: list[str]
        List of length 3 of the axes labels

    Returns
    -------
    coordinates_df: pd.DataFrame
        RowLabel = Atom
        ColumnLabel = CoordinateAxis
        Values = CoordinateValue
    """
    coordinates_df = pd.DataFrame(data=coordinates, columns=column_labels)
    coordinates_df["Atom"] = atom_numbering
    coordinates_df = coordinates_df.set_index("Atom")
    return coordinates_df


def get_axis_indexed_df(data, column_labels, axis_labels):
    """Convert an M x N array into a DataFrame with corresponding labels

    Parameters
    ----------
    data: np.array[float]
        Input array of dimension 2
    column_labels: list[str]
        List of length N of the labels for the columns
    axis_labels: list[str]
        List of length M of the axis labels for the rows

    Returns
    -------
    data_df: pd.DataFrame
        RowLabel = axis_labels
        ColumnLabel = column_labels
        Values = data
    """
    data_df = pd.DataFrame(data=data, columns=column_labels)
    data_df["Axis"] = axis_labels
    data_df = data_df.set_index("Axis")
    return data_df


def get_theta_df(isotopologue_names, theta_data):
    """Convert theta data dictionary to DataFrame.

    Parameters
    ----------
    isotopologue_names : list[str]
        List of isotopologue names.
    theta_data : dict
        key = isotopologue_name: str
        value = theta results (structure TBD)

    Returns
    -------
    theta_df : pd.DataFrame
        Row
    """
    theta_df = pd.DataFrame().from_dict(theta_data, orient="index")
    return theta_df


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
    COM_values,
    theta_data=None,
):
    """Composite function to obtain dataframes for computed data

    Parameters
    ----------
    atom_masses: dict
        key = isotopologue_name: str
        value = mol_masses: np.array[float] of length n_atoms
    atom_symbols: list[str]
        List of atom symbols of length n_atoms
    rotational_constants: dict
        key = isotopologue_name: str
        value = constant_list: list[float] of length 3 in order of "A", "B", "C" axes
    pa_dipoles: dict
        key = isotopologue_name: str
        value = dipole_list: np.array[float] of length 3 in order of "A", "B", and "C" axes
    isotopologue_names: list[str]
        List of isotopologue names
    com_coordinates: dict
        key = isotopologue_name: str
        value = np.array[float]
            Center-Of-Mass coordinates with n_atoms rows for atoms, 3 columns for axes
    atom_numbering: list[str]
        List of length n_atoms containing labels of the form "<Symbol><AtomNumber>"
    com_inertias: dict
        key = isotopologue_name: str
        value = np.array[float]
            Center-of-Mass inertias, 3 columns and 3 rows
    eigenvectors: dict
        key = isotopologue_name: str
        value = np.array[float]
            3x3 matrix that transforms from the COM coordinates into the Principal Axes coordinates
    pa_inertias: dict
        key = isotopologue_name: str
        value = np.array[float]
            Principal Axes inertias, 3 columns and 3 rows
    pa_coordinates: dict
        key = isotopologue_name: str
        value = np.array[float]
            Principal Axes coordinates with n_atoms rows for atoms, 3 columns for axes
    COM_values: dict
        key = isotopologue_name: str
        value = np.array[float] of length 3
            Center-of-Mass position in the original coordinate system
    theta_data: dict or None, optional
        key = isotopologue_name: str
        value = dict: [str, float] theta values
        If None, theta dataframes are not computed.

    Returns
    -------
    atom_masses_df: pd.DataFrame
        RowLabel = Atom
        ColumnLabel = IsotopologueName
        Values = ExactMasses
        FinalRow = Total
    rotational_constants_df: pd.DataFrame
        RowLabel = Axis
        ColumnLabel = IsotopologueName
        Values = RotationalConstant
    dipole_components_df: pd.DataFrame
        RowLabel = Axis
        ColumnLabel = IsotopologueName
        Values = PrincipalAxesDipoleComponent
    com_coordinates_df_dict: dict
        key = isotopologue_name: str
        value = dataframe: pd.DataFrame
            RowLabel = Atom
            ColumnLabel = COMCoordinateAxis
            Values = COMCoordinateValue
    com_inertias_df_dict: dict
        key = isotopologue_name: str
        value = dataframe: pd.DataFrame
            RowLabel = COMCoordinateAxis
            ColumnLabel = COMCoordinateAxis
            Values = COMInertia
    eigenvectors_df_dict: dict
        key = isotopologue_name: str
        value = dataframe: pd.DataFrame
            RowLabel = COMCoordinateAxis
            ColumnLabel = PrincipalAxis
            Values = EigenvectorComponents
    pa_inertias_df_dict: dict
        key = isotopologue_name: str
        value = dataframe: pd.DataFrame
            RowLabel = PrincipalAxis
            ColumnLabel = PrincipalAxis
            Values = PrincipalAxisInertia
    pa_coordinates_df_dict: dict
        key = isotopologue_name: str
        value = dataframe: pd.DataFrame
            RowLabel = Atom
            ColumnLabel = PrincipalAxis
            Values = PrincipalAxesCoordinateValue
    com_values_df: pd.DataFrame
        RowLabel = Axis
        ColumnLabel = IsotopologueName
        Values = COMValue
    theta_df: pd.DataFrame or None
        RowLabel = isotopologue_name: str
        ColumnLabel = theta_label: str
        Values = theta_value: float
        None if theta_data was not provided.
    """
    # Atomic masses
    atom_masses_df = get_atom_masses_df(atom_masses, atom_symbols)

    # Rotational constants
    rotational_constants_df = get_rotational_constants_df(rotational_constants)

    # Dipole moments
    dipole_components_df = get_dipole_components_df(pa_dipoles)

    # COM values
    com_values_df = get_COM_values_df(COM_values)

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

    # Theta data
    theta_df = (
        get_theta_df(isotopologue_names, theta_data) if theta_data is not None else None
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
        com_values_df,
        theta_df,
    )
