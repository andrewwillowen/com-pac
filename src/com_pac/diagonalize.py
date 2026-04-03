#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #
from mendeleev.mendeleev import isotope

import numpy as np

import warnings

ISOTOPE_MASS_CACHE = {}


def clear_isotope_mass_cache():
    ISOTOPE_MASS_CACHE.clear()


def get_inertia_matrix(coordinates_array, masses_array):
    """Calculate the inertia matrix

    Parameters
    ----------
    coordinates_array : array-like
        Array of shape (n_atoms, 3) containing the Cartesian coordinates of the atoms.
    masses_array : array-like
        Array of shape (n_atoms,) containing the masses of the atoms.

    Returns
    -------
    np.ndarray
        Inertia matrix of shape (3, 3).

    Notes
    -----
     This implementation uses the identity

     ``I = (sum_i m_i r_i^2) * 1 - sum_i m_i r_i r_i^T``

     where ``r_i = [x_i, y_i, z_i]`` is the Cartesian coordinate vector for atom ``i``.

     The calculation proceeds in vectorized form:

     1. Square each coordinate component element-wise:
         ``coordinates_squared = coordinates**2``

     2. Compute ``r_i^2 = x_i^2 + y_i^2 + z_i^2`` for each atom:
         ``r2 = np.sum(coordinates_squared, axis=1)``

     3. Mass-weight these values:
         ``mr2 = masses * r2``

     4. Sum over atoms to obtain the common diagonal trace term:
         ``mr2_sum = np.sum(mr2)``

     5. Build ``(sum_i m_i r_i^2) * 1``:
         ``mr2_trace = np.eye(3) * mr2_sum``

     6. Compute ``sum_i m_i r_i r_i^T`` using broadcasting and matrix multiplication:
         ``weighted_transpose_broadcast = (coordinates * masses[:, np.newaxis]).T @ coordinates``

     The final inertia matrix is then

     ``inertia_matrix = mr2_trace - weighted_transpose_broadcast``

     which yields the correct diagonal and off-diagonal terms in one expression.
    """
    coordinates = np.asarray(coordinates_array)
    masses = np.asarray(masses_array)

    # The math in the docstring can be condensed into these lines:
    mr2_trace = np.eye(3) * np.sum(masses * np.sum(coordinates**2, axis=1))
    weighted_transpose_broadcast = (coordinates * masses[:, np.newaxis]).T @ coordinates
    return mr2_trace - weighted_transpose_broadcast


def inertia_to_rot_const(inertia):
    rot_constant = 505379.0046 / inertia
    return rot_constant


def get_isotopes_dict(atom_symbols, atom_mass_numbers, n_atoms):
    # TODO: When pulled into a proper data structure, add type checking.
    isotopes_dict = {
        i: {"symbol": atom_symbols[i], "mass_number": atom_mass_numbers[i]}
        for i in range(0, n_atoms)
    }
    return isotopes_dict


def get_unique_isotopes(isotopes_dict):
    as_tuples = [(v["symbol"], v["mass_number"]) for k, v in isotopes_dict.items()]
    unique_isotopes = set(as_tuples)
    return unique_isotopes


def get_isotopes_mass(symbol, mass_number):
    cache_key = (symbol, mass_number)
    if cache_key in ISOTOPE_MASS_CACHE:
        return ISOTOPE_MASS_CACHE[cache_key]

    try:
        mass = isotope(symbol, mass_number).mass
    except Exception as exc:
        raise ValueError(
            f"Isotopic mass not found for {symbol} with mass number {mass_number}."
        ) from exc

    ISOTOPE_MASS_CACHE[cache_key] = mass

    return mass


def get_unique_isotopes_mass_dict(unique_isotopes):
    # TODO: This can be done for **all** the isotopologues in the input file
    #       right after parsing the input file - that way, each isotopologue
    #       isn't repeating the lookup for its atoms; only has to read the
    #       main dictionary.
    mass_dict = {
        iso_tuple: get_isotopes_mass(*iso_tuple) for iso_tuple in unique_isotopes
    }
    return mass_dict


def get_mol_masses(atom_symbols, atom_mass_numbers, n_atoms):
    # TODO: Should be able to inherit the `isotopes` data structure from the input parser.
    isotopes_dict = get_isotopes_dict(atom_symbols, atom_mass_numbers, n_atoms)
    unique_isos_dict = get_unique_isotopes_mass_dict(get_unique_isotopes(isotopes_dict))
    masses = [
        unique_isos_dict[(atom_symbols[i], atom_mass_numbers[i])]
        for i in range(0, n_atoms)
    ]
    return np.array(masses)


def get_COM_coordinates(masses, coordinates):
    if len(masses) != len(coordinates):
        raise ValueError(
            'Length of "masses" array must match length of "coordinates" array.'
        )

    n_points = len(masses)
    masses_sum = masses.sum()
    if np.allclose(0, masses_sum):
        raise ValueError('Sum of "masses" array is zero!')

    COM = (1 / masses_sum) * np.dot(masses, coordinates)
    return np.subtract(coordinates, COM), COM


def get_eigens(matrix, sign_convention=False, right_handed=True):
    """Diagonalize a real symmetric matrix and return eigenvalues and eigenvectors
    with an optionally standardized orientation.

    Parameters
    ----------
    matrix : array-like
        A real symmetric 3×3 matrix (e.g. an inertia tensor).
    sign_convention : bool, optional
        When ``True``, each eigenvector column is negated if the element
        with the largest absolute value is negative.  This removes the
        arbitrary ±1 sign that ``eigh`` may assign.  Defaults to ``False``.
    right_handed : bool, optional
        When ``True`` (the default), the eigenvector matrix is guaranteed to
        form a right-handed coordinate system (det = +1).  If the determinant
        is −1 the last column is negated to correct the orientation.
        Defaults to ``True``.

    Returns
    -------
    evals : np.ndarray, shape (3,)
        Eigenvalues sorted in ascending order (Ia ≤ Ib ≤ Ic).
    evecs : np.ndarray, shape (3, 3)
        Corresponding eigenvectors as columns.

    Notes
    -----
    ``np.linalg.eigh`` is used instead of ``np.linalg.eig`` because the
    inertia matrix is always real and symmetric.  ``eigh`` guarantees real
    eigenvalues, an orthonormal eigenvector basis, and ascending sort order
    — making an explicit ``argsort`` unnecessary.

    Two optional normalisation steps can be applied after solving:

    1. **Sign convention** (controlled by ``sign_convention``) — for each
       eigenvector column the element with the largest absolute value is made
       non-negative by negating the whole column if needed.

    2. **Right-handedness** (controlled by ``right_handed``) — if the
       determinant of the eigenvector matrix is −1 (i.e. a reflection rather
       than a proper rotation), the last column is negated to promote the
       result to a proper rotation (det = +1).  Physically, this ensures that
       the principal axis frame is consistently right-handed across all
       isotopologues so that relative atomic positions are not mirror-inverted.
    """
    evals, evecs = np.linalg.eigh(matrix)

    # Step 1: deterministic sign convention per eigenvector column.
    if sign_convention:
        for i in range(evecs.shape[1]):
            col = evecs[:, i]
            if col[np.argmax(np.abs(col))] < 0:
                evecs[:, i] = -col

    # Step 2: enforce right-handed coordinate system.
    if right_handed and np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1

    return evals, evecs


def rotate_coordinates(coordinates, rotation):
    rotated_coordinates = np.dot(coordinates, rotation)
    return rotated_coordinates


def get_isotopologue_principal_axes(
    mol_coordinates, atom_mass_numbers, atom_symbols, n_atoms
):
    # Lookup exact masses
    mol_masses = get_mol_masses(atom_symbols, atom_mass_numbers, n_atoms)
    # Shift into Center of Mass coordinate system
    com_coordinate, COM = get_COM_coordinates(mol_masses, mol_coordinates)
    # Calculate inertia matrix in COM system
    com_inertia = get_inertia_matrix(com_coordinate, mol_masses)
    # Diagonalize said matrix
    evals, evecs = get_eigens(com_inertia)
    # Use resulting eigenvectors to rotate COM system into Principal Axes system
    pa_coordinate = rotate_coordinates(com_coordinate, evecs)
    # Calculate inertia matrix in PA system, to later check if actually diagonalized
    pa_inertia = get_inertia_matrix(pa_coordinate, mol_masses)

    return (
        mol_masses,
        com_coordinate,
        com_inertia,
        evecs,
        evals,
        pa_coordinate,
        pa_inertia,
        COM,
    )


def get_theta_values(
    isotopologue_names,
    atom_masses,
    pa_coordinates,
):
    """Calculate theta values for each isotopologue.

    Parameters
    ----------
    isotopologue_names : list[str]
        List of isotopologue names.
    atom_masses : dict
        key = isotopologue_name: str
        value = np.array[float] of atom masses
    pa_coordinates : dict
        key = isotopologue_name: str
        value = np.array[float] of PA coordinates

    Returns
    -------
    theta_data : dict or None
        key = isotopologue_name: str
        value = theta results (TBD)
        None if parent isotopologue is not planar in it's principal axes frame.

    """
    # The paper says that the all the coordinates need to be in the
    # principal axes frame of the parent species, i.e., the first isotopologue
    parent_iso = isotopologue_names[0]
    parent_coordinates = pa_coordinates[parent_iso]
    parent_masses = atom_masses[parent_iso]
    # Need to check if planar - paper does not describe how to handle non-planar...
    parent_inertia_matrix = get_inertia_matrix(parent_coordinates, parent_masses)
    piac = parent_inertia_matrix[0, 2]
    pica = parent_inertia_matrix[2, 0]
    pibc = parent_inertia_matrix[1, 2]
    picb = parent_inertia_matrix[2, 1]
    if not np.allclose([0, 0, 0, 0], [piac, pica, pibc, picb]):
        print(
            "WARNING: Parent isotopologue is non-planar - skipping Theta calculation!"
        )
        return None

    # To get the isos in the parent coordinates, just need to pair with the
    # correct masses.
    #
    theta_results_dict = {}
    iso_inertia_in_parent_dict: dict = {}
    theta_7_dict: dict = {}
    theta_8_dict: dict = {}
    theta_9_parent_dict: dict = {}
    theta_9_iso_dict: dict = {}
    for iso in isotopologue_names:
        # Next, need to calculate the inertia tensor for each iso
        inertia_matrix = get_inertia_matrix(parent_coordinates, atom_masses[iso])
        # theta_equilibrium in equation 7 is given by
        #   tan(2*theta_7) = (2 * I{e}_ab) / (I{e}_aa - I{e}bb)
        # or
        #   theta_7 = (1/2) * arctan((2 * I{e}_ab) / (I{e}_aa - I{e}bb))
        iab = inertia_matrix[0, 1]
        iaa = inertia_matrix[0, 0]
        ibb = inertia_matrix[1, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                theta_7 = (1 / 2) * np.arctan((2 * iab) / (iaa - ibb))
                theta_7 = np.degrees(theta_7)
            except Exception as e:
                print(f"""{iso} encountered error while calculating theta_7
    {iab=}
    {iaa=}
    {ibb=}
    (2 * iab) / (iaa - ibb) = {(2*iab) / (iaa - ibb)}
Exception: {e}""")
                theta_7 = None

        # theta_equilibrium in equation 8 is given by
        #   I{e}_aa - I{e}_bb = (I{e}_a - I{e}_b)*cos(2*theta_8)
        # or
        #   theta_8 = (1/2)*arccos( (I{e}_aa - I{e}_bb) / (I{e}_a - I{e}_b) )
        # The implication here is that I{e}_a and I{e}_b are the moments of inertia
        # in the *isotopologue's principal axes frame*
        iso_pa_inertia = get_inertia_matrix(pa_coordinates[iso], atom_masses[iso])
        ia = iso_pa_inertia[0, 0]
        ib = iso_pa_inertia[1, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                theta_8 = (1 / 2) * np.arccos((iaa - ibb) / (ia - ib))
                theta_8 = np.degrees(theta_8)
            except Exception as e:
                print(f"""{iso} encountered error while calculating theta_8
    {iaa=}
    {ibb=}
    {ia=}
    {ib=}
    (iaa - ibb) / (ia - ib) = {(iaa - ibb) / (ia - ib)}
Exception: {e}""")
                theta_8 = None

        # theta_equilibrium in equation 9 is given by
        #   I{e}_ab = (1/2)*(I{e}_a - I{e}_b)*sin(2*theta_9)
        # or
        #   theta_9 = (1/2)*arcsin((2*I{e}_ab) / (I{e}_a - I{e}_b))
        # The problem, however, is that it is not clear from context whether I{e}_ab
        # is the off-diagonal element in the parent PA frame, or in the iso PA frame!
        # So I guess we'll calculate it twice
        #
        # Edit: nvm, the values of theta_9_iso are all within floating point precision of zero!
        #
        #   theta_9_parent_frame = (1/2)*arcsin((2*I{e, parent}_ab) / (I{e}_a - I{e}_b))
        iab_parent = iab
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                theta_9_parent_frame = (1 / 2) * np.arcsin((2 * iab_parent) / (ia - ib))
                theta_9_parent_frame = np.degrees(theta_9_parent_frame)
            except Exception as e:
                print(f"""{iso} encountered error while calculating theta_9_parent
    {iab_parent=}
    {ia=}
    {ib=}
    (2 * iab_parent) / (ia - ib) = {(2*iab_parent) / (ia - ib)}
Exception: {e}""")
                theta_9_parent_frame = None

        #         #   theta_9_iso_frame = (1/2)*arcsin((2*I{e, iso}_ab) / (I{e}_a - I{e}_b))
        #         iab_iso = iso_pa_inertia[0, 1]
        #         with warnings.catch_warnings():
        #             warnings.simplefilter("error", RuntimeWarning)
        #             try:
        #                 theta_9_iso_frame = (1 / 2) * np.arcsin((2 * iab_iso) / (ia - ib))
        #             except Exception as e:
        #                 print(f"""{iso} encountered error while calculating theta_9_iso
        #     {iab_iso=}
        #     {ia=}
        #     {ib=}
        #     (2 * iab_iso) / (ia - ib) = {(2*iab_iso) / (ia - ib)}
        # Exception: {e}""")
        #                 theta_9_iso_frame = None

        # TODO: Andrew's naive vector approach

        # Update dictionaries
        theta_results_dict[iso] = {
            "theta_7": theta_7,
            "theta_8": theta_8,
            "theta_9_par": theta_9_parent_frame,
            # "theta_9_iso": theta_9_iso_frame,
        }

    return theta_results_dict


def check_for_length_mismatch(listlike, expected_length: int, message: str):
    if not isinstance(expected_length, int):
        raise TypeError("'expected_length' must be of type 'int'")

    actual_length = len(listlike)
    if actual_length != expected_length:
        raise ValueError(f"{actual_length=} vs {expected_length=}: {message}")


def check_for_bad_diagonal(matrix, eigenvalues, message):
    if not np.allclose(matrix, np.diag(eigenvalues)):
        print(message)


def transform_dipole(dipole, vectors):
    return abs(np.dot(dipole, vectors))


def get_principal_axes(
    isotopologue_names,
    isotopologue_dict,
    n_atoms,
    atom_symbols,
    mol_coordinates,
    mol_dipole,
):
    # initialize empty data structures
    atom_masses = {}
    com_coordinates = {}
    com_inertias = {}
    eigenvectors = {}
    eigenvalues = {}
    pa_dipoles = {}
    pa_coordinates = {}
    pa_inertias = {}
    rotational_constants = {}
    COM_values = {}

    bad_diagonal_warnings = {}

    for iso in isotopologue_names:
        # validate data
        atom_mass_numbers = isotopologue_dict[iso]
        check_for_length_mismatch(
            atom_mass_numbers,
            n_atoms,
            f"Number of atoms in isotopologue_dict[{iso}] does not match number of atoms in coordinates.",
        )

        # do calculations

        (
            mol_masses,
            com_coordinate,
            com_inertia,
            evecs,
            evals,
            pa_coordinate,
            pa_inertia,
            COM,
        ) = get_isotopologue_principal_axes(
            mol_coordinates, atom_mass_numbers, atom_symbols, n_atoms
        )

        # update data structures with results of calculations
        atom_masses[iso] = mol_masses
        com_coordinates[iso] = com_coordinate
        com_inertias[iso] = com_inertia
        eigenvectors[iso] = evecs
        eigenvalues[iso] = evals
        pa_coordinates[iso] = pa_coordinate
        pa_inertias[iso] = pa_inertia
        pa_dipoles[iso] = transform_dipole(mol_dipole, evecs)
        rotational_constants[iso] = list(map(inertia_to_rot_const, evals))
        COM_values[iso] = COM

        check_for_bad_diagonal(
            pa_inertia,
            evals,
            f"WARNING! The inertia matrix calculated using the principal axes system is not diagonal for {iso}",
        )

    return (
        atom_masses,
        rotational_constants,
        pa_dipoles,
        pa_coordinates,
        pa_inertias,
        com_coordinates,
        com_inertias,
        eigenvectors,
        eigenvalues,
        COM_values,
    )
