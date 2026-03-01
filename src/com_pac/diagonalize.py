#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #
from mendeleev.mendeleev import isotope

from mendeleev import element
import numpy as np


# TODO: rename function "inertia_matrix" to "get_inertia_matrix"
def inertia_matrix(coordinates_array, masses_array):
    # TODO: properly vectorize this function
    matrix = np.zeros((3, 3))
    for axis1 in [0, 1, 2]:
        [axis2, axis3] = [x for x in [0, 1, 2] if x != axis1]
        diagonal = sum(
            [
                (
                    masses_array[i]
                    * (
                        (coordinates_array[i][axis2]) ** 2
                        + (coordinates_array[i][axis3]) ** 2
                    )
                )
                for i in range(len(masses_array))
            ]
        )
        off_diagonal = (-1) * sum(
            [
                masses_array[i]
                * coordinates_array[i][axis2]
                * coordinates_array[i][axis3]
                for i in range(len(masses_array))
            ]
        )
        matrix[axis1, axis1] = diagonal
        matrix[axis2, axis3] = off_diagonal
        matrix[axis3, axis2] = off_diagonal
    return matrix


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
    try:
        mass = isotope(symbol, mass_number).mass
    except Exception as exc:
        raise ValueError(
            f"Isotopic mass not found for {symbol} with mass number {mass_number}."
        )

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
    # TODO: properly vectorize this calculation to take advantage of numpy speed
    if len(masses) != len(coordinates):
        raise ValueError(
            'Length of "masses" array must match length of "coordinates" array.'
        )

    n_points = len(masses)
    COM = (1 / masses.sum()) * np.array(
        [masses[x] * coordinates[x] for x in range(0, n_points)]
    ).sum(axis=0)
    return np.array([x - COM for x in coordinates])


def get_eigens(matrix):
    """
    This implementation was "state of the art" at the time of initial writing, ca. 2022.
    """
    # TODO: Update to latest (?) implementation for numpy >=2.2
    evals, evecs = np.linalg.eig(matrix)
    sort_key = evals.argsort()[::1]
    evals = evals[sort_key]
    evecs = evecs[:, sort_key]
    return evals, evecs


def rotate_coordinates(coordinates, rotation):
    # TODO: Make sure "rotation" does not do a mirror inversion!
    #       That is, the right hand vector of any 3 non-colinear points  should be
    #       the same before **and** after the rotation.
    #       (Current hypothesis is that arbitrary sorting of evecs with the same
    #        eval is responsible for mirror inversion.)
    rotated_coordinates = np.dot(coordinates, rotation)
    return rotated_coordinates


def get_isotopologue_principal_axes(
    mol_coordinates, atom_mass_numbers, atom_symbols, n_atoms
):
    # Lookup exact masses
    mol_masses = get_mol_masses(atom_symbols, atom_mass_numbers, n_atoms)
    # Shift into Center of Mass coordinate system
    com_coordinate = get_COM_coordinates(mol_masses, mol_coordinates)
    # Calculate inertia matrix in COM system
    com_inertia = inertia_matrix(com_coordinate, mol_masses)
    # Diagonalize said matrix
    evals, evecs = get_eigens(com_inertia)
    # Use resulting eigenvectors to rotate COM system into Principal Axes system
    pa_coordinate = rotate_coordinates(com_coordinate, evecs)
    # Calculate inertia matrix in PA system, to later check if actually diagonalized
    pa_inertia = inertia_matrix(pa_coordinate, mol_masses)

    return (
        mol_masses,
        com_coordinate,
        com_inertia,
        evecs,
        evals,
        pa_coordinate,
        pa_inertia,
    )


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

    bad_diagonal_warnings = {}

    for iso in isotopologue_names:
        # validate data
        atom_mass_numbers = isotopologue_dict[iso]
        if len(atom_mass_numbers) != n_atoms:
            raise ValueError(
                f"Number of atoms in isotopologue_dict[{iso}] does not match number of atoms in coordinates."
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
        pa_dipoles[iso] = abs(np.dot(mol_dipole, evecs))
        rotational_constants[iso] = list(map(inertia_to_rot_const, evals))

        if not np.allclose(pa_inertia, np.diag(evals)):
            bad_diagonal_pas = """WARNING! The inertia matrix calculated using the principal axes system 
    is not diagonal for {}""".format(
                iso
            )
            print(bad_diagonal_pas)
            bad_diagonal_warnings[iso] = bad_diagonal_pas

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
    )
