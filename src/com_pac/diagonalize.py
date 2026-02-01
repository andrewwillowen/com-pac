#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #

from mendeleev import element
import numpy as np


def inertia_matrix(coordinates_array, masses_array):
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


def get_principal_axes(
    isotopologue_names,
    isotopologue_dict,
    n_atoms,
    atom_symbols,
    mol_coordinates,
    mol_dipole,
):
    rotational_constants = {}
    atom_masses = {}
    com_coordinates = {}
    com_inertias = {}
    eigenvectors = {}
    eigenvalues = {}
    pa_dipoles = {}
    pa_coordinates = {}
    pa_inertias = {}
    bad_diagonal_warnings = {}

    for iso in isotopologue_names:
        atom_mass_numbers = isotopologue_dict[iso]
        isotopes = []
        for atom in range(0, n_atoms):
            isotopes.append([atom_symbols[atom], atom_mass_numbers[atom]])
        masses = []
        for label, mass_number in isotopes:
            mass = None
            for isotope in element(label).isotopes:
                if isotope.mass_number == mass_number:
                    mass = isotope.mass
            if mass is None:
                raise ValueError(
                    "\n    Isotopic mass not found for {} with mass number {}\n".format(
                        label, mass_number
                    )
                )
            else:
                masses.append(mass)
        mol_masses = np.array(masses)
        atom_masses[iso] = mol_masses
        mol_COM = (1 / mol_masses.sum()) * np.array(
            [mol_masses[x] * mol_coordinates[x] for x in range(0, n_atoms)]
        ).sum(axis=0)
        com_coordinate = np.array([x - mol_COM for x in mol_coordinates])
        com_coordinates[iso] = com_coordinate
        com_inertia = inertia_matrix(com_coordinate, mol_masses)
        com_inertias[iso] = com_inertia
        evals, evecs = np.linalg.eig(com_inertia)
        sort_key = evals.argsort()[::1]
        evals = evals[sort_key]
        evecs = evecs[:, sort_key]
        eigenvectors[iso] = evecs
        eigenvalues[iso] = evals
        pa_coordinate = np.dot(com_coordinate, evecs)
        pa_coordinates[iso] = pa_coordinate
        pa_inertia = inertia_matrix(pa_coordinate, mol_masses)
        pa_inertias[iso] = pa_inertia
        if not np.allclose(pa_inertia, np.diag(evals)):
            bad_diagonal_pas = """WARNING! The inertia matrix calculated using the principal axes system 
    is not diagonal for {}""".format(
                iso
            )
            print(bad_diagonal_pas)
            bad_diagonal_warnings[iso] = bad_diagonal_pas
        pa_dipoles[iso] = abs(np.dot(mol_dipole, evecs))
        rotational_constants[iso] = list(map(inertia_to_rot_const, evals))

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
