"""
Investigating the potential axis inversion bug.
"""

import numpy as np

from com_pac.parser import parse_input_file
from com_pac.diagonalize import (
    get_mol_masses,
    get_COM_coordinates,
    get_eigens,
    get_inertia_matrix,
    rotate_coordinates,
)


def get_angle(u, v):
    u = np.asarray(u)
    v = np.asarray(v)

    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # avoid numeric issues
    return np.arccos(cos_theta)


def get_angle_degrees(u, v):
    return np.degrees(get_angle(u, v))


def get_angle_sign(u, v):
    u = np.asarray(u)
    v = np.asarray(v)

    cross_product = np.cross(u, v)
    sign = np.sign(cross_product[2])  # Assuming we are in the xy-plane
    return sign


x_point = np.array([0.0, 0.0, 1.0])
a_point = np.array([0.5, 0.5, 0.0])
x = np.array(x_point) / np.linalg.norm(x_point)
a = np.array(a_point) / np.linalg.norm(a_point)
print(f"x: {x}")
print(f"a: {a}")
print(
    f"Angle between x and a: {get_angle_sign(x, a) * get_angle_degrees(x, a)} degrees"
)


# input_file_path = "full-isos.txt"
# with open(input_file_path, "r") as infile:
#     input_file = infile.read()

# # Get data from input file
# (
#     isotopologue_names,
#     isotopologue_dict,
#     n_atoms,
#     atom_symbols,
#     mol_coordinates,
#     mol_dipole,
#     atom_numbering,
# ) = parse_input_file(input_file)


# # iso001

# iso001_atom_mass_numbers = isotopologue_dict["iso001"]
# iso001_mol_masses = get_mol_masses(atom_symbols, iso001_atom_mass_numbers, n_atoms)
# iso001_com_coordinate, iso001_com = get_COM_coordinates(
#     iso001_mol_masses, mol_coordinates
# )
# iso001_com_inertia = get_inertia_matrix(iso001_com_coordinate, iso001_mol_masses)
# iso001_evals, iso001_evecs = get_eigens(iso001_com_inertia)

# iso001_rhv = np.cross(iso001_com_coordinate[0], iso001_com_coordinate[1])

# # The rotation applied to the COM coordinates should result in the same right-hand-vector.
# # That is
# # A = (1) RHV, (2) Rotation
# # should be the same as
# # B = (1) Rotation, (2) RHV

# iso001_pa_coordinate = rotate_coordinates(iso001_com_coordinate, iso001_evecs)
# iso001_rhv_rotated = rotate_coordinates(iso001_rhv, iso001_evecs)
# iso001_rotated_rhv = np.cross(iso001_pa_coordinate[0], iso001_pa_coordinate[1])

# print(f"iso001: A == B? {np.allclose(iso001_rhv_rotated, iso001_rotated_rhv)}")

# # iso007

# iso007_atom_mass_numbers = isotopologue_dict["iso007"]
# iso007_mol_masses = get_mol_masses(atom_symbols, iso007_atom_mass_numbers, n_atoms)
# iso007_com_coordinate, iso007_com = get_COM_coordinates(
#     iso007_mol_masses, mol_coordinates
# )
# iso007_com_inertia = get_inertia_matrix(iso007_com_coordinate, iso007_mol_masses)
# iso007_evals, iso007_evecs = get_eigens(iso007_com_inertia)

# iso007_rhv = np.cross(iso007_com_coordinate[0], iso007_com_coordinate[1])
# iso007_pa_coordinate = rotate_coordinates(iso007_com_coordinate, iso007_evecs)
# iso007_rhv_rotated = rotate_coordinates(iso007_rhv, iso007_evecs)
# iso007_rotated_rhv = np.cross(iso007_pa_coordinate[0], iso007_pa_coordinate[1])

# print(f"iso007: A == B? {np.allclose(iso007_rhv_rotated, iso007_rotated_rhv)}")

# unit_vectors = np.eye(3)
# iso001_rotated_unit_vectors = rotate_coordinates(unit_vectors, iso001_evecs)
# iso007_rotated_unit_vectors = rotate_coordinates(unit_vectors, iso007_evecs)

# print(f"iso001 rotated unit vectors:\n{iso001_rotated_unit_vectors}")
# print(f"iso007 rotated unit vectors:\n{iso007_rotated_unit_vectors}")


# def angle_between(u, v):
#     u = np.asarray(u)
#     v = np.asarray(v)

#     cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)  # avoid numeric issues
#     return np.arccos(cos_theta)


# iso001_rotation = angle_between(unit_vectors[0], iso001_rotated_unit_vectors[0])
# iso007_rotation = angle_between(unit_vectors[0], iso007_rotated_unit_vectors[0])

# print(f"iso001 rotation angle: {iso001_rotation:.2f} radians")
# print(f"iso007 rotation angle: {iso007_rotation:.2f} radians")

# iso007_corrected_evecs = np.array([iso007_evecs[1], iso007_evecs[0], iso007_evecs[2]])
# iso007_corrected_rotated_unit_vectors = rotate_coordinates(
#     unit_vectors, iso007_corrected_evecs
# )
# iso007_corrected_rotation = angle_between(
#     unit_vectors[0], iso007_corrected_rotated_unit_vectors[0]
# )

# print(
#     f"iso007 corrected rotated unit vectors:\n{iso007_corrected_rotated_unit_vectors}"
# )
# print(f"iso007 corrected rotation angle: {iso007_corrected_rotation:.2f} radians")
# print(
#     f"Difference due to correction: {iso007_rotation - iso007_corrected_rotation:.2f} radians"
# )

# # iso008

# iso008_atom_mass_numbers = isotopologue_dict["iso008"]
# iso008_mol_masses = get_mol_masses(atom_symbols, iso008_atom_mass_numbers, n_atoms)
# iso008_com_coordinate, iso008_com = get_COM_coordinates(
#     iso008_mol_masses, mol_coordinates
# )
# iso008_com_inertia = get_inertia_matrix(iso008_com_coordinate, iso008_mol_masses)
# iso008_evals, iso008_evecs = get_eigens(iso008_com_inertia)
# iso008_rhv = np.cross(iso008_com_coordinate[0], iso008_com_coordinate[1])
# iso008_pa_coordinate = rotate_coordinates(iso008_com_coordinate, iso008_evecs)
# iso008_rhv_rotated = rotate_coordinates(iso008_rhv, iso008_evecs)
# iso008_rotated_rhv = np.cross(iso008_pa_coordinate[0], iso008_pa_coordinate[1])
# print(f"iso008: A == B? {np.allclose(iso008_rhv_rotated, iso008_rotated_rhv)}")
# iso008_rotated_unit_vectors = rotate_coordinates(unit_vectors, iso008_evecs)
# iso008_rotation = angle_between(unit_vectors[0], iso008_rotated_unit_vectors[0])
# print(f"iso008 rotated unit vectors:\n{iso008_rotated_unit_vectors}")
# print(f"iso008 rotation angle: {iso008_rotation:.2f} radians")
# iso008_corrected_evecs = np.array([iso008_evecs[1], iso008_evecs[0], iso008_evecs[2]])
# iso008_corrected_rotated_unit_vectors = rotate_coordinates(
#     unit_vectors, iso008_corrected_evecs
# )
# iso008_corrected_rotation = angle_between(
#     unit_vectors[0], iso008_corrected_rotated_unit_vectors[0]
# )
# print(
#     f"iso008 corrected rotated unit vectors:\n{iso008_corrected_rotated_unit_vectors}"
# )
# print(f"iso008 corrected rotation angle: {iso008_corrected_rotation:.2f} radians")
# print(
#     f"Difference due to correction: {iso008_rotation - iso008_corrected_rotation:.2f} radians"
# )
