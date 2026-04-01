"""
Exploring an alternative way to rotate coordinates into the principal axes frame.
"""

import numpy as np
from com_pac.diagonalize import get_inertia_matrix, rotate_coordinates


def rotation_solver(coordinates, masses):
    """
    Given the coordinates and masses, we can calculate the inertia matrix.
    The goal is to rotate the coordinates such that the inertia matrix is diagonal.
    Let's try to solve this numerically.
    """
    tolerance = 1e-6
    loss = np.inf
    rotation_matrix = np.eye(3)  # Start with the identity rotation
    counter = 0
    while loss > tolerance:
        rotated_coordinates = rotate_coordinates(coordinates, rotation_matrix)
        inertia_matrix = get_inertia_matrix(rotated_coordinates, masses)
        # Derivative of the inertia_matrix with respect to the rotation matrix is complex, so we will use a numerical approximation.
        # For simplicity, let's just use a small random perturbation to the rotation matrix and see if it reduces the off-diagonal elements of the inertia matrix.
        perturbation = np.random.randn(3, 3) * 0.01  # Small random perturbation
        new_rotation_matrix = rotation_matrix + perturbation
        new_rotation_matrix = new_rotation_matrix / np.linalg.norm(
            new_rotation_matrix, axis=0
        )  # Normalize columns to maintain orthogonality
        new_rotated_coordinates = rotate_coordinates(coordinates, new_rotation_matrix)
        new_inertia_matrix = get_inertia_matrix(new_rotated_coordinates, masses)
        # Calculate the loss as the sum of squares of the off-diagonal elements of the inertia matrix
        loss = np.sum(inertia_matrix**2) - np.sum(np.diag(inertia_matrix) ** 2)
        new_loss = np.sum(new_inertia_matrix**2) - np.sum(
            np.diag(new_inertia_matrix) ** 2
        )
        if new_loss < loss:
            rotation_matrix = new_rotation_matrix  # Accept the new rotation matrix if it reduces the loss
            loss = new_loss  # Update the loss to the new loss
        counter += 1
        if counter % 100 == 0:
            print(f"Iteration {counter}, Loss: {loss}")

    return rotation_matrix
