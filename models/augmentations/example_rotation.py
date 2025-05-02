import torch

from models.augmentations.rotation import Axis

# To run write "python -m models.augmentations.example_rotation"

# Example usage:
if __name__ == "__main__":
    # Example with NaCl
    # A is a 2 x 2 matrix with the atom nr. and charge in each row
    A = torch.tensor([
        [11, 1],
        [17, -1]
    ], dtype=torch.float32)
    # X contains the atomic coordinates.
    X = torch.tensor([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    # L is the lattice matrix (here, identity for simplicity).
    L = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Define a rotation axis. For instance, use the z-axis:
    axis_vector = torch.tensor([0, 0, 1], dtype=torch.float32)  # Rotation around z-axis
    rotation_degrees = 90  # Rotate by 90 degrees

    axis = Axis(axis_vector, rotation_degrees)
    A_rotated, X_rotated, L_rotated = axis.rotate_crystal(A, X, L)

    print("Rotated Atoms:")
    print(X_rotated)

    print("\nRotated Lattice:")
    print(L_rotated)
