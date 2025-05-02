import torch

from models.augmentations.reflection import Plane

# To run write "python -m models.augmentations.example_reflection"

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
        [0.0, 0.0, 0.0],
        [2.5, 2.5, 2.5],
    ], dtype=torch.float32)
    # L is the lattice matrix (here, identity for simplicity).
    L = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ], dtype=torch.float32)

    # Define a mirror plane. For instance, use the x-z plane:
    normal = torch.tensor([0, 1, 0], dtype=torch.float32)  # Normal vector along y-axis
    # Optionally, you can define a point p on the plane; if omitted, p defaults to origin.
    p = None  # Mirror plane passes through the origin

    plane = Plane(normal, p)
    A_ref, X_ref, L_ref = plane.reflect_crystal(A, X, L)

    print("Original Coordinates X:")
    print(X)
    print("\nReflected Coordinates X_ref:")
    print(X_ref)

    print("\nOriginal Lattice L:")
    print(L)
    print("\nReflected Lattice L_ref:")
    print(L_ref)
