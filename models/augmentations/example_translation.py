import torch

from models.components.crystal import Crystal

# To run write "python -m models.augmentations.example_translation"

# Example usage
if __name__ == "__main__":
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

    # Define translation coefficients
    coefficients = torch.tensor([1, 1, 1], dtype=torch.int32)  # Translate by one unit cell in each direction

    # Create a Crystal object
    crystal = Crystal(A, X, L)

    # Translate the crystal
    A_translated, X_translated, L_translated = crystal.translate_crystal(coefficients)

    print("Original Coordinates X:")
    print(X)
    print("\nTranslated Coordinates X_translated:")
    print(X_translated)

    print("\nOriginal Lattice L:")
    print(L)
    print("\nTranslated Lattice L_translated:")
    print(L_translated)

