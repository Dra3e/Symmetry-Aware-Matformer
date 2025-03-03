import numpy as np
from components.translation import TranslateRand

#To run write "python -m models.translation_example"

# Example usage
if __name__ == "__main__":
    # A is a 2 x 2 matrix with the atom nr. and charge in each row
    A = np.array([
        [11, 1],
        [17, -1]
    ])
    # X contains the atomic coordinates.
    X = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 2.5, 2.5],
    ])
    # L is the lattice matrix (here, identity for simplicity).
    L = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ])
    A_translated, X_translated, L_translated = TranslateRand(A, X, L)

    print("Translated Matrix:")
    print(X_translated)

