import numpy as np
from components.reflection import reflect_crystal

#To run write "python -m models.reflection_example"

# Example usage:
if __name__ == "__main__":
    # Example with NaCl
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


    # Define a mirror plane. For instance, use the x-z plane:
    normal = [0, 1, 0]  # Normal vector along y-axis
    # Optionally, you can define a point p on the plane; if omitted, p defaults to origin.
    p = None  # Mirror plane passes through the origin
    
    A_ref, X_ref, L_ref = reflect_crystal(A, X, L, normal, p)
    
    print("Original Coordinates X:")
    print(X)
    print("\nReflected Coordinates X_ref:")
    print(X_ref)
    
    print("\nOriginal Lattice L:")
    print(L)
    print("\nReflected Lattice L_ref:")
    print(L_ref)