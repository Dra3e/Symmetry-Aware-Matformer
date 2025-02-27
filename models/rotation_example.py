import numpy as np
from components.rotation import rotate_3d_object

#To run write "python -m models.rotation_example"

# Example usage
if __name__ == "__main__":
    A = np.array([
        [11, 1],
        [17, -1]
    ])
    X = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])
    L = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    axis = [0, 0, 1]  # Rotate around the Z-axis
    angle = 90  # 90 degrees
    
    A_rotated, X_rotated, L_rotated = rotate_3d_object(A, X, L, axis, angle)
    print("Rotated Atoms:")
    print(X_rotated)

    print("\nRotated Lattice:")
    print(L_rotated)