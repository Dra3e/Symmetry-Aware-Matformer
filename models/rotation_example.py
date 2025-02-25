import numpy as np
from components.rotation import rotate_3d_object

#To run write python -m models.rotation_example

# Example usage
if __name__ == "__main__":
    points = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])
    axis = [0, 0, 1]  # Rotate around the Z-axis
    angle = 90  # 90 degrees
    lattice = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    rotated_points, rotated_lattice = rotate_3d_object(points, axis, angle, lattice)
    print("Rotated Points:")
    for point in rotated_points:
        print(point)
    
    if rotated_lattice is not None:
        print("\nRotated Lattice:")
        for vector in rotated_lattice:
            print(vector)