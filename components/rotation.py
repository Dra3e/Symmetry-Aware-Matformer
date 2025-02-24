import numpy as np

def rotate_3d_object(points, axis, angle_degrees, lattice=None):
    """
    Rotates a set of 3D points and an optional lattice around a given axis by a specified angle in degrees.
    
    Parameters:
    - points (np.ndarray): An (N, 3) array of 3D points.
    - axis (np.ndarray): A 3D vector representing the rotation axis.
    - angle_degrees (float): The rotation angle in degrees.
    - lattice (np.ndarray, optional): A (3, 3) lattice matrix to be rotated.
    
    Returns:
    - tuple: Rotated 3D points and optionally rotated lattice if provided.
    """
    angle = np.deg2rad(angle_degrees)  # Convert degrees to radians
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Apply rotation
    rotated_points = np.dot(points, R.T)
    rotated_points = np.round(rotated_points, decimals=6)  # Round for cleaner output
    
    rotated_lattice = None
    if lattice is not None:
        rotated_lattice = np.dot(lattice, R.T)
        rotated_lattice = np.round(rotated_lattice, decimals=6)
    
    return rotated_points, rotated_lattice

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
