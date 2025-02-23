import numpy as np

def rotate_3d_object(points, axis, angle_degrees):
    """
    Rotates a set of 3D points around a given axis by a specified angle in degrees.
    
    Parameters:
    - points (np.ndarray): An (N, 3) array of 3D points.
    - axis (np.ndarray): A 3D vector representing the rotation axis.
    - angle_degrees (float): The rotation angle in degrees.
    
    Returns:
    - np.ndarray: Rotated 3D points.
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
    return np.round(rotated_points, decimals=6)  # Round for cleaner output

# Example usage
if __name__ == "__main__":
    points = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])
    axis = [0, 0, 1]  # Rotate around the Z-axis
    angle = 90  # 90 degrees
    
    rotated = rotate_3d_object(points, axis, angle)
    print("Rotated Points:")
    for point in rotated:
        print(point)
