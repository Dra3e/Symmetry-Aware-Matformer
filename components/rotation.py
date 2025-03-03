import numpy as np

def rotate_3d_object(A, X, L, axis, angle_degrees):
    """
    Rotates a set of 3D points and an optional lattice around a given axis by a specified angle in degrees.
    
    Parameters:
    - A: numpy array of shape (n, f) containing atomic features (remains unchanged).
    - X (np.ndarray): An (N, 3) array of 3D points.
    - L (np.ndarray, optional): A (3, 3) lattice matrix to be rotated.
    - axis (np.ndarray): A 3D vector representing the rotation axis.
    - angle_degrees (float): The rotation angle in degrees.
    
    Returns:
    - A_rotated: same as A.
    - X_rotated: numpy array of shape (n, 3) with rotated atomic coordinates.
    - L_rotated: numpy array of shape (3, 3) with the rotated lattice.
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
    X_rotated = np.dot(X, R.T)
    X_rotated = np.round(X_rotated, decimals=6)  # Round for cleaner output
    
    L_rotated = np.dot(L, R.T)
    L_rotated = np.round(L_rotated, decimals=6)
    
    return A, X_rotated, L_rotated
