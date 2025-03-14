import torch

def rotate_crystal(A, X, L, axis, angle_degrees):
    """
    Rotates a set of 3D points and an optional lattice around a given axis by a specified angle in degrees.
    
    Parameters:
    - A: Tensor of shape (n, f) containing atomic features (remains unchanged).
    - X: Tensor of shape (n, 3) containing atomic coordinates.
    - L: Tensor of shape (3, 3) representing the lattice vectors.
    - axis: Tensor of length 3, the rotation axis.
    - angle_degrees: The rotation angle in degrees.
    
    Returns:
    - A_rotated: Same as A.
    - X_rotated: Tensor of shape (n, 3) with rotated atomic coordinates.
    - L_rotated: Same as L.
    """
    angle = torch.deg2rad(torch.tensor(angle_degrees, dtype=torch.float32))  # Convert degrees to radians
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / torch.norm(axis)  # Normalize the rotation axis
    
    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=torch.float32)
    
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
    
    # Apply rotation
    X_rotated = torch.matmul(X, R.T)
    X_rotated = torch.round(X_rotated * 1e6) / 1e6  # Round for cleaner output
    
    return A, X_rotated, L