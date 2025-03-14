import torch

def translate_crystal(A, X, L, coefficients): 
    """
    Move a crystal structure through 3D space by a translation vector that is a multiple of the lattice vectors.

    Parameters:
    - A: Tensor of shape (n, f) containing atomic features (remains unchanged).
    - X: Tensor of shape (n, 3) containing atomic coordinates.
    - L: Tensor of shape (3, 3) representing the lattice vectors (remains unchanged).
    - coefficients: Tensor of shape (3,) containing integer coefficients for the lattice translation.

    Returns:
    - A_translated: Same as A.
    - X_translated: Tensor of shape (n, 3) with atomic coordinates shifted by a lattice-based translation vector.
    - L_translated: Same as L.
    """
    try:
        if not isinstance(coefficients, torch.Tensor):
            coefficients = torch.tensor(coefficients, dtype=torch.int32)
        if A is None:
            raise ValueError("A can't be none")
        if X.shape[1] != 3:
            raise ValueError("X-matrix is not Nx3")
        if L.shape != (3, 3):
            raise ValueError("Lattice matrix L must be 3x3")
        if len(coefficients) != 3 or not coefficients.dtype.is_floating_point:
            raise ValueError("Translation coefficients must be a 3-element integer tensor")
    except AttributeError:
        raise ValueError("Invalid Input")
    
    # Compute translation vector as an integer combination of lattice vectors
    translation_vector = torch.matmul(coefficients.float(), L)

    # Apply translation
    X_translated = X + translation_vector

    return A, X_translated, L