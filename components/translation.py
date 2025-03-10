import numpy as np

def TranslateRand(A, X, L, coefficients): 
    """
    Move a crystal structure through 3D space by a translation vector that is a multiple of the lattice vectors.

    Parameters:
    - A: numpy array of shape (n, f) containing atomic features (remains unchanged).
    - X: numpy array of shape (n, 3) containing atomic coordinates.
    - L: numpy array of shape (3, 3) representing the lattice vectors (remains unchanged).
    - coefficients: numpy array of shape (3,) containing integer coefficients for the lattice translation.

    Returns:
    - A_translated: same as A.
    - X_translated: numpy array of shape (n, 3) with atomic coordinates shifted by a lattice-based translation vector.
    - L_translated: same as L.
    """
    try:
        if A is None:
            raise ValueError("A can't be none")
        if X.shape[1] != 3:
            raise ValueError("X-matrix is not Nx3")
        if L.shape != (3, 3):
            raise ValueError("Lattice matrix L must be 3x3")
        if len(coefficients) != 3 or not np.issubdtype(coefficients.dtype, np.integer):
            raise ValueError("Translation coefficients must be a 3-element integer array")
    except AttributeError:
        raise ValueError("Invalid Input")
    
    # Compute translation vector as an integer combination of lattice vectors
    translation_vector = np.dot(coefficients, L)

    # Apply translation
    X_translated = X + translation_vector

    return A, X_translated, L
