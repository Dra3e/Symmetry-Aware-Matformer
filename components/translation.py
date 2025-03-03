import numpy as np

def TranslateRand(A, X, L, direction):
    """
    Move a crystal structure through 3d space by a directional vector.

    Parameters:
    - A: numpy array of shape (n, f) containing atomic features (remains unchanged).
    - X: numpy array of shape (n, 3) containing atomic coordinates.
    - L: numpy array of shape (3, 3) representing the lattice vectors (remains unchanged).
    
    Returns:
    - A_translated: same as A.
    - X_translated: numpy array of shape (n, 3) with atomic coordinates shifted by a directional vector.
    - L_translated: same as L
    """
    try:
        if(X.shape[1]!=3):
            raise Exception("X-matrix is not Nx3")
    except AttributeError:
        raise Exception("Invalide Input")
    else:
        X_translated = np.empty([X.shape[0],3]) # Allocates memory for the translated X-matrix
        for r in range(X.shape[0]): # Goes through every row of the X-matrix and adds the directional vector
            X_translated[r] = np.add(X[r],direction) 
        return A, X_translated, L
    