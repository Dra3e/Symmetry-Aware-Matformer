import numpy as np

def reflection_matrix(normal_vector):
    """
    Compute the reflection matrix R for a given mirror plane's normal vector.
    Assumes 'normal_vector' is a non-zero vector; it is normalized inside the function.
    
    R = I - 2 * n * n^T
    """
    normal_vector = np.array(normal_vector, dtype=float)
    norm = np.linalg.norm(normal_vector) # Compute the length of norm_vector
    if norm == 0:
        raise ValueError("Normal vector cannot be the zero vector.")
    normal_vector /= norm
    R = np.eye(3) - 2 * np.outer(normal_vector, normal_vector) # Compute the plane by 'dotting' the normal_vector using outer product
    return R

def reflect_crystal(A, X, L, normal_vector, point=None):
    """
    Reflect a crystal structure across a mirror plane.

    Parameters:
    - A: numpy array of shape (n, f) containing atomic features (remains unchanged).
    - X: numpy array of shape (n, 3) containing atomic coordinates.
    - L: numpy array of shape (3, 3) representing the lattice vectors.
    - normal_vector: array-like of length 3, the normal vector of the mirror plane.
    - point: optional array-like of length 3, a point on the mirror plane. If None, 
         the mirror plane is assumed to pass through the origin.
    
    Returns:
    - A_reflected: same as A.
    - X_reflected: numpy array of shape (n, 3) with reflected atomic coordinates.
    - L_reflected: numpy array of shape (3, 3) with the reflected lattice.
    """
    R = reflection_matrix(normal_vector)
    
    if point is None:
        # Mirror plane through the origin
        X_reflected = np.dot(R, X.T).T
    else:
        point = np.array(point, dtype=float)
        # Subtract the point, reflect, then add it back.
        X_reflected = np.dot(R, (X - point).T).T + point

    # Update the lattice so that the periodicity is preserved.
    L_reflected = np.dot(R, L)

    return A, X_reflected, L_reflected
