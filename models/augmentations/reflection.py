import torch

class Plane:
    def __init__(self, normal_vector, point=None):
        self.normal_vector = normal_vector
        self.point = point

    def reflection_matrix(self):
        """
        Compute the reflection matrix R for a given mirror plane's normal vector.
        Assumes 'normal_vector' is a non-zero vector; it is normalized inside the function.
        
        R = I - 2 * n * n^T
        """
        normal_vector = self.normal_vector.clone().detach().float()
        norm = torch.norm(normal_vector)  # Compute the length of normal_vector
        if norm == 0:
            raise ValueError("Normal vector cannot be the zero vector.")
        normal_vector /= norm
        R = torch.eye(3) - 2 * torch.outer(normal_vector, normal_vector)  # Compute the reflection matrix
        return R

    def reflect_crystal(self, A, X, L):
        """
        Reflect a crystal structure across a mirror plane.

        Parameters:
        - A: Tensor of shape (n, f) containing atomic features (remains unchanged).
        - X: Tensor of shape (n, 3) containing atomic coordinates.
        - L: Tensor of shape (3, 3) representing the lattice vectors.
        - normal_vector: Tensor of length 3, the normal vector of the mirror plane.
        - point: Optional Tensor of length 3, a point on the mirror plane. If None, 
                the mirror plane is assumed to pass through the origin.
        
        Returns:
        - A_reflected: Same as A.
        - X_reflected: Tensor of shape (n, 3) with reflected atomic coordinates.
        - L_reflected: Same as L.
        """
        R = self.reflection_matrix()
        
        if self.point is None:
            # Mirror plane through the origin
            X_reflected = torch.matmul(R, X.T).T
        else:
            point = self.point.clone().detach().float()
            # Subtract the point, reflect, then add it back
            X_reflected = torch.matmul(R, (X - point).T).T + point

        return A, X_reflected, L