import torch
import spglib
import numpy as np

class Crystal:
    def __init__(self, A, X, L):
        # Initialize the crystal with atomic features (A), atomic coordinates (X), and lattice vectors (L)
        self.A = A.clone().detach() if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
        self.X = X.clone().detach() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.L = L.clone().detach() if isinstance(L, torch.Tensor) else torch.tensor(L, dtype=torch.float32)

    import torch

    def apply_augmentations(self, symmetry_operations):
        """
        Applies symmetry operations to a crystal and returns all augmented crystal objects.

        Args:
            symmetry_operations (list of torch.Tensor): List of 3x3 rotation matrices representing symmetry operations.

        Returns:
            list of Crystal: List containing augmented Crystal objects, each with updated atomic positions.
        """
        augmented_crystals = []  # List to store augmented Crystal objects for each symmetry operation

        for op in symmetry_operations:
            # Create a new Crystal object with the current atomic positions and lattice
            augmented_crystal = Crystal(self.A, self.X, self.L)
            
            # Apply the symmetry operation to the atomic positions X: X' = op * X
            augmented_crystal.X = (op @ self.X.T).T  # Apply the symmetry operation to the atomic positions X

            # Append the augmented crystal to the list
            augmented_crystals.append(augmented_crystal)

        return augmented_crystals  # Return the list of augmented Crystal objects

    def translate_crystal(self, coefficients): 
        """
        Move a crystal structure through 3D space by a translation vector that is a multiple of the lattice vectors.

        Parameters:
        - coefficients: Tensor of shape (3,) containing integer coefficients for the lattice translation.

        Returns:
        - A_translated: Same as A.
        - X_translated: Tensor of shape (n, 3) with atomic coordinates shifted by a lattice-based translation vector.
        - L_translated: Same as L.
        """
        try:
            # Ensure coefficients is a tensor of integers
            if not isinstance(coefficients, torch.Tensor):
                coefficients = torch.tensor(coefficients, dtype=torch.int32)
            if len(coefficients) != 3 or torch.is_floating_point(coefficients):
                raise ValueError("Translation coefficients must be a 3-element integer tensor")
        except AttributeError:
            raise ValueError("Invalid Input")
        
        # Compute translation vector as an integer combination of lattice vectors
        translation_vector = torch.matmul(coefficients.float(), self.L)

        # Apply translation
        X_translated = self.X + translation_vector

        return self.A, X_translated, self.L
    
    def get_symmetry_operations(self):
        # Convert torch tensors to numpy
        lattice = self.L.numpy()
        positions = self.X.numpy() @ np.linalg.inv(lattice)  # fractional coords
        try:
            numbers = self.A[:, 0].int().numpy()  # Assumes first column of A is atomic number
        except:
            raise ValueError("Cannot extract atomic numbers from self.A. Ensure atomic number is in first column.")

        # Create crystal structure tuple
        cell = (lattice, positions, numbers)

        # Get symmetry dataset
        dataset = spglib.get_symmetry_dataset(cell)

        # Get rotation matrices and convert to torch tensors
        raw_rotations = dataset['rotations']
        identity = torch.eye(3)

        # Filter out identity matrix
        rotation_matrices = [
            torch.tensor(R, dtype=torch.float32)
            for R in raw_rotations
            if not torch.allclose(torch.tensor(R, dtype=torch.float32), identity)
        ]

        return rotation_matrices