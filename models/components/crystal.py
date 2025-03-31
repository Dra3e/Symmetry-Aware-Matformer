import torch
import spglib
import numpy as np

class Crystal:
    """
    Representation of a crystal structure A, X, L.
        A is a 2D matrix (torch.tensor) of dimension n x f, where n is the number of atoms in the crystal and f is the number of features we know about that atom, such as it magnetism, conductivity, atom number, and so on.
            Each row of this vector a_i corresponds to an atom in the crystal and each member a_ij is a feature of that atom.
        X is a 2D matrix as well of dimension n x 3, where each row x_i is the x,y,z coordinates of the atom a_i in a 3D space.
            We know which atom corresponds to which set of coordinates by their placement i, such that the i'th row in the X matrix is the coordinates for the i'th atom in the A matrix.
        L is a 2D matrix of dimenstion 3 x 3, which represents the crystal lattice.
    """
    def __init__(self, A, X, L):
        """
        Initializes a Crystal structure.

        Args:
            - A: Tensor of shape (n, f) containing atomic features (remains unchanged).
            - X: Tensor of shape (n, 3) containing atomic coordinates.
            - L: Tensor of shape (3, 3) representing the lattice vectors.            

        """
        self.A = A.clone().detach() if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
        self.X = X.clone().detach() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.L = L.clone().detach() if isinstance(L, torch.Tensor) else torch.tensor(L, dtype=torch.float32)

    def apply_augmentations(self, symmetry_operations):
        """
        Applies symmetry operations to a crystal structure.

        Args:
            symmetry_operations: List of 3x3 rotation matrices representing symmetry operations.

        Returns:
            augmented_crystals: List containing transformed Crystal instances, each with updated atomic positions.
        """
        augmented_crystals = []

        for op in symmetry_operations:
            augmented_crystal = Crystal(self.A, self.X, self.L) # New Crystal strucutre with the same atomic positions and lattice

            augmented_crystal.X = (op @ self.X.T).T  # Apply the symmetry operation to the atomic position X: X' = op * X
            augmented_crystals.append(augmented_crystal)

        return augmented_crystals  

    def translate_crystal(self, coefficients): 
        """
        Move a crystal structure through 3D space by a translation vector that is a multiple of the lattice vectors.

        Args:
        - coefficients: Tensor of shape (1, 3) containing integer coefficients for the lattice translation.

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
        X_translated = self.X + translation_vector # Apply translation

        return self.A, X_translated, self.L
    
    def get_symmetry_operations(self):
        """
        Find the symmetry operations that maintain the crystal structure, as per E(3) and periodic invariance.
        Assumes first column of A is atomic number or other unique identifier per atom type.

        Returns:
            symmetry_operations: List of 3x3 rotation matrices representing symmetry operations.
        """
        # Convert torch tensors to numpy
        lattice = self.L.numpy()
        positions = self.X.numpy() @ np.linalg.inv(lattice)  # fractional coords
        try:
            numbers = self.A[:, 0].int().numpy()  # Assumes first column of A is atomic number
        except:
            raise ValueError("Cannot extract atomic numbers from self.A. Ensure atomic number is in first column.")

        cell = (lattice, positions, numbers) # Create crystal structure tuple

        dataset = spglib.get_symmetry_dataset(cell) # Get symmetry dataset
 
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