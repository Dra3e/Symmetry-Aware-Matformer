import torch
import spglib
import numpy as np
import os
import contextlib

# Context manager to suppress stderr (used to silence spglib internal warnings)
@contextlib.contextmanager
def suppress_spglib_warnings():
    """
    Suppresses C-level warnings printed to stdout/stderr by spglib without hiding real Python exceptions.
    """
    with open(os.devnull, 'w') as devnull:
        # Save the real stdout (1) and stderr (2) file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            # Redirect stdout and stderr to /dev/null
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            # Restore original file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

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
        Returns a list of proper 3×3 rotation matrices (torch.Tensor) – no translations,
        no extra dimensions, identity removed.
        """
        lattice   = self.L.cpu().numpy()
        positions = self.X.cpu().numpy() @ np.linalg.inv(lattice)      # fractional
        numbers   = self.A[:, 0].int().cpu().numpy()                   # Z of each atom
        cell      = (lattice, positions, numbers)

        with suppress_spglib_warnings():
            dataset = spglib.get_symmetry_dataset(cell)

        # (N, 3, 3) → torch tensor on same device / dtype as L
        R_all = torch.as_tensor(dataset.rotations,
                                dtype=self.L.dtype, device=self.L.device)          # (N,3,3)

        # drop the identity (within a small tolerance)
        eye   = torch.eye(3, dtype=self.L.dtype, device=self.L.device)
        mask  = (R_all - eye).abs().sum(dim=(1, 2)) > 1e-6                         # (N,)
        R_all = R_all[mask]                                                        # (K,3,3)

        # return a *list* so `for op in sym_ops:` yields a 3×3 tensor each time
        return [R for R in R_all]
    
    def get_translation_vectors_to_neighbour_cells(self, cutoff):
        max_n = int(np.ceil(cutoff / min(np.linalg.norm(self.L, axis=1))))  # worst‑case
        shifts = torch.tensor(
            [[i, j, k]  for i in range(-max_n, max_n + 1)
                        for j in range(-max_n, max_n + 1)
                        for k in range(-max_n, max_n + 1)
                        if (i, j, k) != (0, 0, 0)],
            dtype=self.L.dtype,
            device=self.L.device
        )                                 # (M,3)  of integer offsets
        return shifts
    
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
    
    def to(self, device):
        """
        Move the crystal structure to the specified device (CPU or GPU).

        Args:
            device: Device to move the crystal structure to (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        self.A = self.A.to(device)
        self.X = self.X.to(device)
        self.L = self.L.to(device)
        return self