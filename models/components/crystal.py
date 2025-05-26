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
        # Save the current output streams (stdout and stderr)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            # Redirect stdout and stderr to a "black hole" (no output)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield # Allow the code inside the context manager to run
        finally:
            # Restore original file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd) # Close the saved file descriptors
            os.close(old_stderr_fd)

class Crystal:
    """
    Representation of a crystal structure A, P, L.

    Attributes:
        A (torch.Tensor): (n, f) matrix where n is the number of atoms and f is the number of features per atom.
        P (torch.Tensor): (n, 3) matrix where each row represents the x, y, z coordinates of an atom.
        L (torch.Tensor): (3, 3) matrix representing the lattice vectors of the crystal.
    """
    def __init__(self, A, P, L):
        """
        Initializes a Crystal structure.

        Args:
            A (torch.Tensor): Atomic features of shape (n, f).
            P (torch.Tensor): Atomic coordinates of shape (n, 3).
            L (torch.Tensor): Lattice vectors of shape (3, 3).         
        """
        self.A = A.clone().detach() if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
        self.P = P.clone().detach() if isinstance(P, torch.Tensor) else torch.tensor(P, dtype=torch.float32)
        self.L = L.clone().detach() if isinstance(L, torch.Tensor) else torch.tensor(L, dtype=torch.float32)
    
    def get_symmetry_operations(self):
        """
        Retrieves a list of (3, 3) matrices from spglib as valid symmetry operations to perform on the crystal strucutre.

        Returns:
            symmetry_operations (List[torch.Tensor]): List of (3, 3) matrices excluding the identity matrix.
        """
        # Convert lattice, positions, and atomic numbers to numpy arrays
        lattice   = self.L.cpu().numpy()
        positions = self.P.cpu().numpy() @ np.linalg.inv(lattice)      # Converts the atom positions from cartesian coordinates to fractional coordinates
        numbers   = self.A[:, 0].int().cpu().numpy()                   # First column of the atom feature matrix, used to identify unique atoms, e.g. atomic number.
        cell      = (lattice, positions, numbers)

        with suppress_spglib_warnings():
            dataset = spglib.get_symmetry_dataset(cell) # Use spglib to get symmetry dataset.

        # Extract rotation matrices as symmetry operation tensors.
        symmetry_operations = torch.as_tensor(dataset.rotations,dtype=self.L.dtype, device=self.L.device)
        eye   = torch.eye(3, dtype=self.L.dtype, device=self.L.device)      # Create the identity matrix of the crystal structure.
        mask  = (symmetry_operations - eye).abs().sum(dim=(1, 2)) > 1e-6    # Find all symmetry operations that are not the identity matrix.
        symmetry_operations = symmetry_operations[mask]                     # Remove the identity matrix from symmetry operations.

        return [op for op in symmetry_operations] # Convert to list.
    
    def get_translation_vectors_to_neighbour_cells(self, cutoff):
        """
        Computes the translation vectors to neighboring unit cells within a specified cutoff distance.

        Args:
            cutoff (float): The maximum distance to consider for neighboring unit cells.

        Returns:
            translation_shifts (torch.Tensor): A tensor where each row is a translation vector to a neighboring unit cell.
        """
        # Calculate the maximum number of neighboring cells needed in each direction to cover the cutoff distance.
        max_n = int(np.ceil(cutoff / min(np.linalg.norm(self.L, axis=1))))
        shifts = torch.tensor( # Translation vectors to the neighboring unit cells within the cutoff distance.
            [[i, j, k]  for i in range(-max_n, max_n + 1) # Create neighbours along the x axis.
                        for j in range(-max_n, max_n + 1) # Create neighbours along the y axis.
                        for k in range(-max_n, max_n + 1) # Create neighbours along the z axis.
                        if (i, j, k) != (0, 0, 0)],
            dtype=self.L.dtype,
            device=self.L.device
        )     
        return shifts
    
    def apply_augmentations(self, symmetry_operations):
        """
        Computes a list of augmented versions of the crystal based on a list of symmetry operation matrices.

        Args:
            symmetry_operations (List[torch.Tensor]): list of (3, 3) matrices to apply to the crystal structure.

        Returns:
            augmented_crystals (List[torch.Tensor]): list of crystal augmented structures.
        """
        augmented_crystals = []

        # Apply each of the symmetry operations to the atomic position P: P' = op * P
        for op in symmetry_operations:
            augmented_crystal = Crystal(self.A, self.P, self.L)
            augmented_crystal.P = (op @ self.P.T).T 
            augmented_crystals.append(augmented_crystal)

        return augmented_crystals
    
    def to(self, device):
        """
        Moves the crystal structure to the specified device (CPU or GPU).

        Args:
            device: The target device (e.g., 'cpu' or 'cuda').

        Returns:
            Crystal: The crystal structure moved to the specified device.
        """
        # Move atomic features, positions, and lattice vectors to the target device
        self.A = self.A.to(device)
        self.P = self.P.to(device)
        self.L = self.L.to(device)
        return self