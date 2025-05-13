import torch
import spglib
import tqdm
import os
import contextlib

# Context manager to suppress low-level internal spglib warnings
@contextlib.contextmanager
def suppress_spglib_warnings():
    """
    Suppresses both stdout and stderr at the OS level to hide internal warnings
    from spglib, including those printed from compiled C code.
    """
    with open(os.devnull, 'w') as devnull:
        # Save original file descriptors for stdout (1) and stderr (2)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            # Redirect stdout and stderr to /dev/null (suppresses console output)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            # Restore original stdout and stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

class InvalidAugmenter:
    '''Script class for getting all unique symmetry operations for the crystals and a method that returns all symmetry operations that are not valid'''
    def __init__(self):
        """
        Initializes the InvalidAugmenter class by loading all unique symmetry operations 
        from the spglib database and storing them as PyTorch tensors."""
        all_rotations = set()
        for i in tqdm.tqdm(range(1, 231), desc="Processing Symmetry Datasets", unit="hall marks"): # Loops through all of the hall numbers, and creates a loading bar in the terminal
            with suppress_spglib_warnings(): # Suppress spglib warnings for unsupported or broken hall numbers
                dataset = spglib.get_symmetry_from_database(hall_number=i) # Get symmetry operations from the database

            if dataset is None:
                continue
            for R in dataset['rotations']:
                all_rotations.add(tuple(R.flatten())) # Stores all unique rotations in a set to avoid duplicates

        # Convert the set of unique rotations to PyTorch tensors
        all_valid = [torch.tensor(r, dtype=torch.float32).reshape(3, 3) for r in all_rotations] # Converts all unique rotations to PyTorch tensors
        self.all_valid_tensor = torch.stack(all_valid)
    
    
    def forward(self, crystals_valid_augmentation_matrices):
        '''
        Finds all symmetry operations that are not valid for the crystal.

        Args:
            crystals_valid_augmentation_matrices (list): List of valid augmentation matrices from the crystal.

        Returns:
            torch.Tensor: Tensor of shape [K, 3, 3], where K is the number of unique symmetry
                          matrices from the spglib database that are not in the input list.
        '''
        if not crystals_valid_augmentation_matrices:
            return self.all_valid_tensor.clone()
        all_valid_tensor = self.all_valid_tensor.clone()

        # Stack the input crystal's valid symmetries
        crystal_valid_tensor = torch.stack(crystals_valid_augmentation_matrices)
        
        # Expand dimensions for broadcasting comparison
        # crystals_valid_augmentation_matrices shape: [M, 3, 3]
        # Expand to [1, M, 3, 3] for comparison with all_valid_tensor [N, 1, 3, 3]
        valid_mats = crystal_valid_tensor.unsqueeze(0)  # [1, M, 3, 3]
        all_rots = all_valid_tensor.unsqueeze(1)  # [N, 1, 3, 3]
        
        # Compare all rotations with all valid matrices
        # Element-wise comparison with tolerance
        close = torch.isclose(all_rots, valid_mats, atol=1e-5) # [N, M, 3, 3] — True if elements are within atol
        # Check if all elements in the 3x3 matrix are close
        matrix_close = torch.all(close, dim=(-2, -1))  # [N, M]
        # Check if any of the valid matrices matches each rotation
        is_valid = torch.any(matrix_close, dim=1)  # [N]
        
        # Indices where no match was found — these are "invalid" for the crystal
        invalid_indices = torch.where(~is_valid)[0]
        
        return all_valid_tensor[invalid_indices]
    
    def to(self, device):
        """
        Move the InvalidAugmenter instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.
        """
        self.all_valid_tensor = self.all_valid_tensor.to(device)
        return self