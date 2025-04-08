import numpy as np
import spglib
import torch
import tqdm

class InvalidAugmenter:
    def __init__(self):
        all_rotations = set()
        for i in tqdm.tqdm(range(1, 11), desc="Processing Datasets", unit="dataset"):
            dataset = spglib.get_symmetry_from_database(hall_number=i)

            if dataset is None:
                continue
            for R in dataset['rotations']:
                all_rotations.add(tuple(R.flatten()))

        # Store all unique rotations as PyTorch tensors
        self.all_valid = [torch.tensor(r, dtype=torch.float32).reshape(3, 3) for r in all_rotations]
    
    def forward(self, crystals_valid_augmentation_matrices):
        # Setup the valid tensors
        all_valid_tensor = torch.stack(self.all_valid)
        crystal_valid_tensor = torch.stack(crystals_valid_augmentation_matrices)
        
        # Expand dimensions for broadcasting comparison
        # crystals_valid_augmentation_matrices shape: [M, 3, 3]
        # Expand to [1, M, 3, 3] for comparison with all_valid_tensor [N, 1, 3, 3]
        valid_mats = crystal_valid_tensor.unsqueeze(0)  # [1, M, 3, 3]
        all_rots = all_valid_tensor.unsqueeze(1)  # [N, 1, 3, 3]
        
        # Compare all rotations with all valid matrices
        # Element-wise comparison with tolerance
        close = torch.isclose(all_rots, valid_mats, atol=1e-5)
        # Check if all elements in the 3x3 matrix are close
        matrix_close = torch.all(close, dim=(-2, -1))  # [N, M]
        
        # Check if any of the valid matrices matches each rotation
        is_valid = torch.any(matrix_close, dim=1)  # [N]
        
        # Get invalid rotations (where is_valid is False)
        invalid_indices = torch.where(~is_valid)[0]
        
        return all_valid_tensor[invalid_indices]
        