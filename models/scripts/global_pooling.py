import torch
import torch.nn as nn

class GlobalPooling(nn.Module):
    """ Script class for performing """
    def __init__(self):
        """ Initializes an instance of the GlobalPooling class """
        super(GlobalPooling, self).__init__()

    def forward(self, updated_features):
        """
        Perform global pooling by summing the output of updated crystals.
            F_crystal = SUM_each_atom_in_crystal(f_atom)

        Args:
            updated_crystals: A list with length n of crystal atoms features.

        Returns:
            pooled_features: A scalar with the sum of the atoms of the crystal.
        """
        # Use torch.sum to sum the output of updated crystals along the first dimension (rows)
        pooled_features = torch.sum(updated_features, dim=0)

        return pooled_features
    
    def to(self, device):
        """
        Move the GlobalPooling instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.
        """
        return self