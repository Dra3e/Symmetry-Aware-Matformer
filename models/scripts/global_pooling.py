import torch
import torch.nn as nn

class GlobalPooling(nn.Module):
    """
    Performs global pooling by summing the features of atoms in a crystal.
    """
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
        pooled_features = torch.sum(updated_features, dim=0) # Use torch.sum to sum the output of updated crystals along the first dimension (rows)

        return pooled_features