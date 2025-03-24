import torch
import torch.nn as nn

class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()

    def forward(self, updated_features):
        """
        Perform global pooling by summing the output of updated crystals.

        Parameters:
        - updated_crystals: List of updated crystal features (list of tensors)

        Returns:
        - Pooled features (tensor)
        """
        # Use torch.sum to sum the output of updated crystals along the first dimension (rows)
        pooled_features = torch.sum(torch.stack(updated_features), dim=0)

        return pooled_features