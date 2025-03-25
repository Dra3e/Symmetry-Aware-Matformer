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
    
    def batch_forward(self, updated_features):
        """
        Perform global pooling by summing the output of updated crystals.

        Parameters:
        - updated_crystals: List of updated crystal features (list of tensors) (batch_size, num_nodes, hidden_dim)

        Returns:
        - Pooled features (tensor) (batch_size, hidden_dim)
        """

        # updated_features: (batch_size, num_nodes, hidden_dim)
        pooled_features = updated_features.mean(dim=1)  # Shape: (batch_size, hidden_dim)

        return pooled_features