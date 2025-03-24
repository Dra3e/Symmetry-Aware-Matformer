import torch
import torch.nn as nn

class Update(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Update, self).__init__()
        self.w_V = nn.Linear(input_dim, hidden_dim, bias=False)  # Linear transform without bias
        self.LN = nn.Linear(hidden_dim, hidden_dim)  # Linear transformation
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer normalization

    def forward(self, aggregated_input, atom_features):
        """
        Perform the update operation using a linear transformation and layer normalization.

        Parameters:
        - aggregated_input: Aggregated features (n x d)
        - atom_features: Original features (n x d)

        Returns:
        - Updated features (n x d)
        """
        # Compute w_V * atom_features
        transformed_features = self.w_V(atom_features)

        # Compute aggregated_input + transformed_features
        summed = aggregated_input + transformed_features

        # Apply linear transformation (LN)
        ln_transformed = self.LN(summed)

        # Apply layer normalization
        updated_features = self.layer_norm(ln_transformed)

        return updated_features