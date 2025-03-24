import torch
import torch.nn as nn

def update(aggregated_input, atom_features, hidden_dim):
    """
    Perform the update operation using a linear transformation and layer normalization.

    Parameters:
    - aggregated_input: Aggregated features (n x d)
    - atom_features: Original features (n x d)
    - hidden_dim: Hidden dimension for transformation

    Returns:
    - Updated features (n x d)
    """

    # Define transformations
    w_V = nn.Linear(atom_features.shape[1], hidden_dim, bias=False)  # Linear transform without bias
    LN = nn.Linear(hidden_dim, hidden_dim)  # Linear transformation
    layer_norm = nn.LayerNorm(hidden_dim)  # Layer normalization

    # Compute w_V * atom_features
    transformed_features = w_V(atom_features)

    # Compute aggregated_input + transformed_features
    summed = aggregated_input + transformed_features

    # Apply linear transformation (LN)
    ln_transformed = LN(summed)

    # Apply layer normalization
    updated_features = layer_norm(ln_transformed)

    return updated_features
