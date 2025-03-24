import torch
import torch.nn as nn

def predicted_property(pooled_features, hidden_dim):
    """
    Compute the predicted property using two linear transformations and the SiLU activation.

    Parameters:
    - pooled_features: Input features (n x d)
    - hidden_dim: Hidden dimension for the transformations

    Returns:
    - y_property: Output tensor (n x d)
    """

    # Define transformations
    LN1 = nn.Linear(pooled_features.shape[1], hidden_dim)  # First Linear Transform
    LN2 = nn.Linear(hidden_dim, hidden_dim)  # Second Linear Transform

    # Compute LN(F_out)
    transformed = LN1(pooled_features)

    # Apply SiLU activation
    activated = torch.nn.functional.silu(transformed)

    # Apply second Linear Transformation
    y_property = LN2(activated)

    return y_property