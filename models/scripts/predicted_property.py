import torch
import torch.nn as nn

class PredictedProperty(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PredictedProperty, self).__init__()
        self.LN1 = nn.Linear(input_dim, hidden_dim)  # First Linear Transform
        self.LN2 = nn.Linear(hidden_dim, hidden_dim)  # Second Linear Transform

    def forward(self, pooled_features):
        """
        Compute the predicted property using two linear transformations and the SiLU activation.

        Parameters:
        - pooled_features: Input features (n x d)

        Returns:
        - y_property: Output tensor (n x d)
        """
        # Compute LN(F_out)
        transformed = self.LN1(pooled_features)

        # Apply SiLU activation
        activated = torch.nn.functional.silu(transformed)

        # Apply second Linear Transformation
        y_property = self.LN2(activated)

        return y_property
    
    def batch_forward(self, pooled_features):
        """
        Compute the predicted property using two linear transformations and the SiLU activation.

        Parameters:
        - pooled_features: Input features (batch_size, hidden_dim)

        Returns:
        - y_property: Output tensor (batch_size, hidden_dim)
        """
        # Compute LN(F_out)
        transformed = self.LN1(pooled_features) # Shape: (batch_size, hidden_dim)

        # Apply SiLU activation
        activated = torch.nn.functional.silu(transformed) # Shape: (batch_size, hidden_dim)

        # Apply second Linear Transformation
        y_property = self.LN2(activated) # Shape: (batch_size, hidden_dim)

        return y_property