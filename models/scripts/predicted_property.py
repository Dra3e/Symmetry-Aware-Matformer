import torch
import torch.nn as nn

class PredictedProperty(nn.Module):
    """ 
    Script class for the prediction step of the transformer model. 
    This class is used to predict properties of the input data using a two-layer feedforward neural network.
    The first layer applies a linear transformation followed by the SiLU activation function,
    and the second layer applies another linear transformation.
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the PredictedProperty class.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden and outputs layer.
        """
        super(PredictedProperty, self).__init__()
        self.LN1 = nn.Linear(input_dim, hidden_dim)  # First Linear Transform
        self.LN2 = nn.Linear(hidden_dim, hidden_dim)  # Second Linear Transform

    def forward(self, pooled_features):
        """
        Compute the predicted property using two linear transformations and the SiLU activation on the crystal features.
            y_property = LinearTransform(SiLu(LinearTransform(F_crystal)))

        Parameters:
            pooled_features (torch.Tensor): The pooled features of the entire crystal.

        Returns:
            y_property: The value of the predicted property of the crystal.
        """
        transformed = self.LN1(pooled_features) # Compute LN(F_out) 
        activated = torch.nn.functional.silu(transformed) # Apply SiLU activation
        y_property = self.LN2(activated) # Apply second Linear Transformation
        return y_property
    
    def to(self, device):
        """
        Move the PredictedProperty instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.
        """
        self.LN1 = self.LN1.to(device)
        self.LN2 = self.LN2.to(device)
        return self