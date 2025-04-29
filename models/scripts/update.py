import torch
import torch.nn as nn

class Update(nn.Module):
    """ Script class for the update step of the transformer model. """
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the Update class.
        
        Args:
            input_dim (int): Dimension of the input features
            hidden_dim (int): Dimension of the hidden features
        """
        super(Update, self).__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=False)  # Linear transform without bias for weight
        self.weight_projection = nn.Linear(input_dim, hidden_dim, bias=False)  # Projection for aggregated_input
        self.LN = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, aggregated_input, atom_features):
        """
        Perform the update operation using a linear transformation and layer normalization.
        Reintroduces weighted atom features to ensure they are not forgotten after finding atom strength.
            f*_i = LNorm(LinearTransformation(atom_strenght + Weight*f_i))

        Parameters:
            aggregated_input (torch.tensor): An ordered list of length n with the strenght/effect of the atoms in the crystal.
            atom_features (torch.tensor): An ordered matrix of size n x 120, where the placement of each row matches the strength/effect of the atom in the aggregated_input list.

        Returns:
            updated_features (torch.tensor): An updated list of atom features.
        """
        # Project aggregated_input to match the feature dimension of atom_features
        aggregated_input = self.weight_projection(aggregated_input)
        
        ln_transformed = self.LN(aggregated_input + self.weight(atom_features))
        updated_features = self.layer_norm(ln_transformed)

        return updated_features
    
    def to(self, device):
        """
        Move the Update instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.
        """
        self.weight = self.weight.to(device)
        self.weight_projection = self.weight_projection.to(device)
        self.LN = self.LN.to(device)
        self.layer_norm = self.layer_norm.to(device)
        return self