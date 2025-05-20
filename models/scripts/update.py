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
        # --- Linear projections from raw space to model space
        self.atom_proj = nn.Linear(input_dim,  hidden_dim, bias=False)  # W_atom
        self.msg_proj  = nn.Linear(input_dim,  hidden_dim, bias=False)  # W_msg

        self.beta = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # --- Pre-norm & position-wise feed-forward network (Model space only)
        self.pre_norm  = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

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
        # 1) Project both tensors to model space (hidden_dim)
        f = self.atom_proj(atom_features)           # (B, N, hidden_dim)
        m = self.msg_proj(aggregated_input)         # (B, N, hidden_dim)

        # 2) Residual message add  (Matformer keeps the base features alive)
        gate = torch.sigmoid(self.beta(torch.cat([f, m], dim=-1)))
        f = f * gate + m * (1 - gate)


        # 3) Pre‑norm + feed‑forward + residual
        f = f + self.ffn(self.pre_norm(f))

        # 4) Optional final norm (Matformer uses it)
        return self.post_norm(f)
    
    def to(self, device):
        """
        Move the Update instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.
        """
        self.beta      = self.beta.to(device)
        self.atom_proj = self.atom_proj.to(device)
        self.msg_proj  = self.msg_proj.to(device)
        self.pre_norm = self.pre_norm.to(device)
        self.ffn = self.ffn.to(device)
        self.post_norm = self.post_norm.to(device)
        return self