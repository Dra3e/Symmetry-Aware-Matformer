import torch
import torch.nn as nn
import torch.nn.functional as F

class Update(nn.Module):
    """ Script class for the update step of the transformer model. """
    def __init__(self, input_dim, hidden_dim, use_beta: bool = True):
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

        # --- β-gate (scalar per node)
        self.beta_norm = nn.LayerNorm(hidden_dim)
        self.use_beta  = use_beta
        if self.use_beta:
            self.lin_beta = nn.Linear(hidden_dim * 3, 1, bias=False)    # W_β
        else:
            self.register_parameter("lin_beta", None)

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
        m = F.silu(self.beta_norm(m))               # LayerNorm + SiLU shrinks the variance of the tensor, so lin_beta never sees huge values.

        # 2) beta-gated residual merge   out <- beta * f + (1 - beta) * m
        if self.use_beta and self.lin_beta is not None:
            gate_in = torch.cat([m, f, m - f], dim=-1)   # (B, N, 3·H)
            beta    = torch.sigmoid(self.lin_beta(gate_in))  # (B, N, 1)
            f = beta * f + (1.0 - beta) * m
        else:
            f = f + m


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
        self.beta_norm = self.beta_norm.to(device)
        self.lin_beta = self.lin_beta.to(device)
        self.atom_proj = self.atom_proj.to(device)
        self.msg_proj  = self.msg_proj.to(device)
        self.pre_norm = self.pre_norm.to(device)
        self.ffn = self.ffn.to(device)
        self.post_norm = self.post_norm.to(device)
        return self