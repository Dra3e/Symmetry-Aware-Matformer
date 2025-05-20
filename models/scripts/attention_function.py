import math
import torch
import torch.nn as nn

class AttentionFunction(nn.Module):
    """Multi-headed attention function for edge-based attention calculation."""
    def __init__(self, hidden_dim, head_size):
        """
        Initializes the AttentionFunction module.

        Args:
            hidden_dim (int): Dimensionality of node and edge input features.
            head_size (int): Number of attention heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = head_size
        self.feature_dim = hidden_dim * 3  # For value: value_i + value_j + edge

        # Linear projections for each input, packed across all heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key_i = nn.Linear(hidden_dim, hidden_dim)
        self.key_j = nn.Linear(hidden_dim, hidden_dim)
        self.value_i = nn.Linear(hidden_dim, hidden_dim)
        self.value_j = nn.Linear(hidden_dim, hidden_dim)
        self.edge    = nn.Linear(hidden_dim, hidden_dim)

        # Normalization applied before sigmoid attention activation
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        # Linear transform for edge symmetry contribution (no bias)
        self.symmetry = nn.Linear(1, hidden_dim, bias=False)

    def split_heads(self, x):
        """
        Splits the input tensor into multiple heads.

        Args:
            x (torch.Tensor): Input tensor of shape (N, D).

        Returns:
            torch.Tensor: Tensor of shape (H, N, D/H), where H is the number of heads.
        """
        N = x.size(0)
        head_dim = self.hidden_dim // self.num_heads
        x = x.view(N, self.num_heads, head_dim)  # (N, H, D/H)
        return x.permute(1, 0, 2)  # (H, N, D/H)

    def forward(self, edge, gradient):
        """
        Computes attention weights and value matrix for a batch of edges.

        Args:
            edge (tuple): Tuple of 5 tensors (each [N, D]) representing:
                edge[0]: f_i — feature of target node
                edge[1]: f_j — feature of source node
                edge[2]: e_ij — edge features
                edge[3]: e_sym — symmetry info (0 = physical, 1 = augmented)
                edge[4]: e_mask — validity (1 = valid, 0 = invalid)

            gradient (float): Controls how much attention to give to invalid edges (higher = less).

        Returns:
            alpha (Tensor): Attention weights, shape (H, N, 3D/H)
            value_C (Tensor): Value matrix, shape (H, N, 3D/H)
        """
        N = edge.size(0)
        head_dim = self.hidden_dim // self.num_heads
        feature_dim_per_head = 3 * head_dim

        f_i     = edge[:, 0, :]  # (N, D)
        f_j     = edge[:, 1, :]  # (N, D)
        e_ij    = edge[:, 2, :]  # (N, D)
        m_sym   = edge[:, 3, 0]  # (N,)
        m_valid = edge[:, 4, 0]  # (N,)

        # Apply linear projections and reshape for each head
        query      = self.split_heads(self.query(f_i))       # (H, N, D/H)
        key_i      = self.split_heads(self.key_i(f_i))
        key_j      = self.split_heads(self.key_j(f_j))
        value_i    = self.split_heads(self.value_i(f_i))
        value_j    = self.split_heads(self.value_j(f_j))
        edge_trans = self.split_heads(self.edge(e_ij))

        # Concatenate features along the last dimension
        query_C = torch.cat((query, query, query), dim=-1)          # (H, N, 3D/H)
        key_C   = torch.cat((key_i, key_j, edge_trans), dim=-1)     # (H, N, 3D/H)
        value_C = torch.cat((value_i, value_j, edge_trans), dim=-1) # (H, N, 3D/H)

        # Raw attention score (element-wise product)
        d = math.sqrt(3 * (self.hidden_dim // self.num_heads))
        alpha_raw = query_C * key_C / d                             # (H, N, 3D/H)

        # LayerNorm + sigmoid
        # layerNorm always normalises the last dimension, so permute(1,0,2) swaps H and N dimension.
        # In order to apply layerNorm to all heads without looping, reshape(N,-1) collects all heads into one tensor
        alpha = alpha_raw.permute(1, 0, 2).reshape(N, -1)           # (N, 3D)
        alpha = torch.sigmoid(self.layer_norm(alpha))               # (N, 3D)
        alpha = alpha.view(N, self.num_heads, feature_dim_per_head).permute(1, 0, 2)  # (H, N, 3D/H)

        # Symmetry projection: (N,) → (N, 1) → (H, N, D/H)
        m_sym = m_sym.unsqueeze(-1)  # (N, 1)
        sym_proj = self.split_heads(self.symmetry(m_sym))           # (H, N, D/H)
        symmetry = 1 + sym_proj * edge_trans                        # (H, N, D/H)
        symmetry = symmetry.repeat(1, 1, 3)                         # (H, N, 3D/H)

        # Validity modifier
        m_valid = torch.sigmoid(gradient * (m_valid - 0.5))         # (N,)
        m_valid = m_valid.unsqueeze(0).unsqueeze(-1)                # (1, N, 1)
        m_valid = m_valid.repeat(1, 1, feature_dim_per_head)        # (1, N, 3D/H)

        # Final attention weights
        alpha = alpha * symmetry * m_valid                          # (H, N, 3D/H)

        return alpha, value_C  # both (H, N, 3D/H)
    
    def to(self, device):
        """
        Move the AttentionFunction instance to the specified device.

        Args:
            device (torch.device): The device to move the instance to.
        """
        self.query = self.query.to(device)
        self.key_i = self.key_i.to(device)
        self.key_j = self.key_j.to(device)
        self.value_i = self.value_i.to(device)
        self.value_j = self.value_j.to(device)
        self.edge = self.edge.to(device)
        self.layer_norm = self.layer_norm.to(device)
        self.symmetry = self.symmetry.to(device)
        return self