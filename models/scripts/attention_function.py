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
        self.query = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.key_i = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.key_j = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.value_i = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.value_j = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.edge    = nn.Linear(hidden_dim, head_size * hidden_dim)

        # Normalization applied before sigmoid attention activation
        self.layer_norm = nn.LayerNorm(self.feature_dim * head_size)

        # Linear transform for edge symmetry contribution (no bias)
        self.symmetry = nn.Linear(hidden_dim, head_size * hidden_dim, bias=False)

    def split_heads(self, x):
        # Reshape input from (N, H*D) → (H, N, D) for multi-head parallelism
        N = x.size(0)
        return x.view(N, self.num_heads, self.hidden_dim).permute(1, 0, 2)

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
            alpha (Tensor): Attention weights, shape (H, N, 3D)
            value_C (Tensor): Value matrix, shape (H, N, 3D)
        """
        N = edge[0].size(0)

        # Apply linear projections and reshape for each head
        query     = self.split_heads(self.query(edge[0]))      # (H, N, D)
        key_i     = self.split_heads(self.key_i(edge[0]))
        key_j     = self.split_heads(self.key_j(edge[1]))
        value_i   = self.split_heads(self.value_i(edge[0]))
        value_j   = self.split_heads(self.value_j(edge[1]))
        edge_trans= self.split_heads(self.edge(edge[2]))

        # Concatenate inputs for attention and value computation
        query_C = torch.cat((query, query, query), dim=-1)          # (H, N, 3D)
        key_C   = torch.cat((key_i, key_j, edge_trans), dim=-1)     # (H, N, 3D)
        value_C = torch.cat((value_i, value_j, edge_trans), dim=-1) # (H, N, 3D)

        # Raw attention score (dot product scaled by dim)
        d = key_C.size(-1)
        alpha_raw = query_C * key_C / d                             # (H, N, 3D)

        # LayerNorm + sigmoid activation on raw attention scores
        alpha = torch.sigmoid(self.layer_norm(alpha_raw.permute(1, 0, 2).reshape(N, -1)))
        alpha = alpha.view(N, self.num_heads, self.feature_dim).permute(1, 0, 2)  # (H, N, 3D)

        # Symmetry modifier: 1 + (W_sym * e_sym) * edge
        sym_proj = self.split_heads(self.symmetry(edge[3]))        # (H, N, D)
        symmetry = 1 + sym_proj * edge_trans                       # (H, N, D)

        # Validity modifier: sigmoid(gradient * (mask - 0.5))
        m_valid = torch.sigmoid(torch.tensor(gradient) * (edge[4] - 0.5)).view(1, N, 1)

        # Final attention weights: combine base alpha with modifiers
        alpha = alpha * symmetry * m_valid  # (H, N, 3D)

        return alpha, value_C