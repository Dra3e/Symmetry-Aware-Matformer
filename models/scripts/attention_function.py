import torch
import torch.nn as nn

class AttentionFunction(nn.Module):
    """ script class for attention calculation step in the transformer """
    def __init__(self, hidden_dim, head_size):
        """
        Initilizes AttentionFunction instance for calculating attentions of edges.

        Args:
            hidden_dim (int): The expected dimension of the input and the dimension of the output.
            head_size (int): How many heads to initialize.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = head_size
        self.feature_dim = hidden_dim * 3

        self.query = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.key_i = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.key_j = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.value_i = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.value_j = nn.Linear(hidden_dim, head_size * hidden_dim)
        self.edge = nn.Linear(hidden_dim, head_size * hidden_dim)

        self.layer_norm = nn.LayerNorm(self.feature_dim * head_size)
        self.symmetry = nn.Linear(hidden_dim, head_size * hidden_dim, bias=False)
    
    def split_heads(self, x):
        # x: (N, H*D) → (H, N, D)
        N = x.size(0)
        return x.view(N, self.num_heads, self.hidden_dim).permute(1, 0, 2)

    def forward(self, edge, gradient):
        """
        Calculates the attention of a list of edges.
            alpha_ij = Sigmoid(LinearTranform(Q*K/d)) + (1+Weight*M_sym*edge_ij) + Sigmoid(M_valid)

        Args:
            edge: A tensor of tuples with size 5 that represent edges that point away from nodes in node_chunk.
                edge_tuple[0]: A vector of size hidden_dim representing the node the edge points away from.
                edge_tuple[1]: A vector of size hidden_dim representing the node the edge points to.
                edge_tuple[2]: A vector of size hidden_dim representing the strength of the edge.
                edge_tuple[3]: M_sym: a scaler of value 0 or 1, 0 if physical, 1 if augmentation displacement.
                edge_tuple[4]: M_valid: a scaler of value 0 or 1, 1 if valid, 0 if invalid.
            gradient (float): A positive value of how much attention should be payed to invalid edges.
                If <10 then they weigh the same, if >20 then invalid edges are ignored.
        """
        # edge: tuple of 5 tensors, each [N, D]
        N = edge[0].size(0)

        # Project and split per head
        query = self.split_heads(self.query(edge[0]))  # (H, N, D)
        key_i = self.split_heads(self.key_i(edge[0]))
        key_j = self.split_heads(self.key_j(edge[1]))
        value_i = self.split_heads(self.value_i(edge[0]))
        value_j = self.split_heads(self.value_j(edge[1]))
        edge_trans = self.split_heads(self.edge(edge[2]))

        # Attention input prep
        query_C = torch.cat((query, query, query), dim=-1)         # (H, N, 3D)
        key_C   = torch.cat((key_i, key_j, edge_trans), dim=-1)     # (H, N, 3D)
        value_C = torch.cat((value_i, value_j, edge_trans), dim=-1) # (H, N, 3D)

        d = key_C.size(-1)
        alpha_raw = query_C * key_C / d  # (H, N, 3D)
        alpha = torch.sigmoid(self.layer_norm(alpha_raw.permute(1, 0, 2).reshape(N, -1)))
        alpha = alpha.view(N, self.num_heads, self.feature_dim).permute(1, 0, 2)  # (H, N, 3D)

        # Symmetry factor
        sym_proj = self.split_heads(self.symmetry(edge[3]))  # (H, N, D)
        symmetry = 1 + sym_proj * edge_trans  # (H, N, D)

        # Masking factor
        m_valid = torch.sigmoid(torch.tensor(gradient) * (edge[4] - 0.5)).view(1, N, 1)  # (1, N, 1)

        alpha = alpha * symmetry * m_valid  # (H, N, 3D)
        return alpha, value_C