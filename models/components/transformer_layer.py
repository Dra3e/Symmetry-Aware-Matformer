import torch
import torch.nn as nn
from torch_scatter import scatter
from models.scripts.attention_function import AttentionFunction

class TransformerLayer(nn.Module):
    """ A multi-headed transformer layer """
    def __init__(self, hidden_dim, head_size):
        """
        Initializes TransformerLayer instance.

        Args:
            hidden_dim (int): Dimension of node features.
            head_size (int): Number of attention heads.
        """
        super(TransformerLayer, self).__init__()
        self.head_size = head_size
        self.feature_dim = hidden_dim * 3  # From attention: value_i + value_j + edge

        self.attention_function = AttentionFunction(hidden_dim, head_size)

        # Per-head linear weights and bias: (H, D, D)
        self.linear_weight = nn.Parameter(torch.randn(head_size, self.feature_dim, self.feature_dim))
        self.linear_bias = nn.Parameter(torch.zeros(head_size, self.feature_dim))

    def forward(self, node_chunk, edge_chunk, gradient):
        """
        Calculates the influence of neighboring atoms using attention.

        Args:
            node_chunk (torch.Tensor): [M, hidden_dim] Node features for a chunk of M nodes.
            edge_chunk (tuple): Tuple of edge data with:
                edge_chunk[0] (f_i): [N, D] features of destination nodes
                edge_chunk[1] (f_j): [N, D] features of source nodes
                edge_chunk[2] (e_ij): [N, D] edge feature vector
                edge_chunk[3] (e_sym): [N, D] symmetry info
                edge_chunk[4] (e_mask): [N, D] edge validity
                edge_chunk[5] (target_index): [N] index of the destination node in `node_chunk`
            gradient (float or tensor): Determines how much invalid edges affect attention.

        Returns:
            torch.Tensor: [M, feature_dim] The aggregated message for each node in node_chunk.
        """
        f_i, f_j, e_ij, e_sym, e_mask, target_index = edge_chunk

        alpha, value = self.attention_function((f_i, f_j, e_ij, e_sym, e_mask), gradient)  # (H, N, D)

        # Permute to (N, H, D) so we can batch-multiply per edge (N) with per-head weights (H, D, D)
        # value: (H, N, D) → (N, H, D)
        value = value.permute(1, 0, 2)

        # Linear transformation per head
        transformed = torch.einsum("nhd,hdf->nhf", value, self.linear_weight) + self.linear_bias.unsqueeze(1)
        transformed = transformed.permute(1, 0, 2)  # (H, N, D)

        # Apply attention weights
        weighted = alpha * transformed  # (H, N, D)

        # Sum across heads → (N, D)
        m_i_per_edge = weighted.sum(dim=0)

        # Aggregate per node using target_index → (M, D)
        m_i_chunk = scatter(m_i_per_edge, target_index, dim=0, dim_size=node_chunk.size(0), reduce='sum')

        return m_i_chunk