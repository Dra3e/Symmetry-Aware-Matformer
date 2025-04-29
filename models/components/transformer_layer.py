import torch
import torch.nn as nn
from models.scripts.attention_function import AttentionFunction

class TransformerLayer(nn.Module):
    """
    A multi-headed transformer layer that computes edge-based attention
    and aggregates messages per node.
    
    Input node features:     [M, D]
    Input edge features:     [N, 6, D]
    Output node features:    [M, D]
    """
    def __init__(self, hidden_dim, head_size, gradient):
        """
        Args:
            hidden_dim (int): Dimension D of node and edge features.
            head_size (int): Number of attention heads H.
            gradient (float): Scaling for soft-masking invalid edges.
        """
        super().__init__()
        self.hidden_dim = hidden_dim  # D
        self.head_size = head_size   # H
        self.gradient = gradient

        self.feature_dim = hidden_dim * 3  # 3D (from value_i + value_j + edge)
        self.feature_dim_per_head = self.feature_dim // head_size  # 3D / H

        # Multi-head attention function
        self.attention_function = AttentionFunction(hidden_dim, head_size)

        # Learnable per-head linear transform: one per head
        # Projects each value vector: (3D/H) → (3D/H)
        self.linear_weight = nn.Parameter(torch.randn(
            head_size, self.feature_dim_per_head, self.feature_dim_per_head
        ))  # Shape: (H, 3D/H, 3D/H)

        self.linear_bias = nn.Parameter(torch.zeros(
            head_size, self.feature_dim_per_head
        ))  # Shape: (H, 3D/H)

        # Final projection: collapse from 3D → D after aggregation
        self.out_projection = nn.Linear(self.feature_dim, hidden_dim)

    def forward(self, node_chunk, edge_chunk):
        """
        Args:
            node_chunk: [M, D]  — features for M nodes
            edge_chunk: [N, 6, D] — edge info:
                [:, 0, :] = f_i     [N, D]  → destination node features
                [:, 1, :] = f_j     [N, D]  → source node features
                [:, 2, :] = e_ij    [N, D]  → edge features
                [:, 3, 0] = e_sym   [N]     → symmetry indicator
                [:, 4, 0] = e_mask  [N]     → validity mask
                [:, 5, 0] = tgt_idx [N]     → indices into node_chunk
        Returns:
            m_i_chunk: [M, D] → aggregated message per node
        """
        # Compute multi-head attention and value output
        alpha, value = self.attention_function(edge_chunk, self.gradient)
        # alpha, value: [H, N, 3D/H]

        # Apply linear transform to each head's values:
        # value:      [H, N, 3D/H]
        # weight:     [H, 3D/H, 3D/H]
        # bias:       [H, 3D/H]
        transformed = torch.einsum("hnd,hdf->hnf", value, self.linear_weight) + self.linear_bias.unsqueeze(1)  # [H, N, 3D/H]

        # Weight values using attention coefficients
        weighted = alpha * transformed  # [H, N, 3D/H]

        # Reshape and merge heads: [H, N, 3D/H] → [N, 3D]
        m_i_per_edge = weighted.permute(1, 0, 2).reshape(-1, self.feature_dim)

        # Scatter to sum messages per destination node:
        # m_i_per_edge: [N, 3D] → aggregated: [M, 3D]
        m_i_chunk = torch.zeros(
            node_chunk.size(0),  # num nodes (M)
            m_i_per_edge.size(1),  # feature size (3D)
            device=m_i_per_edge.device,
            dtype=m_i_per_edge.dtype
        )

        # Use scatter_add to accumulate values across nodes
        m_i_chunk.scatter_add(
            0,  # Dimension for accumulation
            edge_chunk[:, 5, 0].long().unsqueeze(1).expand(-1, m_i_per_edge.size(1)),  # Indices, expanded to match feature size
            m_i_per_edge  # Values to add
        )

        # Final projection from 3D → D
        m_i_chunk = self.out_projection(m_i_chunk)  # [M, D]

        return m_i_chunk
    
    def to(self, device):
        """
        Move the TransformerLayer instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.
        """
        self.attention_function = self.attention_function.to(device)
        self.linear_weight = nn.Parameter(self.linear_weight.to(device))
        self.linear_bias = nn.Parameter(self.linear_bias.to(device))
        self.out_projection = self.out_projection.to(device)
        return self