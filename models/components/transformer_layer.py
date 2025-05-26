import torch
import torch.nn as nn

from models.scripts.attention_function import AttentionFunction

class TransformerLayer(nn.Module):
    """
    A multi-headed transformer layer that computes edge-based attention and aggregates messages per node.
    
    Attributes:
        hidden_dim (int): Epected dimension of inputs and outputs.
        head_size (int): Number of attention heads.
        Gradient (float): Scaling for soft-masking invalid edges.
        feature_dim (int): Dimension of concatenated queries, keys, and values, (3*hidden_dim).
        feature_dim_per_head (int): Dimension allocated to each head (3*hidden_dim/head_size).
        attention_function (AttentionFunction): Instance of the AttentionFunction class.
        linear_weight (nn.Parameter): Learnable weight for each head.
        linear_bias (nn.Parameter): Learnable bias for each head.
        out_projection (nn.Linear): Linear layer for final projection.
    """
    def __init__(self, hidden_dim, head_size, gradient):
        """
        Initializes the TransformerLayer.

        Args:
            hidden_dim (int): Epected dimension of inputs and outputs.
            head_size (int): Number of attention heads.
            gradient (float): Scaling for soft-masking invalid edges.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.gradient = gradient

        self.feature_dim = hidden_dim * 3  # dimension of concatenated queries, keys, and values.
        self.feature_dim_per_head = self.feature_dim // head_size  # the dimension allocated to each head.

        self.attention_function = AttentionFunction(hidden_dim, head_size)

        # Learnable weight for each head.
        self.linear_weight = nn.Parameter(torch.randn(
            head_size, self.feature_dim_per_head, self.feature_dim_per_head # Create parameter as a 3D matrix.
        )) 

        # Learnable bias for each head.
        self.linear_bias = nn.Parameter(torch.zeros(
            head_size, self.feature_dim_per_head # Create parameter as a 2D matrix.
        ))

        self.out_projection = nn.Linear(self.feature_dim, hidden_dim)

    def forward(self, node_chunk, edge_chunk):
        """
        Forward pass through the transformer layer.

        Args:
            node_chunk (N, hidden_dim): List of N nodes/atoms to process.
            edge_chunk (E, 6, hidden_dim): Edges going to or from nodes in node_chunk.
                edge info:
                [:, 0, :] = f_i         (E, hidden_dim) matrix of destination node features
                [:, 1, :] = f_j         (E, hidden_dim) matrix of source node features
                [:, 2, :] = e_ij        (E, hidden_dim) matrix of embedded distance (edge feature)
                [:, 3, 0] = e_sym       (E) symmetry mask indicating if edge is displaced over augmentation [1=yes, 0=no]
                [:, 4, 0] = e_mask      (E) validity mask for edges indicating if graph is valid [1=yes, 0=no]
                [:, 5, 0] = target_idx  (E) Indices of destination node features in 'node_chunk'.
        Returns:
            m_i_chunk (N, hidden_dim): aggregated message per node in node_chunk.
        """
        # Get attention (alpha) and concatenated value of each edge.
        alpha, value = self.attention_function.forward(edge_chunk, self.gradient)

        # Calculate message of each edge.
        weighted_value = torch.einsum("hnd,hdf->hnf", value, self.linear_weight) + self.linear_bias.unsqueeze(1) # Apply linear transformation to value vectors for each head.
        attention_scores = alpha * weighted_value
        message_per_edge = attention_scores.permute(1, 0, 2).reshape(-1, self.feature_dim) # Reshape and merge heads, such that the 3D matrix; attention_scores, becomes a 2D matrix of size (E, 3*hidden_dim).
        
        # Aggregate attention scores for each node in node_chunk.
        message_per_atom_in_chunk = torch.zeros( # Create a (M, 3*hidden_dim) matrix where all entries are 0.
            node_chunk.size(0),             # Number of nodes: M
            message_per_edge.size(1),       # feature_dim: 3*hidden_dim
            device=message_per_edge.device,
            dtype=message_per_edge.dtype
        )
        message_per_atom_in_chunk.scatter_add_( # Calculate message of each node by summarizing messages of each edge with source from that node.
            dim=0,
            index=edge_chunk[:, 5, 0].long().unsqueeze(1).expand(-1, message_per_edge.size(1)),
            src=message_per_edge
        )
        
        message_per_atom_in_chunk = self.out_projection(message_per_atom_in_chunk) # Apply final projection to reduce features

        return message_per_atom_in_chunk
    
    def to(self, device):
        """
        Move the TransformerLayer instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.

        Returns:
            TransformerLayer: instance of the TransformerLayer class moved to the specified device. 
        """
        self.attention_function = self.attention_function.to(device)
        self.linear_weight = nn.Parameter(self.linear_weight.to(device))
        self.linear_bias = nn.Parameter(self.linear_bias.to(device))
        self.out_projection = self.out_projection.to(device)
        return self