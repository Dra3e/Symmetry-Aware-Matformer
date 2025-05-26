import math
import torch
import torch.nn as nn

class AttentionFunction(nn.Module):
    """
    Multi-headed attention function for edge-based attention calculation.
        base_attention = Sigmoid( LNorm( Hadmard( Query, Key ) / 3*dimension_key ))
        sym_masked_attention = Hadamard( base_attention, 1 + weight * symmetry_mask * embedded_distance )
        validity_masked_attention = Hadamard( sym_masked_attention, validity_mask )

    Attributes:
        hidden_dim (int): Dimension D of node and edge features.
        num_heads (int): Number of attention heads H.
        feature_dim (int): Total feature dimension for queries, keys, and values.
        query (nn.Linear): Linear layer for query projection.
        key_i (nn.Linear): Linear layer for key projection from target node.
        key_j (nn.Linear): Linear layer for key projection from source node.
        value_i (nn.Linear): Linear layer for value projection from target node.
        value_j (nn.Linear): Linear layer for value projection from source node.
        edge (nn.Linear): Linear layer for edge feature projection.
        layer_norm (nn.LayerNorm): Layer normalization applied before sigmoid activation.
        symmetry (nn.Linear): Linear layer for symmetry projection.
    """
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
        self.feature_dim = hidden_dim * 3  # Total feature dimension for queries, keys, and values.

        # Linear projections for each input, packed across all heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key_i = nn.Linear(hidden_dim, hidden_dim)
        self.key_j = nn.Linear(hidden_dim, hidden_dim)
        self.value_i = nn.Linear(hidden_dim, hidden_dim)
        self.value_j = nn.Linear(hidden_dim, hidden_dim)
        self.edge    = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(self.feature_dim) # Normalization to apply before sigmoid attention activation

        self.symmetry = nn.Linear(1, hidden_dim, bias=False) # Linear transform for edge symmetry contribution (no bias)

    def split_heads(self, x):
        """
        Splits the input tensor into multiple heads.

        Args:
            x (torch.Tensor): (N, hidden_dim) matrix.

        Returns:
            torch.Tensor: Tensor of shape (H, N, D/H), where H is the number of heads.
        """
        N = x.size(0)
        head_dim = self.hidden_dim // self.num_heads # Calculate the size of each attention head.
        x = x.view(N, self.num_heads, head_dim) # Reshape the tensor to separate the heads
        return x.permute(1, 0, 2)  # (H, N, D/H)

    def forward(self, edges, gradient):
        """
        Computes attention weights and value matrix for a batch of edges.

        Args:
            gradient (float): Controls how much attention to give to invalid edges (higher = less).
            edges (E, 6, hidden_dim): Edges to calculate attention of.
                edge info:
                [:, 0, :] = f_i             (E, hidden_dim) matrix of destination node features
                [:, 1, :] = f_j             (E, hidden_dim) matrix of source node features
                [:, 2, :] = e_ij            (E, hidden_dim) matrix of embedded distance (edge feature)
                [:, 3, 0] = e_sym           (E) symmetry mask indicating if edge is displaced over augmentation [1=yes, 0=no]
                [:, 4, 0] = e_mask          (E) validity mask for edges indicating if graph is valid [1=yes, 0=no]
                [:, 5, 0] = target_index    (E) Indices of destination node features in 'node_chunk'.

        Returns:
            alpha (torch.Tensor): (head_size, E, 3*hidden_dim/head_size) matrix of attention scores per edge.
            value_C (torch.Tensor): (head_size, E, 3*hidden_dim/head_size) matrix of concatenated values.
        """
        E = edges.size(0)
        head_dim = self.hidden_dim // self.num_heads
        feature_dim_per_head = 3 * head_dim

        # Extract edge features
        f_i     = edges[:, 0, :]  # (E, hidden_dim) matrix of destination node features
        f_j     = edges[:, 1, :]  # (E, hidden_dim) matrix of source node features
        e_ij    = edges[:, 2, :]  # (E, hidden_dim) matrix of embedded distance (edge feature)
        m_sym   = edges[:, 3, 0]  # (E) symmetry mask indicating if edge is displaced over augmentation [1=yes, 0=no]
        m_valid = edges[:, 4, 0]  # (E) validity mask for edges indicating if graph is valid [1=yes, 0=no]

        # Apply linear projections and reshape for each head
        query      = self.split_heads(self.query(f_i))
        key_i      = self.split_heads(self.key_i(f_i))
        key_j      = self.split_heads(self.key_j(f_j))
        value_i    = self.split_heads(self.value_i(f_i))
        value_j    = self.split_heads(self.value_j(f_j))
        edge_trans = self.split_heads(self.edge(e_ij))

        # Concatenate features along the last dimension
        query_C = torch.cat((query, query, query), dim=-1)
        key_C   = torch.cat((key_i, key_j, edge_trans), dim=-1)     
        value_C = torch.cat((value_i, value_j, edge_trans), dim=-1) 

        # Raw attention score (element-wise product)
        dimension_of_key = math.sqrt(3 * (self.hidden_dim // self.num_heads))
        alpha_raw = query_C * key_C / dimension_of_key                             # (H, N, 3D/H)

        # LayerNorm + sigmoid
        # layerNorm always normalises the last dimension, so permute(1,0,2) swaps H and N dimension.
        # In order to apply layerNorm to all heads without looping, reshape(N,-1) collects all heads into one tensor
        alpha = alpha_raw.permute(1, 0, 2).reshape(E, -1)           # (N, 3D)
        alpha = torch.sigmoid(self.layer_norm(alpha))               # (N, 3D)
        alpha = alpha.view(E, self.num_heads, feature_dim_per_head).permute(1, 0, 2)  # (H, N, 3D/H)

        # Symmetry projection: (N,) → (N, 1) → (H, N, D/H)
        m_sym = m_sym.unsqueeze(-1)  # (N, 1)
        sym_proj = self.split_heads(self.symmetry(m_sym))           # (head_size, E, hidden_dim/head_size)
        symmetry = 1 + sym_proj * edge_trans                        # (head_size, E, hidden_dim/head_size)
        symmetry = symmetry.repeat(1, 1, 3)                         # (head_size, E, 3*hidden_dim/head_size)

        # Validity modifier that modify the attention weights based on validity mask
        m_valid = torch.sigmoid(gradient * (m_valid - 0.5))         # (N,)
        m_valid = m_valid.unsqueeze(0).unsqueeze(-1)                # (1, N, 1)
        m_valid = m_valid.repeat(1, 1, feature_dim_per_head)        # (1, N, 3*idden_dim/head_size)

        # Final attention weights
        alpha = alpha * symmetry * m_valid                          # (head_size, E, 3*hidden_dim/head_size)

        return alpha, value_C  # both (head_size, E, 3*hidden_dim/head_size)
    
    def to(self, device):
        """
        Move the AttentionFunction instance to the specified device.

        Args:
            device (torch.device): The device to move the instance to.

        Returns:
            Attention Funcion: instance of the Attention Function class moved to the specified device. 
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