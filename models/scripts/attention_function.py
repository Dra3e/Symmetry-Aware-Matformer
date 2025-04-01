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
        super(AttentionFunction, self).__init__()
        self.head_size = head_size  # Number of attention heads

        # Use single linear layers with multiple output dimensions for all heads
        self.query = nn.Linear(hidden_dim, hidden_dim * head_size)
        self.key_i = nn.Linear(hidden_dim, hidden_dim * head_size)
        self.key_j = nn.Linear(hidden_dim, hidden_dim * head_size)
        self.value_i = nn.Linear(hidden_dim, hidden_dim * head_size)
        self.value_j = nn.Linear(hidden_dim, hidden_dim * head_size)
        self.edge = nn.Linear(hidden_dim, hidden_dim * head_size)

        self.layer_norm = nn.LayerNorm(hidden_dim * 3)  
        self.symmetry = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, edge, gradient):
        """
        Calculates the attention of a list of edges.
            alpha_ij = Sigmoid(LinearTranform(Q*K/d)) + (1+Weight*M_sym*edge_ij) + Sigmoid(M_valid)

        Args:
            edge: A list of tuples with size 5 that represent edges that point away from nodes in node_chunk.
                edge_tuple[0]: A vector of size hidden_dim representing the node the edge points away from.
                edge_tuple[1]: A vector of size hidden_dim representing the node the edge points to.
                edge_tuple[2]: A vector of size hidden_dim representing the strength of the edge.
                edge_tuple[3]: M_sym: a scaler of value 0 or 1, 0 if physical, 1 if augmentation displacement.
                edge_tuple[4]: M_valid: a scaler of value 0 or 1, 1 if valid, 0 if invalid.
            gradient (float): A value between 1 and 0, how to much attention should be payed to invalid edges.
                If 0 then they weigh the same, if 1 and invalid edges are ignored.
        """
        batch_size, num_edges, feat_dim = edge.shape  # Assuming edge is (batch_size, num_edges, feat_dim)
        head_dim = feat_dim // self.head_size

        # Project the inputs for all heads in parallel
        query = self.query(edge[:, :, 0]).view(batch_size, num_edges, self.head_size, head_dim)
        key_i = self.key_i(edge[:, :, 0]).view(batch_size, num_edges, self.head_size, head_dim)
        key_j = self.key_j(edge[:, :, 1]).view(batch_size, num_edges, self.head_size, head_dim)
        value_i = self.value_i(edge[:, :, 0]).view(batch_size, num_edges, self.head_size, head_dim)
        value_j = self.value_j(edge[:, :, 1]).view(batch_size, num_edges, self.head_size, head_dim)
        edge_transformed = self.edge(edge[:, :, 2]).view(batch_size, num_edges, self.head_size, head_dim)

        query_C = torch.cat((query, query, query), dim=-1)
        key_C = torch.cat((key_i, key_j, edge_transformed), dim=-1)
        value_C = torch.cat((value_i, value_j, edge_transformed), dim=-1)

        d = key_C.shape[-1]
        alpha_original = torch.sigmoid(self.layer_norm(query_C * key_C / d))

        symmetry = 1 + self.symmetry(edge[:, :, 3] * edge[:, :, 2])
        m_valid = 1 / (1 + torch.exp(-gradient * (edge[:, :, 4] - 0.5)))

        alpha = alpha_original * symmetry * m_valid

        return alpha, value_C  # Shape: (batch_size, num_edges, head_size, head_dim)