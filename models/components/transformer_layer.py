import torch
import torch.nn as nn
from models.scripts.attention_function import AttentionFunction

class TransformerLayer(nn.Module):
    """ A multi headed transformer layer """
    def __init__(self, hidden_dim, head_size):
        """
        Initilizes TransformerLayer instance as a multi headed transformer layer with its dimensions.

        Args:
            hidden_dim (int): The expected dimension of the input and the dimension of the output.
            head_size (int): How many heads to initialize.
        """
        super(TransformerLayer, self).__init__()
        self.head_size = head_size
        self.attention_function = AttentionFunction(hidden_dim, head_size)
        feature_dim = 5 * hidden_dim  
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, node_chunk, edge_chunk, gradient):
        """
            Calculates the strength/effect of each atom in the crystal on other atoms.

            Args:
                node_chunk (torch.tensor): A matrix of size chunk_size x hidden_dim, representing a set of node in a graph.
                edge_chunk: A list of tuples with size 5 that represent edges that point away from nodes in node_chunk.
                    edge_tuple[0]: A vector of size hidden_dim representing the node the edge points away from.
                    edge_tuple[1]: A vector of size hidden_dim representing the node the edge points to.
                    edge_tuple[2]: A vector of size hidden_dim representing the strength of the edge.
                    edge_tuple[3]: A scaler of value 0 or 1, 0 if physical, 1 if augmentation displacement.
                    edge_tuple[4]: A scaler of value 0 or 1, 1 if valid, 0 if invalid.
                gradient (float): A value between 1 and 0, how to much attention should be payed to invalid edges.
                    If 0 then they weigh the same, if 1 and invalid edges are ignored.
            
            Returns:
                m_chunk (torch.tensor): Ordered list of the strength of each node in the crystal.
                    The placement of m value determines what atom (node) it represents.
        """
        m_chunk = []

        for atoms, edges in zip(node_chunk, edge_chunk):
            # Convert edge list into a tensor
            crystal_edges_tensor = torch.stack(edges, dim=0)  # (num_edges, edge_feature_dim)

            # Maps each edge to its corresponding atom's unique index.
            unique_atoms, atom_indices = torch.unique(crystal_edges_tensor[:, 0], return_inverse=True)

            # Get dimensions for edges per atom
            max_edges = atom_indices.bincount().max().item() # The maximum number of edges that any atom in this crystal has.
            nr_Atoms = len(unique_atoms) # The number of atoms in this crystal.
            edge_feature_dim = crystal_edges_tensor.shape[1]

            # Efficient edge gathering via scatter
            padded_edges = torch.zeros(nr_Atoms, max_edges, edge_feature_dim, device=crystal_edges_tensor.device)
            mask = torch.zeros(nr_Atoms, max_edges, dtype=torch.bool, device=crystal_edges_tensor.device)

            # Create index matrix for scatter operation, scatter_indices computes where each edge should be placed.
            num_edges_per_atom = atom_indices.bincount()
            scatter_indices = torch.arange(len(atom_indices), device=crystal_edges_tensor.device) - torch.cumsum(num_edges_per_atom, dim=0) + num_edges_per_atom
            scatter_indices = scatter_indices.clamp(max=max_edges - 1)

            # Scatter edges into the padded tensor and mask empty spaces
            padded_edges.index_add_(1, scatter_indices, crystal_edges_tensor) # can run in parallel in stead of looping
            mask.index_fill_(1, scatter_indices, 1)

            # Apply multi-head attention in parallel
            alpha_matrix, value_matrix = self.attention_function(padded_edges, gradient)

            # Apply linear transformation
            transformed_value_matrix = self.linear(value_matrix)

            # Compute weighted values
            weighted_values = alpha_matrix * transformed_value_matrix  # (batch_size, max_edges, head_size, feature_dim)
            summed_over_heads = torch.sum(weighted_values, dim=2)  # Sum across heads -> (batch_size, max_edges, feature_dim)

            # Use masked_fill_() instead of multiplication for efficiency
            summed_over_heads.masked_fill_(~mask.unsqueeze(-1), 0)
            m_i = torch.sum(summed_over_heads, dim=1)  # Sum across valid edges -> (batch_size, feature_dim)

            m_chunk.append(m_i)

        return m_chunk