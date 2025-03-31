import torch
import torch.nn as nn

class Aggregation(nn.Module):
    """ script class for aggregation step in the transformer """
    def __init__(self):
        """ Initializes an instance of the Aggregation class """
        super(Aggregation, self).__init__()
    
    def forward(self, alpha, V):
        """
        Performs the aggregation of the attention scores and values of the matformer.
            m_i = SUM_edge_to_neighbors_of_atom_i( SUM_heads_in_layer( alpha_head*Value_head ) )
        
        Args:
            attention (torch.Tensor): The attention scores for each edge for each head.
            value (torch.Tensor): The transformer values for each edge for each head.

        Returns:
            atom_strength (torch.Tensor): representing the strength/effect for each atom.
        """
        # Compute weighted_values
        weighted_values_list = [alpha * value for alpha, value in zip(alpha, V)] # zip() pairs up corresponding alpha_matrix and value_matrix from both lists.

        # Compute summation
        all_weighted_values = torch.stack(weighted_values_list, dim=0)  # (N, head_size, batch_size, 1)
        summed_over_heads = torch.sum(all_weighted_values, dim=1)  # (N, batch_size, 1)
        m_i = torch.sum(summed_over_heads, dim=0)  # (batch_size, 1)

        return m_i