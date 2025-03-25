import torch
import torch.nn as nn

class Aggregation(nn.Module):
    def aggregate(self, alpha, V):
        """
        input: Array of inputs
        alpha: list of Attention weights (list of n x n matrixes)
        V: list of Value matrix (list of n x d matrixes, hvor d er feature dimension)

        Output: Aggregated features (batch_size x 1)
        """
        # Compute weighted_values
        weighted_values_list = [alpha * value for alpha, value in zip(alpha, V)] # zip() pairs up corresponding alpha_matrix and value_matrix from both lists.

        # Compute summation
        all_weighted_values = torch.stack(weighted_values_list, dim=0)  # (N, head_size, batch_size, 1)
        summed_over_heads = torch.sum(all_weighted_values, dim=1)  # (N, batch_size, 1)
        m_i = torch.sum(summed_over_heads, dim=0)  # (batch_size, 1)

        return m_i

# class Aggregation(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(Aggregation, self).__init__()
#         self.linear_layer = nn.Linear(input_dim, hidden_dim)

#     def forward(self, neighbors, alpha, V, edges):
#         """
#         input: Array of inputs
#         alpha: Attention weights (n x n matrix)
#         V: Value matrix (n x d, hvor d er feature dimension)
#         edges: List of edges

#         Output: Aggregated features (n x hidden_dim)
#         """
#         # Use nn.Linear on V
#         V_transformed = self.linear_layer(V)

#         return sum(
#             sum(alpha * V_transformed for _ in range(len(neighbors)))
#             for _ in range(len(edges))
#         )