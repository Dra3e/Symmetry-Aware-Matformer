import torch.nn as nn

class Aggregation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Aggregation, self).__init__()
        self.linear_layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, neighbors, alpha, V, edges):
        """
        input: Array of inputs
        alpha: Attention weights (n x n matrix)
        V: Value matrix (n x d, hvor d er feature dimension)
        edges: List of edges

        Output: Aggregated features (n x hidden_dim)
        """
        # Use nn.Linear on V
        V_transformed = self.linear_layer(V)

        return sum(
            sum(alpha * V_transformed for _ in range(len(neighbors)))
            for _ in range(len(edges))
        )