import torch.nn as nn

def aggregation(input, alpha, V, edges, hidden_dim):
    """
    input: Array of inputs
    alpha: Attention weights (n x n matrix)
    V: Value matrix (n x d, hvor d er feature dimension)
    edges: List of edges
    hidden_dim: Dimension of the hid

    Output: Aggregated features (n x hidden_dim)
    """
    # Create nn.Linear-laget
    linear_layer = nn.Linear(V.shape[1], hidden_dim)

    # Use nn.Linear on V
    V_transformed = linear_layer(V)

    return sum(
        sum(alpha * V_transformed for _ in range(len(input)))
        for _ in range(len(edges))
    )