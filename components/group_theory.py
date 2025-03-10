import torch

def compute_e3_invariant_edges(X):
    """
    Computes edges invariant under E(3) transformations.
    
    Parameters:
    - X (torch.Tensor): A tensor of shape (N, 3) representing N atom positions in 3D space.

    Returns:
    - edge_index (torch.Tensor): A (2, num_edges) tensor of node connections.
    - edge_attr (torch.Tensor): A (num_edges,) tensor containing edge distances.
    """
    num_nodes = X.shape[0]  # Number of atoms (nodes)
    edges = []  # Store node connections
    features = []  # Store edge attributes (distances)

    # Iterate over all unique pairs (i, j) to compute edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = torch.norm(X[i] - X[j])  # Compute Euclidean distance (E(3)-invariant)
            edges.append((i, j))  # Store edge connection
            features.append(distance)  # Store distance as an edge feature

    # Convert lists to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape: (2, num_edges)
    edge_attr = torch.tensor(features, dtype=torch.float32)  # Shape: (num_edges,)

    return edge_index, edge_attr