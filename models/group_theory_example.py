import torch
from components.group_theory import compute_e3_invariant_edges

#  Example: Generate a random set of 5 atoms in 3D space
# X_transformed = torch.rand((5, 3))
# Example: 5 atoms in 3D space
X_transformed = torch.tensor([
    [0.0, 0.0, 0.0],  # Atom 0
    [1.0, 0.0, 0.0],  # Atom 1
    [0.0, 1.0, 0.0],  # Atom 2
    [0.0, 0.0, 1.0],  # Atom 3
    [1.0, 1.0, 1.0]   # Atom 4
])

# Compute the E(3)-invariant edges
edge_index, edge_attr = compute_e3_invariant_edges(X_transformed)

# Print graph edges and attributes
print("Edge Index:\n", edge_index)
print("Edge Attributes (Distances):\n", edge_attr)