import numpy as np
import torch

from components.group_theory import compute_e3_invariant_edges

#To run write "python -m models.group_theory_example"

# Define a simple cubic unit cell with three atoms
L = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])  # Unit cell of size 1x1x1

X = np.array([
    [0.0, 0.0, 0.0],  # Atom 1 at (0, 0, 0)
    [0.5, 0.5, 0.0],  # Atom 2 at (0.5, 0.5, 0)
    [0.5, 0.0, 0.5]   # Atom 3 at (0.5, 0, 0.5)
])

A = np.array([
    [1],  # Feature for Atom 1
    [2],  # Feature for Atom 2
    [3]   # Feature for Atom 3
])

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
L_tensor = torch.tensor(L, dtype=torch.float32)
A_tensor = torch.tensor(A, dtype=torch.float32)

# Run edge computation
edge_index, edge_attr = compute_e3_invariant_edges(A_tensor, X_tensor, L_tensor)

print("Edge Index:\n", edge_index.numpy())
print("Edge Attributes (Distances):\n", edge_attr.numpy())
