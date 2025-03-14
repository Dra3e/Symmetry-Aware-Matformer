import torch
import numpy as np

from components.group_theory import Axis, Crystal, Plane, apply_augmentations

#To run write "python -m models.group_theory_example"

def test_crystal_augmentations():
    # Define a simple crystal with 3 atoms and a basic lattice
    A = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)  # Atomic features
    X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)  # Atomic positions
    L = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float32)  # Lattice vectors

    # Create a crystal instance
    crystal = Crystal(A, X, L)

    # Define test axes and planes
    axes = [Axis([1, 0, 0], 90), Axis([0, 1, 0], 180)]  # Rotation axes
    planes = [Plane([1, 0, 0]), Plane([0, 1, 0])]  # Reflection planes

    # Apply augmentations
    augmentations = apply_augmentations(crystal, axes, planes)
    
    # Generate edges for each augmented crystal
    augmented_crystals = [aug[1] for aug in augmentations]  # Extract Crystal objects
    graphs = torch.Graph.create_batch(augmented_crystals)
    
    # Stitch graphs together
    stitched_graph = torch.Graph.stitch_graphs(graphs)
    
    # Print results
    print("Original Crystal:")
    print("A:", crystal.A, "X:", crystal.X, "L:", crystal.L)
    print("Number of augmentations:", len(augmentations))
    print("Number of graphs:", len(graphs))
    print("Stitched Graph Nodes:", stitched_graph.nodes.shape)
    print("Stitched Graph Adjacency Matrix:", stitched_graph.adjacency_matrix.shape)
    
    return stitched_graph

# Run the test
test_crystal_augmentations()
