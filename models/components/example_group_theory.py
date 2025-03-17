import torch
import numpy as np

from components.group_theory import Axis, Crystal, Plane, apply_augmentations, create_invalid_augmented_crystal_graphs, join_valid_augmented_crystal_graphs, output_all_graphs

#To run write "python -m models.group_theory_example"

def test_crystal_augmentations():
    # Define sample data
    A = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    X = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    L = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    crystal = Crystal(A, X, L)

    # Define axes and planes
    axes = [Axis([1, 0, 0], 90), Axis([0, 1, 0], 90), Axis([0, 0, 1], 90)]
    planes = [Plane([1, 0, 0]), Plane([0, 1, 0]), Plane([0, 0, 1])]

    # Apply augmentations
    augmentations = crystal.apply_augmentations(axes, planes)
    print(f"Number of augmentations: {len(augmentations)}")

    # Create graphs for each augmented crystal
    graphs = torch.Graph.create_batch(augmentations)
    print(f"Number of graphs created: {len(graphs)}")

    # Stitch all graphs together
    stitched_graph = torch.Graph.stitch_graphs(graphs)
    print("Stitched Graph:")
    print("Nodes:", stitched_graph.nodes)
    print("Edges:", stitched_graph.edges)

    # Test the new functions
    crystals = [crystal]  # List of crystals to test with

    # Join valid augmented crystal graphs
    valid_stitched_graph = join_valid_augmented_crystal_graphs(crystals, axes, planes)
    print("Valid Stitched Graph:")
    print("Nodes:", valid_stitched_graph.nodes)
    print("Edges:", valid_stitched_graph.edges)

    # Create invalid augmented crystal graphs
    invalid_graphs = create_invalid_augmented_crystal_graphs(crystals, axes, planes)
    print("\nInvalid Graphs:")
    for i, graph in enumerate(invalid_graphs):
        print(f"Invalid Graph {i+1}:")
        print("Nodes:", graph.nodes)
        print("Edges:", graph.edges)

    # Output all graphs
    output_all_graphs(valid_stitched_graph, invalid_graphs)

# Run the test
test_crystal_augmentations()
