import torch

# Apply Group Theory to Join All Valid Augmented Crystal Graphs
def join_valid_augmented_crystal_graphs(crystals, axes, planes, directions):
    valid_augmentations = []
    for crystal in crystals:
        augmentations = crystal.apply_augmentations(axes, planes, directions)
        valid_augmentations.extend(augmentations)

    valid_graphs = torch.Graph.create_batch(valid_augmentations)
    stitched_graph = Graph.stitch_graphs(valid_graphs)
    return stitched_graph

# Create a Series of Invalid Augmented Crystals
def create_invalid_augmentations(crystals, axes, planes, directions):
    invalid_augmentations = []
    for crystal in crystals:
        augmentations = crystal.apply_augmentations(axes, planes, directions)
        # Apply some invalid transformations (e.g., invalid angles or planes)
        for aug in augmentations:
            invalid_augmentations.append(aug)  # Add invalid logic here
    return invalid_augmentations

# Apply Transformations to Invalid Augmented Crystals
def apply_transformations_to_invalid_crystals(invalid_augmentations):
    invalid_graphs = Graph.create_batch(invalid_augmentations)
    return invalid_graphs

# Create Graphs of Invalid Augmented Crystals
def create_invalid_augmented_crystal_graphs(crystals, axes, planes, directions):
    invalid_augmentations = create_invalid_augmentations(crystals, axes, planes, directions)
    invalid_graphs = apply_transformations_to_invalid_crystals(invalid_augmentations)
    return invalid_graphs



# ----------------------------------------------------------------------------------------- #

#Output All Graphs
def output_all_graphs(valid_graph, invalid_graphs):
    print("Valid Stitched Graph:")
    print("Nodes:", valid_graph.nodes)
    print("Edges:", valid_graph.edges)

    print("\nInvalid Graphs:")
    for i, graph in enumerate(invalid_graphs):
        print(f"Invalid Graph {i+1}:")
        print("Nodes:", graph.nodes)
        print("Edges:", graph.edges)