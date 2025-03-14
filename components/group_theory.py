import torch
import spglib
import numpy as np

from components.rotation import rotate_crystal
from components.reflection import reflect_crystal
from components.translation import translate_crystal

class Plane:
    def __init__(self, N, P=None):
        self.N = torch.tensor(N, dtype=torch.float32)  # Convert to PyTorch tensor
        self.P = torch.tensor(P, dtype=torch.float32) if P is not None else None

class Axis:
    def __init__(self, V, angle, P=None):
        self.V = torch.tensor(V, dtype=torch.float32)  # Convert to PyTorch tensor
        self.angle = angle
        self.P = torch.tensor(P, dtype=torch.float32) if P is not None else None

class Crystal:
    def __init__(self, A, X, L):
        self.A = A.clone().detach() if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
        self.X = X.clone().detach() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.L = L.clone().detach() if isinstance(L, torch.Tensor) else torch.tensor(L, dtype=torch.float32)

    def apply_augmentations(self, axes, planes):
        """
        Generates valid augmentations for a crystal structure based on the given axes and planes.
        Parameters:
        - axes: List of Axis objects representing rotation axes.
        - planes: List of Plane objects representing reflection planes.
        """
        augmentations = []

        # Apply translation
        for translation in [None, [1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            augmented_crystal = Crystal(self.A, self.X, self.L)
            if translation is not None:
                # Convert translation to a PyTorch tensor with integer dtype
                translation_tensor = torch.tensor(translation, dtype=torch.int32)
                augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = translate_crystal(
                    self.A, self.X, self.L, translation_tensor
                )
            augmentations.append(augmented_crystal)

        # Apply rotations
        for axis in axes:
            for angle in [90, 180, 270]:
                augmented_crystal = Crystal(self.A, self.X, self.L)
                augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = rotate_crystal(
                    self.A, self.X, self.L, axis.V, angle
                )
                augmentations.append(augmented_crystal)

        # Apply reflections
        for plane in planes:
            augmented_crystal = Crystal(self.A, self.X, self.L)
            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = reflect_crystal(
                self.A, self.X, self.L, plane.N
            )
            augmentations.append(augmented_crystal)

        # Apply combinations of transformations
        for translation in [None, [1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            for axis in axes:
                for angle in [90, 180, 270]:
                    # Translation + Rotation
                    augmented_crystal = Crystal(self.A, self.X, self.L)
                    if translation is not None:
                        translation_tensor = torch.tensor(translation, dtype=torch.int32)
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = translate_crystal(
                            self.A, self.X, self.L, translation_tensor
                        )
                    augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = rotate_crystal(
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, axis.V, angle
                    )
                    augmentations.append(augmented_crystal)

                    for plane in planes:
                        # Translation + Reflection
                        augmented_crystal = Crystal(self.A, self.X, self.L)
                        if translation is not None:
                            translation_tensor = torch.tensor(translation, dtype=torch.int32)
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = translate_crystal(
                                self.A, self.X, self.L, translation_tensor
                            )
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = reflect_crystal(
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, plane.N
                        )
                        augmentations.append(augmented_crystal)

                        # Rotation + Reflection
                        augmented_crystal = Crystal(self.A, self.X, self.L)
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = rotate_crystal(
                            self.A, self.X, self.L, axis.V, angle
                        )
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = reflect_crystal(
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, plane.N
                        )
                        augmentations.append(augmented_crystal)

                        # Translation + Rotation + Reflection
                        augmented_crystal = Crystal(self.A, self.X, self.L)
                        if translation is not None:
                            translation_tensor = torch.tensor(translation, dtype=torch.int32)
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = translate_crystal(
                                self.A, self.X, self.L, translation_tensor
                            )
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = rotate_crystal(
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, axis.V, angle
                        )
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = reflect_crystal(
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, plane.N
                        )
                        augmentations.append(augmented_crystal)

        return augmentations

    # # Print augmentations
    # for aug in augmentations:
    #     print(f"A: {aug.A}, X: {aug.X}, L: {aug.L}")

    #for name, aug in augmentations:
    #    print(f"{name}: A: {aug.A}, X: {aug.X}, L: {aug.L}")
    
class Graph:
    def __init__(self, crystal, cutoff=3.0):
        self.crystal = crystal  # Store the crystal object
        self.nodes = crystal.A
        self.edges = self.get_edges(cutoff)  # Pass only cutoff

    def get_edges(self, cutoff=3.0):
        """
        Computes edges and edge attributes based on atomic positions,
        ensuring E(3) invariance (translation, rotation, reflection).

        Parameters:
        - cutoff (float): Maximum bond distance to consider for edges.

        Returns:
        - edge_distance (torch.Tensor): Tensor of shape (num_edges,) containing edge distances.
        """
        A, X, L = self.crystal.A, self.crystal.X, self.crystal.L
        num_nodes = X.shape[0]
        distances = []

        # Translation vectors for adjacent unit cells
        translation_vectors = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=torch.float32) @ L

        # Create extended unit cells
        extended_X = [X]  # Start with original unit cell
        for vec in translation_vectors:
            vec = vec.unsqueeze(0)  # Reshape vec to (1, 3) for broadcasting
            extended_X.append(X + vec)
        extended_X = torch.cat(extended_X, dim=0)

        # Compute pairwise distances and build edges
        for i in range(num_nodes):
            for j in range(len(extended_X)):
                distance = torch.norm(X[i] - extended_X[j])
                if distance <= cutoff:
                    distances.append(distance)

        edge_distance = torch.tensor(distances, dtype=torch.float32)
        return edge_distance

    @staticmethod
    def create_batch(crystals):
        graphs = []
        for crystal in crystals:
            new_graph = Graph(crystal)
            graphs.append(new_graph)
        return graphs

    @staticmethod
    def stitch_graphs(graphs):
        """Stitches multiple graphs into a single graph by combining their adjacency matrices.
        Parameters:
        - graphs (list of Graph): List of Graph objects to be stitched together.

        Returns:
        - stitched_graph (Graph): A new Graph object representing the stitched graph.
        """
        if not graphs:
            return None

        # Initialize stitched edge distances and nodes with the first graph's data
        stitched_edge_distance = graphs[0].edges.clone()
        stitched_nodes = graphs[0].nodes.clone()

        for graph in graphs[1:]:
            num_existing_nodes = stitched_nodes.shape[0]
            num_new_nodes = graph.nodes.shape[0]
            new_size = num_existing_nodes + num_new_nodes

            # Expand the stitched edge distances to accommodate the new graph
            expanded_edge_distance = torch.full((new_size, new_size), float('inf'))
            expanded_edge_distance[:num_existing_nodes, :num_existing_nodes] = stitched_edge_distance
            expanded_edge_distance[num_existing_nodes:, num_existing_nodes:] = graph.edges

            # Update the stitched edge distances
            stitched_edge_distance = expanded_edge_distance

            # Concatenate the nodes
            stitched_nodes = torch.cat((stitched_nodes, graph.nodes), dim=0)

        stitched_graph = Graph.__new__(Graph)
        stitched_graph.nodes = stitched_nodes
        stitched_graph.edges = stitched_edge_distance

        return stitched_graph

# Apply Group Theory to Join All Valid Augmented Crystal Graphs
def join_valid_augmented_crystal_graphs(crystals, axes, planes):
    valid_augmentations = []
    for crystal in crystals:
        augmentations = crystal.apply_augmentations(axes, planes)
        valid_augmentations.extend(augmentations)

    valid_graphs = Graph.create_batch(valid_augmentations)
    stitched_graph = Graph.stitch_graphs(valid_graphs)
    return stitched_graph

# Create a Series of Invalid Augmented Crystals
def create_invalid_augmentations(crystals, axes, planes):
    invalid_augmentations = []
    for crystal in crystals:
        augmentations = crystal.apply_augmentations(axes, planes)
        # Apply some invalid transformations (e.g., invalid angles or planes)
        for aug in augmentations:
            invalid_augmentations.append(aug)  # Add invalid logic here
    return invalid_augmentations

# Apply Transformations to Invalid Augmented Crystals
def apply_transformations_to_invalid_crystals(invalid_augmentations):
    invalid_graphs = Graph.create_batch(invalid_augmentations)
    return invalid_graphs

# Create Graphs of Invalid Augmented Crystals
def create_invalid_augmented_crystal_graphs(crystals, axes, planes):
    invalid_augmentations = create_invalid_augmentations(crystals, axes, planes)
    invalid_graphs = apply_transformations_to_invalid_crystals(invalid_augmentations)
    return invalid_graphs

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

if __name__ == "__main__":
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