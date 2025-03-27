import torch
import torch.nn as nn
from models.scripts.graph_embedding import CGCNN, EdgeTransformation

class CrystalGraph(nn.Module):
    """
    A unified class combining graph generation from crystal structures and embedding.
    """

    def __init__(self, nodes=None, edges=None, config=3.0):
        """
        Initializes a CrystalGraph. Use `init_from_crystal()` to create an instance from a crystal.
        """
        super().__init__()
        self.nodes = nodes  # Atomic feature matrix
        self.edges = edges  # Edge distance matrix
        self.config = config

    @classmethod
    def init_from_crystal(cls, crystal, config=3.0):
        """
        Creates a CrystalGraph instance from a given crystal structure.

        Args:
            crystal: A crystal object containing atomic information.
            config (dict, optional): Configuration for cutoff value.

        Returns:
            CrystalGraph instance.
        """
        nodes = crystal.A
        num_nodes = nodes.shape[0]
        edges = []
        translation_offsets = torch.tensor([
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=torch.float32)

        for i in range(num_nodes):
            for j in range(i, num_nodes):  # include i==j to allow self-loops if needed
                for t in translation_offsets:
                    translated = crystal.X[j] + t @ crystal.L
                    distance = torch.norm(crystal.X[i] - translated)
                    if distance <= config:
                        edges.append([i, j, distance.item(), 0])
                        if i != j:
                            edges.append([j, i, distance.item(), 0])  # undirected

        return cls(nodes=nodes, edges=edges, config=config)
    
    @classmethod
    def init_from_augmented_crystal(cls, original_crystal, augmented_crystals, config=3.0):
        """
        Initializes a CrystalGraph from an original crystal and a list of its augmented versions.
        Computes standard edges via init_from_crystal and adds displacement as self-loop edges.
        """
        base_graph = cls.init_from_crystal(original_crystal, config)
        nodes = base_graph.nodes
        edges = base_graph.edges.copy()
        num_nodes = nodes.shape[0]

        for aug in augmented_crystals:
            displacements = torch.norm(aug.X - original_crystal.X, dim=1)
            for i in range(num_nodes):
                edges.append([i, i, displacements[i].item(), 1])  # augmented/self-edge

        return cls(nodes=nodes, edges=edges, config=config)
    

    @staticmethod
    def embed_graph_batch(graphs, atom_feature_dim, hidden_dim, num_rbf_kernels):
        """
        Parallel embedding for multiple graphs.

        Args:
            graphs (list of CrystalGraph): List of graphs with the same nodes.
            atom_feature_dim (int): Feature dimension of atomic nodes.
            hidden_dim (int): Hidden dimension for embeddings.
            num_rbf_kernels (int): Number of RBF kernels.

        Returns:
            f_star (torch.Tensor): Batched node feature tensor.
            e_h (torch.Tensor): Batched edge feature tensor.
        """
        if not graphs:
            raise ValueError("Graph list is empty!")

        # Assume all graphs share the same nodes
        node_tensor = torch.stack([graph.nodes for graph in graphs])  # Shape: (num_graphs, num_nodes, feature_dim)
        edge_tensor = torch.stack([graph.edges for graph in graphs])  # Shape: (num_graphs, num_edges, edge_dim)

        # Initialize models
        atom_embedding = CGCNN(atom_feature_dim, hidden_dim)
        edge_transform = EdgeTransformation(num_rbf_kernels, hidden_dim)

        # Process all graphs at once
        f_star = atom_embedding(node_tensor)  # Parallel computation
        e_h = edge_transform(edge_tensor)    # Parallel computation

        return f_star, e_h  # Now batched over all graphs

    @staticmethod
    def create_batch(crystals):
        """Creates a batch of CrystalGraph objects from a list of crystals."""
        return [CrystalGraph.init_from_crystal(crystal) for crystal in crystals]