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
        cutoff = config

        num_nodes = nodes.shape[0]
        edge_distance = torch.full((num_nodes, num_nodes), float('inf'))
        translation_offsets = torch.tensor([[0, 0, 0],
                                            [1, 0, 0],
                                            [-1, 0, 0],
                                            [0, 1, 0],
                                            [0, -1, 0],
                                            [0, 0, 1],
                                            [0, 0, -1],], dtype=torch.float32)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                for t in translation_offsets:
                    translated = crystal.X[j] + t @ crystal.L
                    distance = torch.norm(crystal.X[i] - translated)
                    if distance <= cutoff and distance < edge_distance[i, j]:
                        edge_distance[i, j] = distance
                        edge_distance[j, i] = distance

        return cls(nodes, edge_distance, config)
    
    @classmethod
    def init_from_augmented_crystal(cls, original_crystal, augmented_crystals, config=3.0):
        """
        Initializes a CrystalGraph from an original crystal and a list of its augmented versions.
        Computes standard edges via init_from_crystal and adds displacement as self-loop edges.
        """
        # Step 1: Get normal edge matrix from the original
        base_graph = cls.init_from_crystal(original_crystal, config)
        nodes = base_graph.nodes
        edge_distance = base_graph.edges.clone()  # clone to avoid modifying the original

        num_nodes = nodes.shape[0]

        # Step 2: Compute average displacement for each atom
        displacements = torch.zeros(num_nodes)
        for aug in augmented_crystals:
            displacements.append(torch.norm(aug.X - original_crystal.X, dim=1).unsqueeze(0))
        displacement = torch.cat(displacements, dim=0)  # shape: (n_augments, num_nodes)
        # mean_disp = displacement.mean(dim=0)
        # var_disp = displacement.var(dim=0)

        # Step 3: Insert displacement as diagonal (self-loop) edges
        for i in range(num_nodes):
            edge_distance[i, i] = displacement[i]

        return cls(nodes=nodes, edges=edge_distance, config=config)



    def embed_graph(self, atom_feature_dim, hidden_dim, num_rbf_kernels):
        """
        Generates node and edge embeddings using dynamically set parameters.

        Args:
            atom_feature_dim (int): Dimension of input atomic feature vectors.
            hidden_dim (int): Dimension of transformed node and edge features.
            num_rbf_kernels (int): Number of RBF kernels for distance expansion.

        Returns:
            f_star (torch.Tensor): Node feature matrix.
            e_h (torch.Tensor): Edge feature matrix.
        """
        if self.nodes is None or self.edges is None:
            raise ValueError("Graph has not been initialized. Use init_from_crystal() first.")

        atom_embedding = CGCNN(atom_feature_dim, hidden_dim)
        edge_transform = EdgeTransformation(num_rbf_kernels, hidden_dim)

        f_star = atom_embedding(self.nodes)
        e_h = edge_transform(self.edges)
        return f_star, e_h

    @staticmethod
    def create_batch(crystals):
        """Creates a batch of CrystalGraph objects from a list of crystals."""
        return [CrystalGraph.init_from_crystal(crystal) for crystal in crystals]