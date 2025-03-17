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
        cutoff = config
        nodes = crystal.A  # Atomic feature matrix
        edges = cls.compute_edges_from_crystal(crystal, cutoff)  # Compute edges dynamically

        return cls(nodes=nodes, edges=edges, config=config)

    @staticmethod
    def compute_edges_from_crystal(crystal, cutoff):
        """
        Computes edges and edge attributes based on atomic positions.

        Args:
            X (torch.Tensor): Atomic position matrix.
            L (torch.Tensor): Lattice matrix.
            cutoff (float): Maximum bond distance to consider for edges.

        Returns:
            edge_distance (torch.Tensor): Tensor containing edge distances.
        """
        num_nodes = crystal.X.shape[0]
        edge_distance = torch.full((num_nodes, num_nodes), float('inf'))

        # Translation vectors for periodic unit cells
        translation_vectors = torch.tensor([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ], dtype=torch.float32) @ L

        # Extended unit cells
        extended_X = [crystal.X]
        for vec in translation_vectors:
            extended_X.append(crystal.X + vec.unsqueeze(0))
        extended_X = torch.cat(extended_X, dim=0)

        # Compute pairwise distances and build edges
        for i in range(num_nodes):
            for j in range(len(extended_X)):
                distance = torch.norm(X[i] - extended_X[j])
                if distance <= cutoff:
                    edge_distance[i, j % num_nodes] = distance

        return edge_distance

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

    @staticmethod
    def stitch_graphs(graphs):
        """
        Stitches multiple graphs into a single graph by combining adjacency matrices.

        Args:
            graphs (list of CrystalGraph): List of graphs to merge.

        Returns:
            stitched_graph (CrystalGraph): A combined graph.
        """
        if not graphs:
            return None

        stitched_nodes = graphs[0].nodes.clone()
        stitched_edges = graphs[0].edges.clone()

        for graph in graphs[1:]:
            num_existing_nodes = stitched_nodes.shape[0]
            num_new_nodes = graph.nodes.shape[0]
            new_size = num_existing_nodes + num_new_nodes

            expanded_edges = torch.full((new_size, new_size), float('inf'))
            expanded_edges[:num_existing_nodes, :num_existing_nodes] = stitched_edges
            expanded_edges[num_existing_nodes:, num_existing_nodes:] = graph.edges

            stitched_edges = expanded_edges
            stitched_nodes = torch.cat((stitched_nodes, graph.nodes), dim=0)

        return CrystalGraph(nodes=stitched_nodes, edges=stitched_edges)