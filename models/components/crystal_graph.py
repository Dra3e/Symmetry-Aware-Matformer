import torch
import torch.nn as nn
from models.scripts.graph_embedding import CGCNN, EdgeTransformation

class CrystalGraph(nn.Module):
    """Graph representation of a crystal structure from models.components.crystal.py"""

    def __init__(self, nodes=None, edges=None, m_valid=1, config=3.0):
        """
        Initializes a CrystalGraph. Use `init_from_crystal()` to create an instance from a crystal.

        Args:
            nodes (torch.tensor): The nodes of the graph
            edges (torch.tensor): The edges of the graph, with shape edge_count X 5, where edge[0] and edge[1] are to and from nodes
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            config (float): Cutoff distance for edge connections.
        """
        super().__init__()
        self.nodes = nodes  # Atomic feature matrix
        self.edges = edges  # Edge distance matrix
        self.config = config # Cutoff configuration
        self.m_valid = m_valid # Validity marker for edges

        # Ensure edges are a proper tensor and match the device of nodes
        if isinstance(edges, torch.Tensor):
            self.edges = edges.to(nodes.device)
        else:
            self.edges = torch.tensor(edges, dtype=torch.float32, device=nodes.device)

    @classmethod
    def init_from_crystal(cls, crystal, translation_vectors, m_valid=1, config=3.0):
        """
        Creates a CrystalGraph instance from a given crystal structure.

        Args:
            crystal (models.components.Crystal): A crystal object containing atomic information.
            translation_vectors (torch.tensor): Tensor of shape (n, 3) containing translation vectors for the crystal.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            config (float): Cutoff distance for edge connections.

        Returns:
            CrystalGraph instance.
        """
        nodes = crystal.A # Atomic feature matrix
        num_nodes = nodes.shape[0]
        edges = []

        # --------------------------------------------------
        # 1) Edges inside the unit cell
        # --------------------------------------------------
        for i in range(num_nodes):
            for j in range(i, num_nodes):                       # i ≤ j
                distance = torch.norm(crystal.X[i] - crystal.X[j])
                if distance <= config:
                    edges.append([i, j, distance.item(), 0, m_valid, j])  # dest=j
                    if i != j:                                           # undirected
                        edges.append([j, i, distance.item(), 0, m_valid, i])

        # --------------------------------------------------
        # 2) Edges for periodicity, if any
        # --------------------------------------------------
        for t in translation_vectors:              # may be an empty list
            vec = t @ crystal.L                    # 3‑vector
            for i in range(num_nodes):
                for j in range(num_nodes):
                    translated = crystal.X[j] + vec
                    distance   = torch.norm(crystal.X[i] - translated)
                    if distance <= config:
                        edges.append([i, j, distance.item(), 0, m_valid, j])

        return cls(nodes=nodes, edges=edges, config=config)
    
    @classmethod
    def init_from_augmented_crystal(cls, original_crystal, augmented_crystals, translation_vectors, m_valid=1, config=3.0):
        """
        Creates a CrystalGraph with additional self-loop edges for atomic displacements.
        
        Args:
            original_crystal (models.components.Crystal): The original unaugmented crystal structure.
            augmented_crystals: A list of augmented versions of the crystal.
            translation_vectors: Tensor of shape (n, 3) containing translation vectors for the crystal.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            config (float): Cutoff distance for edge connections.
        
        Returns:
            CrystalGraph instance with augmented self-loops.
        """
        base_graph = cls.init_from_crystal(original_crystal, translation_vectors, config, m_valid)
        nodes = base_graph.nodes
        edges = base_graph.edges.tolist()  # Ensure it's list-compatible to add more rows
        num_nodes = nodes.shape[0]

        for aug in augmented_crystals:
            if aug.X.shape[0] != original_crystal.X.shape[0]:
                raise ValueError(f"Mismatch in number of atoms: original={original_crystal.X.shape[0]}, aug={aug.X.shape[0]}")
            displacements = torch.norm(aug.X - original_crystal.X, dim=1) # Compute atomic displacements
            for i in range(num_nodes):
                edges.append([i, i, displacements[i].item(), 1, m_valid, i]) # Self-loop for augmentation tracking

        return cls(nodes=nodes, edges=edges, config=config)

    @staticmethod
    def create_batch(crystals, translation_vectors, m_valid=1):
        """
        Creates a batch of CrystalGraph objects from a list of crystals.

        Args:
            crystals: List of crystal structures.
            translation_vectors: Tensor of shape (n, 3) containing translation vectors for the crystal.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
        
        Returns:
            List of CrystalGraph instances.
        """
        return [CrystalGraph.init_from_crystal(crystal, translation_vectors, m_valid) for crystal in crystals]
    
    def to(self, device):
        self.nodes = self.nodes.to(device)
        self.edges = self.edges.to(device)
        return self