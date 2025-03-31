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

    @classmethod
    def init_from_crystal(cls, crystal, m_valid=1, config=3.0):
        """
        Creates a CrystalGraph instance from a given crystal structure.

        Args:
            crystal (models.components.Crystal): A crystal object containing atomic information.
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
        translation_offsets = torch.tensor([
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=torch.float32)

        for i in range(num_nodes):
            for j in range(i, num_nodes): # include i==j to allow self-loops if needed
                for t in translation_offsets:
                    translated = crystal.X[j] + t @ crystal.L # Apply translation to get surrounding unit cells
                    distance = torch.norm(crystal.X[i] - translated) # Compute distance between atoms
                    if distance <= config: # Only keep edges within cutoff distance
                        edges.append([i, j, distance.item(), 0, m_valid])
                        if i != j:
                            edges.append([j, i, distance.item(), 0, m_valid])  # undirected edges

        return cls(nodes=nodes, edges=edges, config=config)
    
    @classmethod
    def init_from_augmented_crystal(cls, original_crystal, augmented_crystals, m_valid=1, config=3.0):
        """
        Creates a CrystalGraph with additional self-loop edges for atomic displacements.
        
        Args:
            original_crystal (models.components.Crystal): The original unaugmented crystal structure.
            augmented_crystals: A list of augmented versions of the crystal.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            config (float): Cutoff distance for edge connections.
        
        Returns:
            CrystalGraph instance with augmented self-loops.
        """
        base_graph = cls.init_from_crystal(original_crystal, config, m_valid)
        nodes = base_graph.nodes
        edges = base_graph.edges.copy()
        num_nodes = nodes.shape[0]

        for aug in augmented_crystals:
            displacements = torch.norm(aug.X - original_crystal.X, dim=1) # Compute atomic displacements
            for i in range(num_nodes):
                edges.append([i, i, displacements[i].item(), 1, m_valid]) # Self-loop for augmentation tracking

        return cls(nodes=nodes, edges=edges, config=config)

    @staticmethod
    def create_batch(crystals, m_valid=1):
        """
        Creates a batch of CrystalGraph objects from a list of crystals.

        Args:
            crystals: List of crystal structures.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
        
        Returns:
            List of CrystalGraph instances.
        """
        return [CrystalGraph.init_from_crystal(crystal, m_valid) for crystal in crystals]