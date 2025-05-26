import torch
import torch.nn as nn

from models.scripts.crystal_graph_vectorize import _pair_edges

class CrystalGraph(nn.Module):
    """
    Graph representation of a crystal structure from models.components.crystal.py

    Attributes:
        nodes (torch.tensor): The nodes of the graph, representing atomic features.
        edges (torch.tensor): The edges of the graph, representing connections between nodes.
        m_valid (int): Binary value to determine if the graph is a valid representation of a crystal.
            1: if true
            0: if false
        cutoff (float): Cutoff distance for edge connections.
    """

    def __init__(self, nodes=None, edges=None, m_valid=1, cutoff=3.0):
        """
        Initializes a CrystalGraph. Use `init_from_crystal()` to create an instance from a crystal.

        Args:
            nodes (torch.tensor): The nodes of the graph
            edges (torch.tensor): The edges of the graph, with shape edge_count X 6, where edge[0] and edge[1] are to and from nodes
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            cutoff (float): Cutoff distance for edge connections.
        """
        super().__init__()
        self.nodes = nodes      # Atomic feature matrix
        self.edges = edges      # Edge distance matrix
        self.cutoff = cutoff    # Cutoff cutoffuration
        self.m_valid = m_valid  # Validity marker for edges

        # Ensure edges are a proper tensor and match the device of nodes
        if isinstance(edges, torch.Tensor):
            self.edges = edges.to(nodes.device)
        else:
            self.edges = torch.tensor(edges, dtype=torch.float32, device=nodes.device)

    @classmethod
    def init_from_crystal(cls, crystal, m_valid=1, cutoff=3.0):
        """
        Creates a CrystalGraph instance from a given crystal structure.

        Args:
            crystal (models.components.Crystal): A crystal object containing atomic information.
            translation_vectors (torch.tensor): (n, 3) matrix containing translation vectors for the crystal.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            cutoff (float): Cutoff distance for edge connections.

        Returns:
            CrystalGraph instance.
        """
        nodes = crystal.A   # Atomic feature matrix
        base_edges = _pair_edges(crystal.P, cutoff, m_valid, m_sym=0) # Genereate distance based edges for the internal unit cell.
        
        # Create distance based periodic edges across unit cell boundaries.
        translation_vectors = crystal.get_translation_vectors_to_neighbour_cells(cutoff) # (T, 3) matrix get the translations to create neighboring unit cells for periodicity.
        translated = crystal.P.unsqueeze(0) + (translation_vectors @ crystal.L)[:, None, :] # Create the neighboring unit cells.  
        # Create distance based edges for the neighboring unit cells.
        periodic_edges = []
        for t_idx in range(translated.size(0)):
            e = _pair_edges(translated[t_idx], cutoff, m_valid, m_sym=0)
            # Create returning edge (undirected graph)
            e[:, 1] %= nodes.shape[0]
            e[:, 5] = e[:, 1]
            periodic_edges.append(e)
        periodic_edges = torch.cat(periodic_edges, dim=0)

        edges = torch.cat((base_edges, periodic_edges), dim=0)
        return cls(nodes=nodes, edges=edges, m_valid=m_valid, cutoff=cutoff)
    
    @classmethod
    def init_from_augmented_crystal(cls, original_crystal, augmented_crystals, m_valid=1, cutoff=3.0):
        """
        Creates a CrystalGraph with additional self-directed edges for atomic displacements during augmentation.
        
        Args:
            original_crystal (models.components.Crystal): The original unaugmented crystal structure.
            augmented_crystals: A list of augmented versions of the crystal.
            translation_vectors: (n, 3) matrix containing translation vectors for the crystal.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
            cutoff (float): Cutoff distance for edge connections.
        
        Returns:
            CrystalGraph instance.
        """
        base = cls.init_from_crystal(original_crystal, m_valid, cutoff) # Create the base graph from the original crystal.
        nodes, edges = base.nodes, base.edges

        if augmented_crystals:
            # Calculate displacement (euclidean distance) between the atom positions of the original crystal each augmented crystal.
            displacements = torch.stack([torch.norm(aug.P - original_crystal.P, dim=1) for aug in augmented_crystals]) 

            # Produce a self-directed edge for each atom in each augmented crystal.
            augmented_crystal_i, atom_index = torch.nonzero(torch.ones_like(displacements), as_tuple=True) # Get the indices of all atoms and augmented crystals.
            disp_flat = displacements[augmented_crystal_i, atom_index] # Get the displacmenet
            self_directed_edges = torch.stack(
                (atom_index, atom_index, disp_flat,
                torch.ones_like(atom_index),          # Creates a matrix with the same shape as atom_idx filled with 1s. edge_type = 1  (augmented)
                torch.full_like(atom_index, m_valid), # Creates a matrix with the same shape as atom_idx filled with m_valid.
                atom_index), dim=1
            )
            edges = torch.cat((edges, self_directed_edges), dim=0)

        return cls(nodes=nodes, edges=edges, cutoff=cutoff, m_valid=m_valid)

    @staticmethod
    def create_batch(crystals, m_valid=1):
        """
        Creates a list of CrystalGraph instances from a list of crystals.
        Each crystal becomes one crystal graph.

        Args:
            crystals: List of crystal structures.
            m_valid (int): Value to determine if the graph is a valid representation of a crystal
                1: if true
                0: if false
        
        Returns:
            List of CrystalGraph instances.
        """
        return [CrystalGraph.init_from_crystal(crystal, m_valid) for crystal in crystals]
    
    def to(self, device):
        """
        Moves the crystal graph to the specified device (CPU or GPU).

        Args:
            device: The target device (e.g., 'cpu' or 'cuda').

        Returns:
            CrystalGraph: The crystal graph moved to the specified device.
        """
        self.nodes = self.nodes.to(device)
        self.edges = self.edges.to(device)
        return self