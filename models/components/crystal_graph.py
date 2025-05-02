import torch
import torch.nn as nn

from models.scripts.crystal_graph_vectorize import _pair_edges

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
        if not isinstance(translation_vectors, torch.Tensor): # turns translation_vectors from list to tensor for matrix operations below
            translation_vectors = (
                torch.empty(0, 3, dtype=crystal.X.dtype, device=crystal.X.device)
                if len(translation_vectors) == 0
                else torch.stack(list(translation_vectors), dim=0).to(crystal.X)
            )
        base_edges = _pair_edges(crystal.X, config, m_valid, m_sym=0)

        if translation_vectors.numel() == 0:           # no periodic edges needed
            return cls(nodes=nodes, edges=base_edges, m_valid=m_valid, config=config)

        # ---------- periodic edges (broadcast) ----------
        # crystal.X : (N,3)          translation_vectors : (T,3)
        translated = crystal.X.unsqueeze(0) + (translation_vectors @ crystal.L)[:, None, :]
        # translated : (T,N,3)   reshape to (T*N,3) for one call
        periodic_edges = []
        for t_idx in range(translated.size(0)):
            e = _pair_edges(translated[t_idx], config, m_valid, m_sym=0)
            # rewrite j-index so it still refers to original atom j
            e[:, 1] %= nodes.shape[0]
            e[:, 5] = e[:, 1]
            periodic_edges.append(e)
        periodic_edges = torch.cat(periodic_edges, dim=0)

        edges = torch.cat((base_edges, periodic_edges), dim=0)
        return cls(nodes=nodes, edges=edges, m_valid=m_valid, config=config)
    
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
        base = cls.init_from_crystal(original_crystal, translation_vectors, m_valid, config)           # uses vectorised ver.
        nodes, edges = base.nodes, base.edges

        # --------- self-loops for augmentations (vectorised) ----------
        if augmented_crystals:
            # Stack all displacements in one tensor: (K,N)
            displacements = torch.stack([torch.norm(aug.X - original_crystal.X, dim=1) for aug in augmented_crystals]) # (K, N)

            # For every augmented copy k, produce N self-loops i→i
            k_idx, atom_idx = torch.nonzero(torch.ones_like(displacements), as_tuple=True)
            disp_flat = displacements[k_idx, atom_idx]

            loop_edges = torch.stack(
                (atom_idx, atom_idx, disp_flat,
                torch.ones_like(atom_idx),          # edge_type = 1  (augmented)
                torch.full_like(atom_idx, m_valid),
                atom_idx), dim=1
            )
            edges = torch.cat((edges, loop_edges), dim=0)

        return cls(nodes=nodes, edges=edges, config=config, m_valid=m_valid)

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