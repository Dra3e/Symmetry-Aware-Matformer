import torch
import torch.nn as nn
from models.components.readout import Readout
from models.components.transformer_layer import TransformerLayer
from models.scripts.graph_embedding import GraphEmbedder

class Transformer(nn.Module):
    """ The transformer model for predicting crystal properties. """
    def __init__(self, atom_feature_dim=92, hidden_dim=120, num_rbf_kernels=50, num_heads=4, gradient=10): 
        super(Transformer, self).__init__()
        """
        Initialize the transformer model.
        
        Args:
            atom_feature_dim (int): The expected number of known features about each atom.
            hidden_dim (int): Dimension of hidden layers.
            num_rbf_kernels (int): Number of radial basis function kernels.
            num_heads (int): Number of attention heads to use in the transformer layer.
            gradient (float): A value between 1 and 0, how to much attention should be payed to invalid edges.
                If 0 then they weigh the same, if 1 and invalid edges are ignored.
        """
        self.graph_embedder = GraphEmbedder(atom_feature_dim, hidden_dim, num_rbf_kernels) # Graph embedding layer for atom and edge features
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, gradient) # Transformer layer for processing node features
        self.readout = Readout(hidden_dim, 1) # Assuming a regression task for property prediction

    def forward(self, graphs, chunk_size: int = 32):
        """
        Predict the property of a single crystal structure.

        Args:
            graphs: The list of graphs that were generated for a crystal.
            chunk_size (int): The size of the chunks to process the graph in parallel.
        
        Returns:
            A list of predicted properties for the crystal.
            A list of predicted latents of atom features for the crystal.
            The graphs generated for the crystal.
        """

        # ---- step 1-2: embed every graph and concatenate ----
        atom_blocks, edge_blocks, node_ranges = [], [], []   # keep (start, end) for every graph
        offset = 0
        for g in graphs:
            atoms, edges = self.graph_embedder(g)            # [Ni, D] / [Ei, 6, D]
            edges = edges.clone()
            edges[:, 5, 0] += offset                         # ❶ shift destination index
            atom_blocks.append(atoms)
            edge_blocks.append(edges)
            node_ranges.append((offset, offset + atoms.size(0)))
            offset += atoms.size(0)

        atom_all  = torch.cat(atom_blocks, 0)                # [ΣNi, D]
        edge_all  = torch.cat(edge_blocks, 0)                # [ΣEi, 6, D]

        # ---- step 3: run the transformer once, still chunked ----
        m_all = torch.zeros_like(atom_all)
        for start in range(0, atom_all.size(0), chunk_size):
            end = min(start + chunk_size, atom_all.size(0))
            node_chunk = atom_all[start:end]                 # [M, D]
            mask       = (edge_all[:, 5, 0] >= start) & (edge_all[:, 5, 0] < end)
            edge_chunk = edge_all[mask].clone()
            edge_chunk[:, 5, 0] -= start                     # localise indices
            m_all[start:end] = self.transformer_layer(node_chunk, edge_chunk)

        # ---- step 4: per-graph read-out ----
        predicted_properties, predicted_latents = [], []
        for (beg, end), _g in zip(node_ranges, graphs):
            m_i   = m_all[beg:end]
            atoms = atom_all[beg:end]
            y, F  = self.readout(m_i, atoms)
            predicted_properties.append(y)
            predicted_latents.append(F)

        return torch.stack(predicted_properties), torch.stack(predicted_latents), graphs
    
    def to(self, device):
        self.graph_embedder = self.graph_embedder.to(device)
        self.transformer_layer = self.transformer_layer.to(device)
        self.readout = self.readout.to(device)
        self.invalid_augmenter = self.invalid_augmenter.to(device)
        return self