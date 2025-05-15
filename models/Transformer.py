import torch
import torch.nn as nn

from models.components.readout import Readout
from models.components.transformer_layer import TransformerLayer
from models.scripts.graph_embedding import GraphEmbedder

class Transformer(nn.Module):
    """ The transformer model for predicting crystal properties. """
    def __init__(self, atom_feature_dim=92, hidden_dim=120, num_rbf_kernels=50, min_rbf_dist=0.0, max_rbf_dist=8.0, num_heads=4, gradient=20): 
        super(Transformer, self).__init__()
        """
        Initialize the transformer model.
        
        Args:
            atom_feature_dim (int): The expected number of known features about each atom.
            hidden_dim (int): Dimension of hidden layers.
            num_rbf_kernels (int): Number of radial basis function (RBF) kernels for features.
            min_rbf_dist (float): Minimum distance for RBF kernel.
            max_rbf_dist (float): Maximum distance for RBF kernel.
            num_heads (int): Number of attention heads to use in the transformer layer.
            gradient (float): A value between 1 and 0, how to much attention should be payed to invalid edges.
                If 0 then they weigh the same, if 1 and invalid edges are ignored.
        """
        # Graph embedding layer for atom and edge features
        self.graph_embedder = GraphEmbedder(atom_feature_dim=atom_feature_dim, hidden_dim=hidden_dim, num_rbf_kernels=num_rbf_kernels, min_rbf_dist=min_rbf_dist, max_rbf_dist=max_rbf_dist) 
        # Transformer layer for processing node features using attention
        self.transformer_layer = TransformerLayer(hidden_dim=hidden_dim, head_size=num_heads, gradient=gradient)
        # Assuming a regression task for property prediction
        self.readout = Readout(hidden_dim=hidden_dim, output_dim=1) 

    def forward(self, graphs, chunk_size: int = 32):
        """
        Predict the property of a single crystal structure.

        Args:
            graphs: The list of graphs that were generated for a crystal.
            chunk_size (int): Number of nodes to process in parallel.
        
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
            edges = edges.detach()                           # Detach edges to avoid gradient flow
            edges[:, 5, 0] += offset                         # ❶ shift destination index for concatenation
            atom_blocks.append(atoms)
            edge_blocks.append(edges)
            node_ranges.append((offset, offset + atoms.size(0))) # Track node ranges
            offset += atoms.size(0)

        atom_all  = torch.cat(atom_blocks, 0)                # Concatenate all node embeddings [ΣNi, D]
        edge_all  = torch.cat(edge_blocks, 0)                # Concatenate all edge embeddings [ΣEi, 6, D]

        # ---- step 3: run the transformer once, still chunked ----
        m_all = torch.zeros_like(atom_all)                   # Initialize output node embeddings
        for start in range(0, atom_all.size(0), chunk_size):
            end = min(start + chunk_size, atom_all.size(0))
            node_chunk = atom_all[start:end]                 # Chunk of nodes [M, D]
            mask       = (edge_all[:, 5, 0] >= start) & (edge_all[:, 5, 0] < end) # Select edges in the chunk
            edge_chunk = edge_all[mask].clone()
            edge_chunk[:, 5, 0] -= start                     # localise edge indices
            m_all[start:end] = self.transformer_layer(node_chunk, edge_chunk)     # Process the chunk
        
        # ---- step 4: per-graph read-out ----
        predicted_properties, predicted_latents = [], []
        for (beg, end), _g in zip(node_ranges, graphs):
            m_i   = m_all[beg:end]                # Node embeddings for the current graph [Ni, D]
            atoms = atom_all[beg:end]             # Original node features
            y, F  = self.readout(m_i, atoms)      # Predict property and latent features
            predicted_properties.append(y)
            predicted_latents.append(F)
        
        return torch.stack(predicted_properties), torch.stack(predicted_latents), graphs
    
    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device (torch.device): Target device (CPU or GPU). """
        self.graph_embedder = self.graph_embedder.to(device)
        self.transformer_layer = self.transformer_layer.to(device)
        self.readout = self.readout.to(device)
        return self