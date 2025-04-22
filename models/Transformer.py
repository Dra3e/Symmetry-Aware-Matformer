import torch
import torch.nn as nn
from models.components.crystal_graph import CrystalGraph
from models.components.readout import Readout
from models.components.transformer_layer import TransformerLayer
from models.scripts.graph_embedding import GraphEmbedder
from models.scripts.invalid_augmentations import InvalidAugmenter

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
        self.invalid_augmenter = InvalidAugmenter()

    def forward(self, C, chunk_size=32):
        """
        Predict the property of a single crystal structure.

        Args:
            C (models.components.Crystal): The crystal structure to predict the property for.
            chunk_size (int): The size of the chunks to process the graph in parallel.
        
        Returns:
            A list of predicted properties for the crystal.
        """
        valid_symmetry_operations, valid_translations = C.get_symmetry_operations() # Get symmetry operations for the crystal

        # Graph generation
        augmented_valid_crystals = C.apply_augmentations(valid_symmetry_operations) # Apply augmentations to the crystal

        valid_crystal_graph = CrystalGraph.init_from_augmented_crystal(C, augmented_valid_crystals, valid_translations, 1) # Initialize crystal graphs from augmented crystals

        # Invalid_symmetry_operations
        invalid_symmetry_operations = self.invalid_augmenter.forward(valid_symmetry_operations)
        augmented_invalid_crystals = C.apply_augmentations(invalid_symmetry_operations) # Apply augmentations to the crystal
        invalid_crystal_graphs = CrystalGraph.create_batch(augmented_invalid_crystals, valid_translations, 0) # Initialize crystal graphs from invalid crystals

        # Combine valid and invalid graphs
        graphs = invalid_crystal_graphs + [valid_crystal_graph] # Combine valid and invalid graphs
        predicted_properties = []  # List to store predicted properties
        predicted_F = []  # List to store predicted F

        for graph in graphs:
            # Step 1: Feature Initialization
            atom_features, edge_features = self.graph_embedder(graph) # Shapes: [N, hidden_dim] and [num_edges, 6, hidden_dim]

            # Step 2: Process Nodes and Edges in Chunks
            m_features_list = []
            num_nodes = atom_features.shape[0] # Number of nodes in the graph
            for start in range(0, num_nodes, chunk_size): 
                end = min(start + chunk_size, num_nodes) # End index for chunking
                node_chunk = atom_features[start:end] # Shape: [chunk_size, hidden_dim] Chunk of node features

                # Edge filtering
                related_edges = (edge_features[:, 5, 0] >= start) & (edge_features[:, 5, 0] < end) # Get related edges for the chunk

                # Filter each field in edge_features (Transformer processing)
                edge_chunk = edge_features[related_edges].clone()  # Chunk of edge features
                # subtract the chunk’s start position to make it 0…M‑1
                edge_chunk[:, 5, :] -= start 

                # Step 3: Calculate attention score and aggreate
                m_chunk = self.transformer_layer.forward(node_chunk, edge_chunk) # Shape: [chunk_size, hidden_dim] Processed chunk of node features
                    
                m_features_list.append(m_chunk) # Append the processed chunk to the list

            # Step 4: Concatenate all chunked features
            m_features = torch.cat(m_features_list, dim=0) # Concatenate all processed chunks

            # Step 5: Readout - Pooling and Prediction
            y_property, F_pooled = self.readout(m_features, atom_features) # Readout layer for pooling and prediction
            predicted_properties.append(y_property) # Append the predicted property to the list
            predicted_F.append(F_pooled) # Append the predicted F to the list

        return torch.stack(predicted_properties), torch.stack(predicted_F), graphs