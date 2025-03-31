import torch
import torch.nn as nn
from models.components.crystal_graph import CrystalGraph
from models.components.readout import Readout
from models.components.transformer_layer import TransformerLayer
from models.scripts.graph_embedding import GraphEmbedder

class Transformer(nn.Module):
    """ The transformer model for predicting crystal properties. """
    def __init__(self, atom_feature_dim=92, hidden_dim=120, num_rbf_kernels=50, num_heads=4, gradient=10): 
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
        super(Transformer, self).__init__()
        self.graph_embedder = GraphEmbedder(atom_feature_dim, hidden_dim, num_rbf_kernels) # Graph embedding layer for atom and edge features
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads) # Transformer layer for processing node features
        self.readout = Readout(hidden_dim, 1) # Assuming a regression task for property prediction
        self.gradient = gradient # Gradient for the transformer model

    def forward(self, C, chunk_size=32):
        """
        Predict the property of a single crystal structure.

        Args:
            C (models.components.Crystal): The crystal structure to predict the property for.
            chunk_size (int): The size of the chunks to process the graph in parallel.
        
        Returns:
            A list of predicted properties for the crystal.
        """
        # Generate graphs from the crystal
        valid_symmetry_operations = C.get_symmetry_operations() # Get symmetry operations for the crystal
        augmented_valid_crystals = C.apply_augmentations(valid_symmetry_operations) # Apply augmentations to the crystal
        valid_crystal_graphs = CrystalGraph.init_from_augmented_crystal(C, augmented_valid_crystals, 1) # Initialize crystal graphs from augmented crystals

        invalid_symmetry_operations = [] 
        augmented_invalid_crystals = C.apply_augmentations(invalid_symmetry_operations) # Apply augmentations to the crystal
        invalid_crystal_graphs = CrystalGraph.create_batch(augmented_invalid_crystals, 0) # Initialize crystal graphs from invalid crystals

        # Combine valid and invalid graphs
        graphs = invalid_crystal_graphs + [valid_crystal_graphs] # Combine valid and invalid graphs
        predicted_properties = []  # List to store predicted properties

        for graph in graphs:
            # Step 1: Feature Initialization
            atom_features, edge_features = self.graph_embedder(graph)

            # Step 2: Process Nodes and Edges in Chunks
            m_features_list = []
            num_nodes = atom_features.shape[0] # Number of nodes in the graph
            for start in range(0, num_nodes, chunk_size): 
                end = min(start + chunk_size, num_nodes) # End index for chunking
                node_chunk = atom_features[start:end] # Chunk of node features

                related_edges = (edge_features[:, 0] >= start) & (edge_features[:, 0] < end) # Get related edges for the chunk
                edge_chunk = edge_features[related_edges]  # Chunk of edge features

                # Step 3: Calculate attention score and aggreate
                m_chunk = self.transformer_layer.forward(node_chunk, edge_chunk, self.gradient) # Process the chunk using transformer layer
                m_features_list.append(m_chunk) # Append the processed chunk to the list

            # Step 4: Concatenate all chunked features
            m_features = torch.cat(m_features_list, dim=0) # Concatenate all processed chunks

            # Step 5: Readout - Pooling and Prediction
            y_property = self.readout(m_features, atom_features) # Readout layer for pooling and prediction
            predicted_properties.append(y_property) # Append the predicted property to the list

        return predicted_properties