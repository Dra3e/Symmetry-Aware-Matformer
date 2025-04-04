import torch
import torch.nn as nn
from models.components.crystal_graph import CrystalGraph
from models.components.readout import Readout
from models.components.transformer_layer import TransformerLayer
from models.scripts.graph_embedding import GraphEmbedder

class Transformer(nn.Module):
    """ The transformer model for predicting crystal properties. """
    def __init__(self, atom_feature_dim=92, hidden_dim=120, num_rbf_kernels=50, num_heads=4, gradient=10): 
        super(Transformer, self).__init__()
        print("\n" + "="*50)
        print(f"Initializing Transformer with:")
        print(f"atom_feature_dim={atom_feature_dim}, hidden_dim={hidden_dim}")
        print(f"num_rbf_kernels={num_rbf_kernels}, num_heads={num_heads}, gradient={gradient}")
        print("="*50 + "\n")
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

    def forward(self, C, chunk_size=32):
        print("\n=== CRYSTAL PROCESSING ===")
        print(f"Input crystal contains {len(C.A)} atoms")
        """
        Predict the property of a single crystal structure.

        Args:
            C (models.components.Crystal): The crystal structure to predict the property for.
            chunk_size (int): The size of the chunks to process the graph in parallel.
        
        Returns:
            A list of predicted properties for the crystal.
        """
        # Generate graphs from the crystal
        print("\n=== SYMMETRY OPERATIONS ===")
        valid_symmetry_operations = C.get_symmetry_operations() # Get symmetry operations for the crystal
        print(f"Found {len(valid_symmetry_operations)} valid symmetry operations")
        
        # Graph generation
        print("\n=== GRAPH GENERATION ===")
        augmented_valid_crystals = C.apply_augmentations(valid_symmetry_operations) # Apply augmentations to the crystal
        print(f"Generated {len(augmented_valid_crystals)} augmented crystals")

        valid_crystal_graph = CrystalGraph.init_from_augmented_crystal(C, augmented_valid_crystals, 1) # Initialize crystal graphs from augmented crystals
        print(f"Valid graph has {len(valid_crystal_graph.nodes)} nodes and {len(valid_crystal_graph.edges)} edges")

        invalid_symmetry_operations = [] 
        augmented_invalid_crystals = C.apply_augmentations(invalid_symmetry_operations) # Apply augmentations to the crystal
        invalid_crystal_graphs = CrystalGraph.create_batch(augmented_invalid_crystals, 0) # Initialize crystal graphs from invalid crystals
        # print(f"Invalid graphs: {len(invalid_crystal_graphs)} with {len(invalid_crystal_graphs[0].edges)} edges each")

        # Combine valid and invalid graphs
        graphs = invalid_crystal_graphs + [valid_crystal_graph] # Combine valid and invalid graphs
        predicted_properties = []  # List to store predicted properties

        for graph in graphs:
            print(f"\nProcessing graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            # Step 1: Feature Initialization
            print("\n--- Feature Embedding ---")
            atom_features, edge_features = self.graph_embedder(graph)
            print(f"atom_features shape: {atom_features.shape}")
            print(f"edge_features number of columns: {edge_features.shape}")
            print(f"edge_features column shapes: {len(edge_features[:, 0])}, {len(edge_features[:, 1])}, {len(edge_features[:, 2])}, {len(edge_features[:, 3])}, {len(edge_features[:, 4])}, {len(edge_features[:, 5])}")

            # Step 2: Process Nodes and Edges in Chunks
            print(f"\n--- Chunk Processing (size={chunk_size}) ---")
            m_features_list = []
            num_nodes = atom_features.shape[0] # Number of nodes in the graph
            for start in range(0, num_nodes, chunk_size): 
                end = min(start + chunk_size, num_nodes) # End index for chunking
                node_chunk = atom_features[start:end] # Chunk of node features
                print(f"node_chunk shape: {node_chunk.shape}")

                # Edge filtering
                print(f"edge_features.attr_2 shape: {edge_features[:, 5, 0].shape}")
                related_edges = (edge_features[:, 5, 0] >= start) & (edge_features[:, 5, 0] < end) # Get related edges for the chunk
                print(f"Found {len(related_edges)} related edges")

                # Filter each field in edge_features (Transformer processing)
                edge_chunk = edge_features[related_edges]  # Chunk of edge features

                if(start == 0):
                    print(f"edge_chunk column shapes: {edge_chunk[:, 0].shape}, {edge_chunk[:, 1].shape}, {edge_chunk[:, 2].shape}, {edge_chunk[:, 3].shape}, {edge_chunk[:, 4].shape}, {edge_chunk[:, 5].shape}")
                # Step 3: Calculate attention score and aggreate
                m_chunk = self.transformer_layer.forward(node_chunk, edge_chunk)
                if(start == 0):
                    print(f"m_chunk shape: {m_chunk.shape}")
                    
                m_features_list.append(m_chunk) # Append the processed chunk to the list
                print(f"m_feature_list size: {len(m_features_list)}")

            # Step 4: Concatenate all chunked features
            print("--- Readout Phase ---")
            m_features = torch.cat(m_features_list, dim=0) # Concatenate all processed chunks
            print(f"Concatenated m_features shape: {m_features.shape}") 

            # Step 5: Readout - Pooling and Prediction
            y_property = self.readout(m_features, atom_features) # Readout layer for pooling and prediction
            print(f"Predicted property: {y_property.item() if y_property.dim() == 0 else y_property.shape}")
            predicted_properties.append(y_property) # Append the predicted property to the list

        print("\n=== PREDICTION SUMMARY ===")
        print(f"Generated {len(predicted_properties)} predictions")
        print("="*50 + "\n")

        return predicted_properties, graphs