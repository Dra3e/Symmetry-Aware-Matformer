import torch
import torch.nn as nn
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from models.scripts.graph_embedding import GraphEmbedder
from models.components.transformer_layer import TransformerLayer
from models.components.readout import Readout


class Transformer(nn.Module):
    def __init__(self, atom_feature_dim=92, hidden_dim=120, num_rbf_kernels=50, num_heads=4):
        super(Transformer, self).__init__()
        self.graph_embedder = GraphEmbedder(atom_feature_dim, hidden_dim, num_rbf_kernels)
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads)
        self.readout = Readout(hidden_dim, 1)

    def forward(self, C):
        # Input and Graph Construction
        valid_axes, valid_planes, valid_directions = C.get_symmetry_operations()
        augmented_valid_crystals = C.apply_augmentations(valid_axes, valid_planes, valid_directions)
        valid_crystal_graph = CrystalGraph.init_from_augmented_crystal(C, augmented_valid_crystals)
        
        invalid_axes, invalid_planes, invalid_directions = [], [], []
        augmented_invalid_crystals = C.apply_augmentations(invalid_axes, invalid_planes, invalid_directions)
        invalid_crystal_graphs = CrystalGraph.create_batch(augmented_invalid_crystals)

        # Feature Initialization
        graphs = invalid_crystal_graphs
        graphs.append(valid_crystal_graph)
        atom_features, edge_features = self.graph_embedder.batch_forward(graphs)

        # Transformer Layer
        # - Preparation Phase
        # - Attention Function
        # - Aggregation Function
        m_features = self.transformer_layer.transform(atom_features, edge_features)

        # Readout Layer
        # - Update Phase
        # - Global Pooling
        # - Predicted Property
        y_property = self.readout(m_features, atom_features)
        return y_property