import torch
import torch.nn as nn
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from models.components.transformer_layer import transformer_layer
from models.components.readout import readout


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, C):
        # Input and Graph Construction
        valid_axes, valid_planes, valid_directions = C.get_symmetry_operations()
        augmented_valid_crystals = C.apply_augmentations(valid_axes, valid_planes, valid_directions)
        valid_crystal_graph = CrystalGraph.init_from_augmented_crystal(C, augmented_valid_crystals)
        
        invalid_axes, invalid_planes, invalid_directions = [], [], []
        augmented_invalid_crystals = C.apply_augmentations(invalid_axes, invalid_planes, invalid_directions)
        invalid_crystal_graphs = CrystalGraph.create_batch(augmented_invalid_crystals)

        # Feature Initialization
        graphs = valid_crystal_graph + invalid_crystal_graphs

        f_star, e_h = CrystalGraph.embed_graph_batch(graphs, C.A[0], C.A[0], 8)

        # Preparation Phase
        # Attention Function
        alpha_matrix_list, value_matrix_list = transformer_layer.attention(e_h[0], e_h[1], e_h[2], e_h[3], 1)

        # Aggregation Function
        m_i = transformer_layer.aggregation(alpha_matrix_list, value_matrix_list)

        # Update Phase
        # Global Pooling
        # Predicted Property
        y_property = readout(m_i, f_star)
        return y_property