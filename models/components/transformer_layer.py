import torch
import torch.nn as nn
from models.scripts.attention_function import AttentionFunction

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, head_size):
        super(TransformerLayer, self).__init__()
        self.heads = nn.ModuleList([AttentionFunction(hidden_dim) for _ in range(head_size)])
        # Create a separate Linear layer for each head for the LN(V) in aggregation
        feature_dim = 5 * hidden_dim # the dimension calculated from the operations that lead to V'
        self.linears = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(head_size)])
    
    def apply_linear_transform(self, value_matrix):
        """
        Applies a separate linear transformation to each head's value.
        """
        transformed_values = [
            self.linears[i](value_matrix[i]) for i in range(len(self.linears))
        ]
        
        return torch.stack(transformed_values, dim=0)  # Stack back to original shape
    
    def transform(self, atom_features, edge_features):
        """
        Uses attention and aggregation function to calculate
        the m features vector for each atom in the crystals
        """
        crystalEdgeList = []
        crystalmList = []
        i = 0

        # Create a dictionary for each crystal where keys are the atoms and values are every edge to the atom
        for crystal in edge_features:
            edgeDictionary = {}
            for edge in crystal:
                key, *value = edge
                if key not in edgeDictionary:
                    edgeDictionary[key] = []
                edgeDictionary[key].append(edge)
            crystalEdgeList.append(edgeDictionary)
        
        # Go through each crystal and calculate the m features vector for each atom in the crystal
        for crystal in atom_features:
            mList = []
            
            for atom in crystal:
                # Attention
                batched_edges = [torch.stack(edges, dim=0) for edges in zip(*crystalEdgeList[i][atom])] # Badge the edge with its neighbours
                alpha_matrix, value_matrix = self.attention(batched_edges)
                
                # Aggregation
                transformed_value_matrix = self.apply_linear_transform(value_matrix)
                weighted_values = alpha_matrix * transformed_value_matrix  # Shape: (N, head_size, batch_size, 1)
                summed_over_heads = torch.sum(weighted_values, dim=1)  # (N, batch_size, 1)
                m_i = torch.sum(summed_over_heads, dim=0)  # (batch_size, 1)
                mList.append(m_i)
            
            crystalmList.append(mList)
            i += 1
        
        return crystalmList

    def attention(self, edge):
        
        alpha_list = []
        value_list = []

        for head in self.heads:
            alpha, value = head(edge)
            alpha_list.append(alpha)
            value_list.append(value)

        alpha_matrix = torch.stack(alpha_list, dim=0)  # Stack into shape (head_size, batch_size, 1)
        value_matrix = torch.stack(value_list, dim=0)  # Stack into shape (head_size, batch_size, 1)

        return alpha_matrix, value_matrix