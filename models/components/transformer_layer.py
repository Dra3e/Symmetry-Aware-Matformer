import torch
import torch.nn as nn
from models.scripts.attention_function import AttentionFunction
from models.scripts.aggregation import Aggregation

class TransformerLayer(nn.Module):
    def __init__(self, edge, head_size):
        super(TransformerLayer, self).__init__()
        self.heads = nn.ModuleList([AttentionFunction(edge) for _ in range(head_size)])
        # Create a separate Linear layer for each head for the LN(V) in aggregation
        self.linears = nn.ModuleList([nn.Linear(1, 1) for _ in range(head_size)])
    
    def apply_linear_transform(self, value_matrix):
        """
        Applies a separate linear transformation to each head's value.
        """
        transformed_values = []
        for i in range(len(self.linears)):
            transformed_value = self.linears[i](value_matrix[i])  # Apply LN to each head separately
            transformed_values.append(transformed_value)
        
        return torch.stack(transformed_values, dim=0)  # Stack back to original shape
    
    def attention_batch(edges):
        for c in range(len(edges)):
            for e in range(len(edges[c])):
                attention(self, edges[c][e])

    def attention(self, edge):
        alpha_list = []
        value_list = []

        for head in self.heads:
            alpha, value = head(edge)
            alpha_list.append(alpha)
            value_list.append(value)

        alpha_matrix = torch.stack(alpha_list, dim=0)  # Stack into shape (head_size, batch_size, 1)
        value_matrix = torch.stack(value_list, dim=0)  # Stack into shape (head_size, batch_size, 1)
        transformed_value_matrix = self.apply_linear_transform(value_matrix)

        return alpha_matrix, transformed_value_matrix

    def aggregation(alpha, V):
        """
        alpha: list of Attention matrixes
        V: list of Value matrixes

        Output: Aggregated features (batch_size x 1)
        """
        # Stack lists into tensors
        alphas = torch.stack(alpha, dim=0)  # Shape: (N, head_size, batch_size, 1)
        values = torch.stack(V, dim=0)  # Shape: (N, head_size, batch_size, 1)

        # Perform element-wise multiplication
        weighted_values = alphas * values  # Shape: (N, head_size, batch_size, 1)

        # compute summation
        summed_over_heads = torch.sum(weighted_values, dim=1)  # (N, batch_size, 1)
        m_i = torch.sum(summed_over_heads, dim=0)  # (batch_size, 1)

        return m_i