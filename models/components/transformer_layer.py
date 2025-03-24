import torch
import torch.nn as nn
from models.scripts.attention_function import AttentionFunction
from models.scripts.aggregation import Aggregation

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, f_i, f_j, e_ij, head_size):
        super(TransformerLayer, self).__init__()
        self.heads = nn.ModuleList([AttentionFunction(f_i, f_j, e_ij) for _ in range(head_size)])
        # self.attention_function = AttentionFunction.__init__(f_i, f_j, e_ij)
        self.aggregation_function = Aggregation(f_i, hidden_dim) 
    
    def foward(self, f_i, f_j, e_ij, M_sym, M_valid, neighbors, edges):
        alpha_list = []
        value_list = []

        for head in self.heads:
            alpha, value = head(f_i, f_j, e_ij, M_sym, M_valid)
            alpha_list.append(alpha)
            value_list.append(value)

        alpha_matrix = torch.stack(alpha_list, dim=0)  # Stack into shape (num_heads, batch, 1)
        value_matrix = torch.stack(value_list, dim=0)  # Stack into shape (num_heads, batch, 1)

        return self.aggregation_function.forward(neighbors, alpha_matrix, value_matrix, edges)

    # def attention(self, f_i_tensor, f_j_tensor, e_ij_tensor, M_sym, M_valid):
    #     return self.attention_function.forward(f_i_tensor, f_j_tensor, e_ij_tensor, M_sym, M_valid)

    # def aggregation(self, neighbors, alpha, V, edges):
    #     return self.aggregation_function.forward(neighbors, alpha, V, edges)