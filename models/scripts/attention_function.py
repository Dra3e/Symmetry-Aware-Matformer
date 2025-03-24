import torch
import torch.nn as nn

class AttentionFunction(nn.Module):
    def __init__(self, f_i, f_j, e_ij):
        super(AttentionFunction, self).__init__()
        self.query = nn.Linear(f_i.shape[-1], f_i.shape[-1])
        self.key_i = nn.Linear(f_i.shape[-1], f_i.shape[-1])
        self.key_j = nn.Linear(f_j.shape[-1], f_j.shape[-1])
        self.value_i = nn.Linear(f_i.shape[-1], f_i.shape[-1])
        self.value_j = nn.Linear(f_j.shape[-1],f_j.shape[-1])
        self.edge = nn.Linear(e_ij.shape[-1], e_ij.shape[-1])

        self.layer_norm = nn.LayerNorm(f_i.shape[-1] * 3)  # 3 because we concatenate over three elements with f_i dimensions
        self.symmetry = nn.Linear(e_ij.shape[-1], e_ij.shape[-1], bias=False)

    def forward(self, f_i, f_j, e_ij, M_sym, M_valid):
        query = self.query(f_i)
        key_i = self.key_i(f_i)
        key_j = self.key_j(f_j)
        value_i = self.value_i(f_i)
        value_j = self.value_j(f_j)
        edge = self.edge(e_ij)

        query_C = torch.cat((query, query, query), dim=-1)
        key_C = torch.cat((key_i, key_j, edge), dim=-1)
        value_C = torch.cat((value_i, value_j, edge), dim=1)
        d = key_C.shape[-1]

        alpha_original = torch.sigmoid(self.layer_norm(torch.mul(query_C, key_C) / d))
        symmetry = 1 + self.symmetry(torch.mul(M_sym,e_ij))
        alpha = torch.mul(torch.mul(alpha_original, symmetry), M_valid)

        return alpha, value_C

# class AttentionFunction(nn.Module):
#     def __init__(self, f_i, f_j, e_ij):
#         super(AttentionFunction, self).__init__()
#         self.query = nn.Linear(f_i, f_i)
#         self.key_i = nn.Linear(f_i, f_i)
#         self.key_j = nn.Linear(f_j, f_j)
#         self.value_i = nn.Linear(f_i, f_i)
#         self.value_j = nn.Linear(f_j,f_j)
#         self.edge = nn.Linear(e_ij, e_ij)

#         self.layer_norm = nn.LayerNorm(f_i * 3)  # 3 because we concatenate over three elements with f_i dimensions
#         self.symmetry = nn.Linear(e_ij, e_ij, bias=False)

#     def forward(self, f_i_tensor, f_j_tensor, e_ij_tensor, M_sym, M_valid):
#         query = self.query(f_i_tensor)
#         key_i = self.key_i(f_i_tensor)
#         key_j = self.key_j(f_j_tensor)
#         value_i = self.value_i(f_i_tensor)
#         value_j = self.value_j(f_j_tensor)
#         edge = self.edge(e_ij_tensor)

#         query_C = torch.cat((query, query, query), dim=-1)
#         key_C = torch.cat((key_i, key_j, edge), dim=-1)
#         value_C = torch.cat((value_i, value_j, edge), dim=1)
#         d = key_C.shape[-1]

#         alpha_original = torch.sigmoid(self.layer_norm(torch.mul(query_C, key_C) / d))
#         symmetry = 1 + self.symmetry(torch.mul(M_sym,e_ij_tensor))
#         alpha = torch.mul(torch.mul(alpha_original, symmetry), M_valid)

#         return alpha, value_C