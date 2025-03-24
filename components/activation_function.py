import torch
import torch.nn as nn

class ActivationFunction(nn.Module):
    def __init__(self, fi, fj, eij, M_sym_dim):
        super(ActivationFunction, self).__init__()
        self.query = nn.Linear(fi, fi)
        self.key_i = nn.Linear(fi, fi)
        self.key_j = nn.Linear(fj, fj)
        self.edge = nn.Linear(eij, eij)

        self.layer_norm = nn.LayerNorm(fi * 3)  # 3 because we concatenate over three elements with fi dimensions
        self.symmetry = nn.Linear(eij, eij, bias=False)

    def forward(self, fi_tensor, fj_tensor, eij_tensor, M_sym, M_val):
        query = self.query(fi_tensor)
        key_i = self.key_i(fi_tensor)
        key_j = self.key_j(fj_tensor)
        edge = self.edge(eij_tensor)

        query_C = torch.cat((query, query, query), dim=-1)
        key_C = torch.cat((key_i, key_j, edge), dim=-1)
        d = key_C.shape[-1]

        alpha_original = torch.sigmoid(self.layer_norm(torch.mul(query_C, key_C) / d))
        symmetry = 1 + self.symmetry(torch.mul(M_sym,eij_tensor))
        alpha = torch.mul(torch.mul(alpha_original, symmetry), M_val)

        return alpha


# import torch
# import torch.nn as nn

# def ActivationFunction(fi, fj, eij, M_sym, M_val):
#     query = nn.Linear(fi, fi)
#     key_i = nn.Linear(fi, fi)
#     key_j = nn.Linear(fj,fj)
#     edge = nn.Linear(eij,eij)

#     query_C = torch.cat((query, query, query), dim=1)
#     key_C = torch.cat((key_i, key_j, edge), dim=1)
#     d = key_C.shape[-1]
#     layer_norm = nn.LayerNorm(3)

#     alpha_original = torch.sigmoid(layer_norm(torch.mul(query_C, key_C) / d))
#     symmetry = 1 + nn.Linear(M_sym * eij, M_sym * eij, False)
#     alpha = torch.mul(torch.mul(alpha_original, symmetry), M_val)

#     return alpha