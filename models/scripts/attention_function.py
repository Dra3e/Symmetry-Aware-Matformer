import torch
import torch.nn as nn

class AttentionFunction(nn.Module):
    def __init__(self, edge):
        super(AttentionFunction, self).__init__()
        self.query = nn.Linear(edge[0].shape[-1], edge[0].shape[-1])
        self.key_i = nn.Linear(edge[0].shape[-1], edge[0].shape[-1])
        self.key_j = nn.Linear(edge[1].shape[-1], edge[1].shape[-1])
        self.value_i = nn.Linear(edge[0].shape[-1], edge[0].shape[-1])
        self.value_j = nn.Linear(edge[1].shape[-1],edge[1].shape[-1])
        self.edge = nn.Linear(edge[2].shape[-1], edge[2].shape[-1])

        self.layer_norm = nn.LayerNorm(edge[0].shape[-1] * 3)  # 3 because we concatenate over three elements with edge[0] dimensions
        self.symmetry = nn.Linear(edge[2].shape[-1], edge[2].shape[-1], bias=False)

    def forward(self, edge):
        query = self.query(edge[0])
        key_i = self.key_i(edge[0])
        key_j = self.key_j(edge[1])
        value_i = self.value_i(edge[0])
        value_j = self.value_j(edge[1])
        edge = self.edge(edge[2])

        query_C = torch.cat((query, query, query), dim=-1)
        key_C = torch.cat((key_i, key_j, edge), dim=-1)
        value_C = torch.cat((value_i, value_j, edge), dim=1)
        d = key_C.shape[-1]

        alpha_original = torch.sigmoid(self.layer_norm(torch.mul(query_C, key_C) / d))
        symmetry = 1 + self.symmetry(torch.mul(edge[3],edge[2]))
        alpha = torch.mul(torch.mul(alpha_original, symmetry), edge[3])

        return alpha, value_C

# class AttentionFunction(nn.Module):
#     def __init__(self, edge[0], edge[1], edge[2]):
#         super(AttentionFunction, self).__init__()
#         self.query = nn.Linear(edge[0], edge[0])
#         self.key_i = nn.Linear(edge[0], edge[0])
#         self.key_j = nn.Linear(edge[1], edge[1])
#         self.value_i = nn.Linear(edge[0], edge[0])
#         self.value_j = nn.Linear(edge[1],edge[1])
#         self.edge = nn.Linear(edge[2], edge[2])

#         self.layer_norm = nn.LayerNorm(edge[0] * 3)  # 3 because we concatenate over three elements with edge[0] dimensions
#         self.symmetry = nn.Linear(edge[2], edge[2], bias=False)

#     def forward(self, edge[0]_tensor, edge[1]_tensor, edge[2]_tensor, edge[3], M_valid):
#         query = self.query(edge[0]_tensor)
#         key_i = self.key_i(edge[0]_tensor)
#         key_j = self.key_j(edge[1]_tensor)
#         value_i = self.value_i(edge[0]_tensor)
#         value_j = self.value_j(edge[1]_tensor)
#         edge = self.edge(edge[2]_tensor)

#         query_C = torch.cat((query, query, query), dim=-1)
#         key_C = torch.cat((key_i, key_j, edge), dim=-1)
#         value_C = torch.cat((value_i, value_j, edge), dim=1)
#         d = key_C.shape[-1]

#         alpha_original = torch.sigmoid(self.layer_norm(torch.mul(query_C, key_C) / d))
#         symmetry = 1 + self.symmetry(torch.mul(edge[3],edge[2]_tensor))
#         alpha = torch.mul(torch.mul(alpha_original, symmetry), M_valid)

#         return alpha, value_C