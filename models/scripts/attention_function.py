import torch
import torch.nn as nn

class AttentionFunction(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFunction, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key_i = nn.Linear(hidden_dim, hidden_dim)
        self.key_j = nn.Linear(hidden_dim, hidden_dim)
        self.value_i = nn.Linear(hidden_dim, hidden_dim)
        self.value_j = nn.Linear(hidden_dim,hidden_dim)
        self.edge = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim * 3)  # 3 because we concatenate over three elements with edge[0] dimensions
        self.symmetry = nn.Linear(hidden_dim, hidden_dim, bias=False)

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
        alpha = torch.mul(torch.mul(alpha_original, symmetry), edge[4])

        return alpha, value_C