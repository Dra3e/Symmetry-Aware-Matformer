# crystal_graph_vec.py  –  only the new / changed bits are shown
import torch

def _pair_edges(coords, cutoff, m_valid, m_sym):
    """
    Build (i, j, distance, edge_type, m_valid, j) for all pairs within cutoff.
    Works on a single tensor of coordinates (N,3).
    """
    D = torch.cdist(coords, coords)                # [N, N]
    mask = torch.triu(D <= cutoff, diagonal=0)     # keep i<=j to avoid dup.+self
    src, dst = mask.nonzero(as_tuple=True)         # 1-D indices
    dist      = D[src, dst]

    # i→j  (and j→i if i≠j)
    edges_ij = torch.stack(
        (src, dst, dist,
         torch.full_like(src, m_sym),          # 0 or 1
         torch.full_like(src, m_valid),            # 0 or 1
         dst), dim=1
    )

    undirected_mask = src != dst                   # avoid doubling self-loops
    edges_ji = edges_ij[undirected_mask].clone()
    edges_ji[:, 0], edges_ji[:, 1], edges_ji[:, 5] = (
        edges_ij[undirected_mask][:, 1],
        edges_ij[undirected_mask][:, 0],
        edges_ij[undirected_mask][:, 0],
    )

    return torch.cat((edges_ij, edges_ji), dim=0)  # shape (E,6)