# crystal_graph_vec.py  –  only the new / changed bits are shown
import torch

def _pair_edges(coords, cutoff, m_valid, m_sym):
    """
    Build (i, j, distance, edge_type, m_valid, j) for all pairs within cutoff distance.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 3) representing the coordinates of N points.
        cutoff (float): Maximum distance for edge creation.
        m_valid (int): Validity flag for edges (1 = valid, 0 = invalid).
        m_sym (int): Symmetry flag for edges (0 = physical, 1 = augmented).

    Returns:
        torch.Tensor: Tensor of shape (E, 6) representing edges, where E is the number of edges.
                      Each edge is represented as (src, dst, distance, edge_type, m_valid, j).
    """
    # Compute pairwise distances [N, N] define D as the Ecuclidean distance function
    D = torch.cdist(coords, coords)  
                  
    # Create a mask for edges within the cutoff distance (upper triangular to avoid duplicates)
    mask = torch.triu(D <= cutoff, diagonal=0) 
    # Extract source and destination indices for edges     
    src, dst = mask.nonzero(as_tuple=True)         # 1-D indices of valid edges
    dist     = D[src, dst]                         

    # Create directed edges (i → j) (and j→i if i≠j)
    edges_ij = torch.stack(
        (src,                                      # Source node index
         dst,                                      # Destination node index
         dist,                                     # Distance between nodes
         torch.full_like(src, m_sym),              # Creates a tensor filled with value of m_sym. 0 or 1
         torch.full_like(src, m_valid),            # Creates a tensor filled with value of m_valid. 0 or 1
         dst                                       # Destination node index (for undirected edge handling)
    ), dim=1)

    # Create reverse edges (j→i) for undirected graphs
    undirected_mask = src != dst                        # avoid doubling self-loops
    edges_ji = edges_ij[undirected_mask].clone()        # clone edges to modify
    edges_ji[:, 0], edges_ji[:, 1], edges_ji[:, 5] = (  
        edges_ij[undirected_mask][:, 1],                # Swap src and dst
        edges_ij[undirected_mask][:, 0],                # Update src node index to dst
        edges_ij[undirected_mask][:, 0],                # Update destination node index to src
    )


    # Concatenate directed and reverse edges
    edges = torch.cat((edges_ij, edges_ji), dim=0)  # Shape: (E, 6)

    return edges