import torch
import spglib
import numpy as np

from components.rotation import rotate_3d_object
from components.reflection import reflect_crystal
from components.translation import TranslateRand

def apply_transformations(A, X, L, compute_e3_invariant_edges, translation_coeffs=None, rotation_axis=None, rotation_angle=None, reflection_normal=None, reflection_point=None):
    """
    Applies transformations (translation, rotation, reflection) to the crystal structure
    and updates edges accordingly, including space group analysis.
    
    Parameters:
    - A: Atomic features (N, F)
    - X: Atomic positions (N, 3)
    - L: Lattice vectors (3, 3)
    - compute_edges: Function to recompute edges after transformations
    - translation_coeffs: (3,) integer array for translation
    - rotation_axis: (3,) array for rotation axis
    - rotation_angle: Float, rotation angle in degrees
    - reflection_normal: (3,) array for reflection normal
    - reflection_point: Optional (3,) array for reflection plane point

    Returns:
    - Updated A, X, L
    - Updated edges (edge_index, edge_attr)
    """
    # Space group analyse ved hjælp af spglib
    crystal = (L, X, A)  # lattice vectors, atom positions og atomfeatures
    dataset = spglib.get_symmetry_dataset(crystal)
    space_group = dataset['spacegroup']
    sym_ops = dataset['symmetry_operations']

    # Udfør transformationer som tidligere
    A_transformed, X_transformed, L_transformed = np.copy(A), np.copy(X), np.copy(L)

    if translation_coeffs is not None:
        A_transformed, X_transformed, L_transformed = TranslateRand(A, X_transformed, L_transformed, translation_coeffs)
    
    if rotation_axis is not None and rotation_angle is not None:
        A_transformed, X_transformed, L_transformed = rotate_3d_object(A, X_transformed, L_transformed, rotation_axis, rotation_angle)
    
    if reflection_normal is not None:
        A_transformed, X_transformed, L_transformed = reflect_crystal(A, X_transformed, L_transformed, reflection_normal, reflection_point)

    # Konverter til PyTorch tensors
    A_tensor = torch.tensor(A_transformed, dtype=torch.float32)
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    L_tensor = torch.tensor(L_transformed, dtype=torch.float32)

    # Genberegn kanter efter transformationerne
    edge_index, edge_attr = compute_e3_invariant_edges(A_tensor, X_tensor, L_tensor)

    return A_tensor, X_tensor, L_tensor, edge_index, edge_attr, space_group, sym_ops


def compute_e3_invariant_edges(A, X, L, cutoff=3.0):
    """
    Computes edges and edge attributes based on atomic positions,
    ensuring E(3) invariance (translation, rotation, reflection).
    
    Parameters:
    - A (torch.Tensor): Atomic features (N, F), where N is number of atoms and F is number of features.
    - X (torch.Tensor): Atomic positions in 3D (N, 3).
    - L (torch.Tensor): Lattice vectors (3, 3).
    - cutoff (float): Maximum bond distance to consider for edges.
    
    Returns:
    - edge_index (torch.Tensor): Tensor of shape (2, num_edges) representing atomic connections.
    - edge_attr (torch.Tensor): Tensor of shape (num_edges,) containing edge attributes (such as distance).
    """
    num_nodes = X.shape[0]
    edges = []
    features = []

    # Translation vectors for adjacent unit cells
    translation_vectors = torch.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=torch.float32) @ L

    # Create extended unit cells
    extended_X = [X]  # Start with original unit cell
    for vec in translation_vectors:
        extended_X.append(X + vec)
    extended_X = torch.cat(extended_X, dim=0)

    # Compute pairwise distances and build edges
    for i in range(num_nodes):
        for j in range(len(extended_X)):
            distance = torch.norm(X[i] - extended_X[j])
            if distance <= cutoff:
                edges.append((i, j % num_nodes))  # Map to original indices
                features.append(distance)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(features, dtype=torch.float32)

    return edge_index, edge_attr