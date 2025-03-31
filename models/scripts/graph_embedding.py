import torch
import torch.nn as nn

class GraphEmbedder(nn.Module):
    """ Script class for embedding crystal graphs"""
    def __init__(self, atom_feature_dim, hidden_dim, num_rbf_kernels=50):
        """
        Initializes the GraphEmbedder with its hidden dimensions.

        Args:
            atom_feature_dim (int): Dimension of input atomic feature vectors.
            hidden_dim (int): Dimension of transformed node and edge features.
            num_rbf_kernels (int): Number of RBF kernels for distance expansion.
        """
        super().__init__()
        self.atom_embedding = CGCNN(atom_feature_dim, hidden_dim)
        self.edge_transform = EdgeTransformation(num_rbf_kernels, hidden_dim)
    
    def forward(self, graph):
        """
        Generates node and edge embeddings using dynamically set parameters.
            f*_i = LinearTransform(CGCNN(a_i))
            e_ij = Softplus(LinearTransform(RBF(d_ij)))

        Args:
            graph (models.components.CrystalGraph): CrystalGraph instance containing nodes and edges.

        Returns:
            atom_features (torch.Tensor): Node feature matrix.
            edge_features (torch.Tensor): Edge feature matrix.
        """
        atom_features = self.atom_embedding.forward(graph.nodes)
        edge_features = self.edge_transform.forward(graph.edges)
        return atom_features, edge_features

class CGCNN(nn.Module):
    """ Transforms the atomic feature matrix A into learned node embeddings."""

    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the CGCNN embedding layer.

        Args:
            input_dim (int): Dimension of atomic input features (d).
            hidden_dim (int): Dimension of the output node embeddings.
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Learnable transformation

    def forward(self, A):
        """
        Applies a linear transformation to atomic feature matrix A.
            f*_i = LinearTransform(CGCNN(a_i))

        Args:
            A (torch.Tensor): Atomic feature matrix of shape (N, d).
                - N: Number of atoms.
                - d: Atomic feature dimension.

        Returns:
            f*_i torch.Tensor: Transformed node feature matrix f*_i of shape (N, h).
        """
        return self.embedding(A)  # Linear transformation

class EdgeTransformation(nn.Module):
    """
    Transforms RBF-expanded features into edge embeddings learned edge embeddings,
    such that they are suitable for attention calculation.
    """

    def __init__(self, num_rbf_kernels, hidden_dim):
        """
        Initializes the edge transformation module.

        Args:
            num_rbf_kernels (int): Number of RBF kernels for distance expansion.
            rbf_expansion (RBF): Converts edges into vectors through Radial Basis Function (RBF) kernels.
            hidden_dim (int): Dimension of the output edge embeddings (h).
        """
        super().__init__()
        self.linear = nn.Linear(num_rbf_kernels, hidden_dim)  # Linear transformation
        self.rbf_expansion = RBF(num_rbf_kernels)   # RBF expansion
        self.activation = nn.Softplus()  # Ensures positive outputs for stability

    def forward(self, edges):
        """
        Embeds the edges using RBF expansion and linear transformation.
            e_ij = Softplus(LinearTransform(RBF(d_ij)))

        Args:
            edges (list of [i, j, distance, type, valid]):
                - i: source node index
                - j: target node index
                - distance: Euclidean distance (float) between atoms i and j
                - type: 0 (physical) or 1 (augmented)
                - valid: 1 (valid) or 0 (invalid)
        
        Returns:
            transformed_edges (torch.Tensor): replaces the distance in the edge list with the transformed distance.
                - transformed_edges: [i, j, transformed_distance, type, valid]
        """
        distances = edges[:, :, 2].unsqueeze(-1)
        expanded_distances = self.rbf_expansion(distances) # RBF Expansion
        transformed_distances = self.activation(self.linear(expanded_distances)) # Linear transformation + Softplus 

        transformed_edges = edges.clone()
        transformed_edges[:, :, 2] = transformed_distances.squeeze(-1)

        return transformed_edges


class RBF(nn.Module):
    """
    Expands the distance matrix D using Radial Basis Function (RBF) kernels.
        exp(-gamma(d_ij-k)^2)

        d_ij    is the distance between atoms a_i and a_j
        k       is the distance of the center of the RBF kernel from atom a_i
        gamma   determines the distance between the kernels (lower gamme = higher distance)

    Instead of using raw distances, RBF transforms them into a high-dimensional 
    feature space where each entry encodes how close a distance is to predefined centers.
    """

    def __init__(self, num_kernels=40, min_dist=0.0, max_dist=8.0):
        """
        Initializes the RBF expansion module.

        Args:
            num_kernels (int): Number of RBF kernels (default: 40).
            min_dist (float): Minimum distance to consider in expansion (default: 0.0 Å).
            max_dist (float): Maximum distance cutoff (default: 8.0 Å).
        """
        super().__init__()
        self.centers = torch.linspace(min_dist, max_dist, num_kernels)  # Define kernel centers
        self.gamma = 1 / ((self.centers[1] - self.centers[0]) ** 2)  # Kernel scaling factor

    def forward(self, D):
        """
        Converts the scalar distance d_ij into a vector of size num_kernels.

        Args:
            D (torch.Tensor): Distance matrix of shape (N, N), where each d_ij represents
            the Euclidean distance between atoms i and j.
                - N: Number of atoms.

        Returns:
            D_expanded: Expanded distance matrix of shape.
        """
        D_expanded = D.unsqueeze(-1)  # Add extra dimension for broadcasting
        return torch.exp(-self.gamma * (D_expanded - self.centers) ** 2)  # Compute RBF transformation