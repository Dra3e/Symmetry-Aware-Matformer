import torch
import torch.nn as nn

class CGCNN(nn.Module):
    """
    Transforms the atomic feature matrix A into learned node embeddings.
    
    This mimics a simplified version of a Crystal Graph Convolutional Neural Network (CGCNN).
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the CGCNN embedding layer.

        Args:
            input_dim (int): Dimension of atomic input features (d).
            hidden_dim (int): Dimension of the output node embeddings (h).
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Learnable transformation

    def forward(self, A):
        """
        Applies a linear transformation to atomic feature matrix A.
            f*_i = LN(CGCNN(a_i))

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
        self.linear = nn.Linear(num_rbf_kernels + 1, hidden_dim)  # Linear transformation (+1 for edge type)
        self.rbf_expansion = RBF(num_rbf_kernels)   # RBF expansion
        self.activation = nn.Softplus()  # Ensures positive outputs for stability

    def forward(self, edges):
        """
        Args:
            edges (list of [i, j, distance, type]):
                - i: source node index
                - j: target node index
                - distance: Euclidean distance (float)
                - type: 0 (physical) or 1 (augmented)
        
        Returns:
            edge_features (Tensor): shape (num_edges, hidden_dim)
        """
        distances = torch.tensor([e[2] for e in edges]).unsqueeze(-1)
        types = torch.tensor([e[3] for e in edges]).unsqueeze(-1).float()

        rbf_encoded = self.rbf_expansion(distances)
        combined = torch.cat([rbf_encoded, types], dim=1)

        return self.activation(self.linear(combined))  # Linear transformation + Softplus

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
        Converts the 2D matrix of distances D into a 3D matrix of RBF kernel vector distances.

        Args:
            D (torch.Tensor): Distance matrix of shape (N, N), where each d_ij represents
            the Euclidean distance between atoms i and j.
                - N: Number of atoms.

        Returns:
            D_expanded torch.Tensor: Expanded distance matrix of shape (N, N, num_kernels).
        """
        D_expanded = D.unsqueeze(-1)  # Add extra dimension for broadcasting
        return torch.exp(-self.gamma * (D_expanded - self.centers) ** 2)  # Compute RBF transformation