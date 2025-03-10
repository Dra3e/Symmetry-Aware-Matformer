import torch
import torch.nn as nn


class CrystalGraph(nn.Module):
    """
    A graph representing the structure of a crystal through learned node and edge embeddings.

    Attributes:
        atom_embedding (CGCNN): Converts atomic features into node embeddings.
        edge_embedding (EdgeTransformation): Converts distances into edge embeddings.
    """

    def __init__(self, atom_feature_dim, hidden_dim, num_rbf_kernels):
        """
        Initializes the weights and kernels for graph transformation.

        Args:
            atom_feature_dim (int): Dimension of input atomic feature vectors (d).
            hidden_dim (int): Dimension of transformed node and edge features (h).
            num_rbf_kernels (int): Number of RBF kernels for distance expansion.
        """
        super().__init__()
        self.atom_embedding = CGCNN(atom_feature_dim, hidden_dim)
        self.edge_transform = EdgeTransformation(num_rbf_kernels, hidden_dim)

    def forward(self, A, D):
        """
        Converts an atomic feature matrix A and a distance matrix D into a graph.
            f*_i = LN(CGCNN(a_i))
            e^h_ij = Softplus(LN(RBF(d_ij)))

        Args:
            A (torch.Tensor): Atomic feature matrix of shape (N, d), where:
                - N: Number of atoms.
                - d: Atomic feature dimension.
                - Each entry row a_i represents the features of an atom.
            D (torch.Tensor): Distance matrix of shape (N, N), where:
                - N: Number of atoms.
                - Each entry d_ij represents the distance between atom i and atom j.

        Returns:
            torch.Tensor: Node feature matrix f*_i of shape (N, h).
            torch.Tensor: Edge feature matrix e^h_ij of shape (N, N, h).
        """
        # f*_i = LN(CGCNN(a_i))
        f_star = self.atom_embedding(A)

        # e^h_ij = Softplus(LN(RBF(d_ij)))
        e_h = self.edge_transform(D)

        return f_star, e_h


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
            torch.Tensor: Transformed node feature matrix f*_i of shape (N, h).
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

    def forward(self, D):
        """
        Applies a linear transformation and Softplus activation to RBF-expanded distance features.
            e^h_ij = Softplus(LN(RBF(d_ij)))

        Args:
            D (torch.Tensor): Distance matrix of shape (N, N), where each d_ij represents
            the Euclidean distance between atoms i and j.
                - N: Number of atoms.

        Returns:
            torch.Tensor: Transformed edge embeddings of shape (N, N, hidden_dim).
        """
        rbf_features = self.rbf_expansion(D)
        return self.activation(self.linear(rbf_features))  # Linear transformation + Softplus

class RBF(nn.Module):
    """
    Expands the distance matrix D using Radial Basis Function (RBF) kernels.

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
