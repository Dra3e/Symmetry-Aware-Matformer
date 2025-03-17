import torch.nn as nn
from models.steps.crystal_graph import CGCNN, EdgeTransformation


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
            f* torch.Tensor: Node feature matrix f*_i of shape (N, h).
            e^h torch.Tensor: Edge feature matrix e^h_ij of shape (N, N, h).
        """
        # f*_i = LN(CGCNN(a_i))
        f_star = self.atom_embedding(A)

        # e^h_ij = Softplus(LN(RBF(d_ij)))
        e_h = self.edge_transform(D)

        return f_star, e_h