import torch
import torch.nn as nn
from mendeleev import element

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
        self.hidden_dim = hidden_dim
    
    def forward(self, graph):
        """
        Generates node and edge embeddings using dynamically set parameters.
            f*_i = LinearTransform(CGCNN(a_i))
            e_ij = Softplus(LinearTransform(RBF(d_ij)))

        Args:
            graph (models.components.CrystalGraph): CrystalGraph instance containing nodes and edges.

        Returns:
            atom_features (torch.Tensor): [N, hidden_dim] Node feature matrix.
            edge_features (torch.Tensor): [num_edges, 6, hidden_dim] Edge feature matrix.
        """
        if isinstance(graph.edges, torch.Tensor):
            edges = graph.edges
        else:
            edges = torch.tensor(graph.edges, dtype=torch.float32)
        
        atom_features = self.atom_embedding.forward(graph.nodes)
        transformed_distances = self.edge_transform.forward(edges)

        from_indices = edges[:, 0].long() # Extract indices of from nodes
        to_indices = edges[:, 1].long() # Extract indices of to nodes
        from_node_emb = atom_features[from_indices] # Get embedded from nodes
        to_node_emb = atom_features[to_indices] # Get embedded to nodes
        
        # Dynamic edge attributes from column 3 onward
        raw_edge_values = edges[:, 3:].float()  # (E, N)
        expanded_attrs = raw_edge_values.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        expanded_attrs = expanded_attrs.permute(1, 0, 2)
        # print(f"expanded_attrs shape: {expanded_attrs.shape}")

        # Directly stack the tensors instead of creating a list of tuples
        edge_features = torch.stack([
            from_node_emb,             # (number_of_edges, hidden_dim)
            to_node_emb,               # (number_of_edges, hidden_dim)
            transformed_distances,     # (number_of_edges, hidden_dim)
            *expanded_attrs  # Already of shape (number_of_edges, N, hidden_dim)
        ], dim=1)  # Shape: (number_of_edges, 6, hidden_dim)

        return atom_features, edge_features
    
    def to(self, device):
        """
        Move the GraphEmbedder instance to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.atom_embedding = self.atom_embedding.to(device)
        self.edge_transform = self.edge_transform.to(device)
        super().to(device)
        return self

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
        self.lookup = AtomFeatureLookup(input_dim)
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
        try:
            atom_numbers = A[:, 0].int()
        except:
            raise ValueError("Cannot extract atomic numbers from self.A. Ensure atomic number is in first column.")
        A_featured = self.lookup(atom_numbers)

        return self.embedding(A_featured)  # Linear transformation
    
    def to(self, device):
        """
        Move the CGCNN instance to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.lookup = self.lookup.to(device)
        self.embedding = self.embedding.to(device)
        super().to(device)
        return self
    
class AtomFeatureLookup(nn.Module):
    def __init__(self, atom_feature_dim=92):
        super().__init__()
        self.register_buffer("atom_features", torch.zeros((atom_feature_dim, atom_feature_dim)))

        for atomic_num in range(1, 93):  # Atomic numbers 1 to 92
            elem = element(atomic_num)
            # Example feature vector: atomic number, atomic mass, electronegativity, group, period
            features = [
                elem.atomic_number,
                elem.atomic_weight,
                elem.electronegativity_pauling(),
                elem.group_id,
                elem.period,
                elem.covalent_radius,
                elem.vdw_radius,
                elem.density
            ]
            features = [f if f is not None else 0.0 for f in features]
            # Pad or truncate to the desired feature length
            features += [0.0] * (atom_feature_dim - len(features))
            self.atom_features[atomic_num - 1] = torch.tensor(features[:atom_feature_dim], dtype=torch.float32)

    def forward(self, atomic_numbers):
        """
        Args:
            atomic_numbers (torch.Tensor): Tensor of atomic numbers (n_atoms,).
        
        Returns:
            torch.Tensor: Atom features (n_atoms, atom_feature_dim).
        """
        return self.atom_features[atomic_numbers - 1]  # Adjust for zero-indexing
    
    def to(self, device):
        """
        Move the AtomFeatureLookup instance to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.atom_features = self.atom_features.to(device)
        super().to(device)
        return self

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

        distances = edges[:, 2].unsqueeze(-1)
        expanded_distances = self.rbf_expansion(distances) # RBF Expansion
        transformed_distances = self.activation(self.linear(expanded_distances)) # Linear transformation + Softplus 

        return transformed_distances
    
    def to(self, device):
        """
        Move the EdgeTransformation instance to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.linear = self.linear.to(device)
        self.rbf_expansion = self.rbf_expansion.to(device)
        self.activation = self.activation.to(device)
        super().to(device)
        return self


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
        output = torch.exp(-self.gamma * (D_expanded - self.centers) ** 2)  # Compute RBF transformation

        return output.squeeze(1)
    
    def to(self, device):
        """
        Move the RBF instance to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.centers = self.centers.to(device)
        self.gamma = self.gamma.to(device)
        super().to(device)
        return self