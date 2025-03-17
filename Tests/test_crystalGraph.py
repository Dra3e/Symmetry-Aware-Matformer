import torch
import unittest
from models.augmentations.reflection import Plane
from models.components.crystal_graph import CrystalGraph
from models.scripts.graph_embedding import CGCNN, RBF, EdgeTransformation


class Test_crystalGraph(unittest.TestCase):

    def setUp(self):
        """Initialize a CrystalGraph instance with predefined parameters."""
        self.atom_feature_dim = 10
        self.hidden_dim = 16
        self.num_rbf_kernels = 40
        self.graph = CrystalGraph(self.atom_feature_dim, self.hidden_dim, self.num_rbf_kernels)

    def test_initialization(self):
        """Ensure CGCNN and EdgeTransformation are properly initialized."""
        self.assertEqual(self.graph.atom_embedding.embedding.in_features, self.atom_feature_dim)
        self.assertEqual(self.graph.atom_embedding.embedding.out_features, self.hidden_dim)
        self.assertEqual(self.graph.edge_transform.linear.in_features, self.num_rbf_kernels)
        self.assertEqual(self.graph.edge_transform.linear.out_features, self.hidden_dim)

    def test_forward_output_shape(self):
        """Check if output shapes are as expected."""
        N = 5  # Number of atoms
        A = torch.rand((N, self.atom_feature_dim))  # Atomic feature matrix (N, d)
        D = torch.rand((N, N))  # Distance matrix (N, N)

        f_star, e_h = self.graph(A, D)
        
        self.assertEqual(f_star.shape, (N, self.hidden_dim))  # Node embeddings
        self.assertEqual(e_h.shape, (N, N, self.hidden_dim))  # Edge embeddings
    
    def test_forward_computation(self):
        """Ensure outputs are transformed correctly and non-negative."""
        N = 4
        A = torch.rand((N, self.atom_feature_dim))
        D = torch.rand((N, N))
        f_star, e_h = self.graph(A, D)

        self.assertFalse(torch.isnan(f_star).any())  # No NaNs in node embeddings
        self.assertFalse(torch.isnan(e_h).any())  # No NaNs in edge embeddings
        self.assertTrue(torch.all(e_h >= 0))  # Edge embeddings should be non-negative (Softplus applied)

    def test_empty_input(self):
        """Test behavior when input matrices are empty (0 atoms)."""
        A = torch.empty((0, self.atom_feature_dim))  # Empty atomic matrix
        D = torch.empty((0, 0))  # Empty distance matrix

        f_star, e_h = self.graph(A, D)
        
        self.assertEqual(f_star.shape, (0, self.hidden_dim))
        self.assertEqual(e_h.shape, (0, 0, self.hidden_dim))

    def test_single_atom(self):
        """Test behavior with a single atom in the graph."""
        A = torch.rand((1, self.atom_feature_dim))
        D = torch.rand((1, 1))  # Self-distance

        f_star, e_h = self.graph(A, D)
        
        self.assertEqual(f_star.shape, (1, self.hidden_dim))
        self.assertEqual(e_h.shape, (1, 1, self.hidden_dim))

    def test_large_input_matrix(self):
        """Test CrystalGraph performance on a large atomic graph."""
        N = 1000  # Large molecule
        A = torch.rand((N, self.atom_feature_dim))
        D = torch.rand((N, N))

        f_star, e_h = self.graph(A, D)
        
        self.assertEqual(f_star.shape, (N, self.hidden_dim))
        self.assertEqual(e_h.shape, (N, N, self.hidden_dim))

    def test_gradient_flow(self):
        """Ensure backpropagation works through the graph network."""
        N = 5
        A = torch.rand((N, self.atom_feature_dim), requires_grad=True)
        D = torch.rand((N, N), requires_grad=True)

        f_star, e_h = self.graph(A, D)
        loss = f_star.sum() + e_h.sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertFalse(torch.isnan(A.grad).any())

    def test_deterministic_behavior(self):
        """Test if the model is deterministic with the same input."""
        torch.manual_seed(42)  # Fix random seed
        N = 5
        A = torch.rand((N, self.atom_feature_dim))
        D = torch.rand((N, N))

        output1 = self.graph(A, D)
        output2 = self.graph(A, D)

        for o1, o2 in zip(output1, output2):
            self.assertTrue(torch.allclose(o1, o2, atol=1e-6))
    
    def test_batched_input(self):
        """Test if CrystalGraph correctly processes batched inputs."""
        N = 5  # Number of atoms per batch
        B = 3  # Number of batches
        A = torch.rand((B, N, self.atom_feature_dim))  # Batched atomic features
        D = torch.rand((B, N, N))  # Batched distance matrices

        f_star, e_h = self.graph(A, D)
        
        self.assertEqual(f_star.shape, (B, N, self.hidden_dim))
        self.assertEqual(e_h.shape, (B, N, N, self.hidden_dim))
    
    def test_crystalgraph_non_square_batches(self):
        """Ensure CrystalGraph handles batched non-square inputs safely."""
        B = 3  # Number of batches
        N1, N2 = 5, 6  # Non-square shapes
        A = torch.rand((B, N1, self.atom_feature_dim))  # Batched atomic features
        D = torch.rand((B, N1, N2))  # Non-square batch distances

        with self.assertRaises(RuntimeError):  # Expect failure
            self.graph(A, D)
    
    def test_invalid_inputs(self):
        """Ensure CrystalGraph raises errors for incorrect inputs."""
        with self.assertRaises(TypeError):
            self.graph("invalid_input", "invalid_input")  # Should be tensors

        with self.assertRaises(RuntimeError):
            self.graph(torch.rand((3, 3, 3)), torch.rand((3, 3)))  # Wrong shapes

class Test_CGCNN(unittest.TestCase):

    def setUp(self):
        """Initialize a CGCNN instance with predefined parameters."""
        self.input_dim = 10  # Feature dimension of each atom
        self.hidden_dim = 16  # Output embedding dimension
        self.cnn = CGCNN(self.input_dim, self.hidden_dim)

    def test_initialization(self):
        """Ensure the linear layer is set up correctly."""
        self.assertEqual(self.cnn.embedding.in_features, self.input_dim)
        self.assertEqual(self.cnn.embedding.out_features, self.hidden_dim)

    def test_forward_output_shape(self):
        """Check if the output shape matches (N, hidden_dim)."""
        A = torch.rand((5, self.input_dim))  # 5 atoms, each with 'input_dim' features
        output = self.cnn(A)
        expected_shape = (5, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_transformation(self):
        """Ensure transformation modifies input features."""
        A = torch.rand((5, self.input_dim))
        output = self.cnn(A)
        self.assertFalse(torch.equal(A, output))  # Output should differ from input

    def test_empty_input(self):
        """Test behavior when input has zero atoms."""
        A = torch.empty((0, self.input_dim))  # No atoms
        output = self.cnn(A)
        self.assertEqual(output.shape, (0, self.hidden_dim))  # Should return empty tensor

    def test_single_atom(self):
        """Test with a single atom feature vector."""
        A = torch.rand((1, self.input_dim))  # 1 atom
        output = self.cnn(A)
        self.assertEqual(output.shape, (1, self.hidden_dim))

    def test_large_input_matrix(self):
        """Test CGCNN performance on a large atomic feature matrix."""
        A = torch.rand((1000, self.input_dim))  # 1000 atoms
        output = self.cnn(A)
        self.assertEqual(output.shape, (1000, self.hidden_dim))

    def test_negative_and_large_values(self):
        """Ensure transformation handles extreme values properly."""
        A = torch.tensor([[-1.0] * self.input_dim, [1e6] * self.input_dim])  # Edge cases
        output = self.cnn(A)
        self.assertFalse(torch.isnan(output).any())  # No NaNs
        self.assertFalse(torch.isinf(output).any())  # No infinite values

    def test_gradient_flow(self):
        """Ensure CGCNN supports backpropagation."""
        A = torch.rand((5, self.input_dim), requires_grad=True)
        output = self.cnn(A)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(A.grad)
        self.assertFalse(torch.isnan(A.grad).any())

class TestEdgeTransformation(unittest.TestCase):

    def setUp(self):
        """Initialize an EdgeTransformation instance."""
        self.num_kernels = 40
        self.hidden_dim = 64
        self.edge_transform = EdgeTransformation(self.num_kernels, self.hidden_dim)

    def test_initialization(self):
        """Ensure proper initialization of RBF expansion and linear layer."""
        self.assertEqual(self.edge_transform.linear.in_features, self.num_kernels)
        self.assertEqual(self.edge_transform.linear.out_features, self.hidden_dim)
        self.assertIsNotNone(self.edge_transform.rbf_expansion)

    def test_forward_output_shape(self):
        """Check if output shape matches expected (N, N, hidden_dim)."""
        D = torch.tensor([[0.5, 1.0], [1.5, 2.0]])  # Sample 2x2 distance matrix
        output = self.edge_transform(D)
        expected_shape = (2, 2, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_computation(self):
        """Ensure that outputs are properly transformed and non-negative."""
        D = torch.tensor([[0.0, 4.0], [4.0, 8.0]])  # Edge case distances
        output = self.edge_transform(D)
        self.assertTrue(torch.all(output >= 0))  # Softplus ensures non-negative values
        self.assertFalse(torch.isnan(output).any())  # Ensure no NaNs

    def test_zero_distance_matrix(self):
        """Test behavior when input distance matrix is all zeros."""
        D = torch.zeros((3, 3))  # All zero distances
        output = self.edge_transform(D)
        self.assertEqual(output.shape, (3, 3, self.hidden_dim))
        self.assertTrue(torch.all(output >= 0))  # Should still be non-negative

    def test_large_distance_matrix(self):
        """Test behavior when input distances are at the max range."""
        D = torch.full((3, 3), 8.0)  # Max distance
        output = self.edge_transform(D)
        self.assertEqual(output.shape, (3, 3, self.hidden_dim))

    def test_negative_distances(self):
        """Ensure negative distances raise an error."""
        D = torch.tensor([[0.5, -1.0], [-1.5, 2.0]])  # Invalid negative values
        with self.assertRaises(ValueError):
            self.edge_transform(D)

    def test_non_square_input(self):
        """Ensure non-square distance matrices raise an error."""
        D = torch.rand((3, 4))  # Non-square tensor
        with self.assertRaises(RuntimeError):  # Expecting a shape-related error
            self.edge_transform(D)

    def test_large_input_matrix(self):
        """Test EdgeTransformation performance on a large input matrix."""
        D = torch.rand((100, 100)) * 8.0  # Large random distances
        output = self.edge_transform(D)
        self.assertEqual(output.shape, (100, 100, self.hidden_dim))

    def test_gradient_flow(self):
        """Ensure gradients flow properly through the transformation."""
        D = torch.tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
        output = self.edge_transform(D)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(D.grad)
        self.assertFalse(torch.isnan(D.grad).any())

class Test_RBF(unittest.TestCase):

    def setUp(self):
        """Initialize an RBF instance with predefined parameters."""
        self.num_kernels = 40
        self.min_dist = 0.0
        self.max_dist = 8.0
        self.rbf = RBF(num_kernels=self.num_kernels, min_dist=self.min_dist, max_dist=self.max_dist)

    def test_initialization(self):
        """Test if RBF initializes correctly with expected attributes."""
        self.assertEqual(self.rbf.centers.shape[0], self.num_kernels)
        self.assertTrue(torch.all(self.rbf.centers >= self.min_dist))
        self.assertTrue(torch.all(self.rbf.centers <= self.max_dist))

    def test_forward_output_shape(self):
        """Test if RBF forward output has correct shape."""
        D = torch.tensor([[0.5, 1.0], [1.5, 2.0]])  # Sample small matrix
        output = self.rbf(D)
        expected_shape = (2, 2, self.num_kernels)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_computation(self):
        """Test if RBF transformation produces valid values."""
        D = torch.tensor([[0.0, 4.0], [4.0, 8.0]])  # Extreme values
        output = self.rbf(D)
        self.assertTrue(torch.all(output >= 0))  # RBF should be non-negative
        self.assertTrue(torch.all(output <= 1))  # Max should be 1

    def test_extreme_distance_values(self):
        """Test RBF with very large and very small distances."""
        D = torch.tensor([[1e-10, 1e3], [1e3, 1e-10]])  # Very small and very large values
        output = self.rbf(D)
        self.assertFalse(torch.isnan(output).any())  # No NaNs
        self.assertFalse(torch.isinf(output).any())  # No inf values

    def test_zero_distance_matrix(self):
        """Test behavior when input distance matrix is all zeros."""
        D = torch.zeros((5, 5))  # All zero distances
        output = self.rbf(D)
        self.assertEqual(output.shape, (5, 5, self.num_kernels))
        self.assertTrue(torch.all(output[:, :, 0] > 0))  # First kernel should have max response

    def test_large_distance_matrix(self):
        """Test behavior when input distances are at max range."""
        D = torch.full((5, 5), self.max_dist)  # Max distance
        output = self.rbf(D)
        self.assertEqual(output.shape, (5, 5, self.num_kernels))
        self.assertTrue(torch.all(output[:, :, -1] > 0))  # Last kernel should have max response

    def test_gradient_flow(self):
        """Test if RBF supports gradient computation for backpropagation."""
        D = torch.tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)  # Track gradients
        output = self.rbf(D)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(D.grad)  # Gradients should exist
        self.assertFalse(torch.isnan(D.grad).any())  # Ensure no NaNs in gradient

    def test_large_input_matrix(self):
        """Test RBF performance on a large input matrix."""
        D = torch.rand((1000, 1000)) * self.max_dist  # Large random distances
        output = self.rbf(D)
        self.assertEqual(output.shape, (1000, 1000, self.num_kernels))
    
    def test_invalid_shape(self):
        """Test if RBF raises an error for non-square inputs."""
        D = torch.rand((3, 4))  # Non-square tensor
        with self.assertRaises(RuntimeError):  # Expecting a shape-related error
            self.rbf(D)
    
    def test_rbf_deterministic_behavior(self):
        """Ensure RBF produces identical outputs for identical inputs."""
        torch.manual_seed(42)  # Fix randomness
        D = torch.rand((5, 5))

        output1 = self.rbf(D)
        output2 = self.rbf(D)

        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_negative_distances(self):
        """Test behavior when distances contain negative values."""
        D = torch.tensor([[0.5, -1.0], [-1.5, 2.0]])  # Negative values
        with self.assertRaises(ValueError):  # Expect an explicit error
            self.rbf(D)

if __name__ == '__main__':
    unittest.main()
