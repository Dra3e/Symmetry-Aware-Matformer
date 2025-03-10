import unittest
import torch

from components.group_theory import compute_e3_invariant_edges

# Assuming that the function 'compute_e3_invariant_edges' is defined in the same file or imported from another module

class TestComputeE3InvariantEdges(unittest.TestCase):
    
    def test_empty_input(self):
        """Test the function using an empty tensor input."""
        X = torch.empty((0, 3))  # Create an empty input tensor (shape: (0, 3))
        edge_index, edge_attr = compute_e3_invariant_edges(X)  # Call the function
        self.assertEqual(edge_index.shape[1], 0)  # Expect no edges
        self.assertEqual(edge_attr.shape[0], 0)    # Expect no edge attributes
        
    def test_single_node(self):
        """Test the function with a single node input."""
        X = torch.tensor([[0.0, 0.0, 0.0]])  # Create a tensor with one node
        edge_index, edge_attr = compute_e3_invariant_edges(X)  # Call the function
        self.assertEqual(edge_index.shape[1], 0)  # Expect no edges
        self.assertEqual(edge_attr.shape[0], 0)    # Expect no edge attributes
        
    def test_two_nodes(self):
        """Test the function with two nodes; should return one edge."""
        X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Two points in space
        edge_index, edge_attr = compute_e3_invariant_edges(X)  # Call the function
        self.assertEqual(edge_index.shape[1], 1)  # Expect one edge
        self.assertEqual(edge_attr.shape[0], 1)    # Expect one edge attribute
        self.assertAlmostEqual(edge_attr[0].item(), 1.0)  # Distance should be 1.0
        
    def test_multiple_nodes(self):
        """Test the function with multiple nodes; verify edge and attribute counts."""
        X = torch.tensor([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 1.0, 1.0]])  # Define four nodes
                          
        edge_index, edge_attr = compute_e3_invariant_edges(X)  # Call the function
        num_edges = edge_index.shape[1]
        self.assertEqual(num_edges, 6)  # Expect C(4, 2) = 6 edges
        self.assertEqual(edge_attr.shape[0], num_edges)  # Number of edges should match edge attributes

    def test_known_distances(self):
        """Test the function with known distances between two nodes."""
        X = torch.tensor([[0.0, 0.0, 0.0],
                          [3.0, 4.0, 0.0]])  # Define two nodes with a known distance
        edge_index, edge_attr = compute_e3_invariant_edges(X)  # Call the function
        self.assertEqual(edge_index.shape[1], 1)  # Expect one edge
        self.assertEqual(edge_attr.shape[0], 1)    # Expect one edge attribute
        self.assertAlmostEqual(edge_attr[0].item(), 5.0)  # Distance should be 5.0 (3-4-0 triangle)

if __name__ == '__main__':
    unittest.main()  # Execute the unit tests