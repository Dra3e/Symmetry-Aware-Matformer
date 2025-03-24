import torch
import unittest
from models.augmentations.rotation import Axis

class Test_Rotation(unittest.TestCase):
    def test_identity_rotation(self):
        """Tests with no rotation, which should result in the matrix being unchanged."""
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # Example atomic features (remains unchanged)
        X = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)  # Unit vectors
        L = torch.eye(3, dtype=torch.float32)  # Identity matrix for lattice
        axis_vector = torch.tensor([0, 0, 1], dtype=torch.float32)
        rotation_degrees = 0  # No rotation
        axis = Axis(axis_vector, rotation_degrees)
        A_rot, X_rot, L_rot = axis.rotate_crystal(A, X, L)
        torch.testing.assert_close(X, X_rot, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(L, L_rot, rtol=1e-6, atol=1e-6)
    
    def test_full_rotation(self):
        """Tests a full 360-degree rotation, which should return the original matrix."""
        X = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        axis_vector = torch.tensor([0, 0, 1], dtype=torch.float32)
        rotation_degrees = 360  # Full rotation
        axis = Axis(axis_vector, rotation_degrees)
        _, X_rot, L_rot = axis.rotate_crystal(None, X, L)
        torch.testing.assert_close(X, X_rot, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(L, L_rot, rtol=1e-6, atol=1e-6)
    
    def test_rotation_90_degrees(self):
        """Tests a 90-degree rotation around the z-axis to check correctness."""
        X = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        axis_vector = torch.tensor([0, 0, 1], dtype=torch.float32)
        rotation_degrees = 90
        axis = Axis(axis_vector, rotation_degrees)
        _, X_rot, _ = axis.rotate_crystal(None, X, L)
        expected_X_rot = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        torch.testing.assert_close(X_rot, expected_X_rot, rtol=1e-6, atol=1e-6)
    
    def test_rotation_around_non_orthogonal_axis(self):
        """Tests rotation around a non-orthogonal axis to verify proper handling."""
        X = torch.tensor([[1, 1, 1]], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        axis_vector = torch.tensor([1, 1, 1], dtype=torch.float32)
        rotation_degrees = 120
        axis = Axis(axis_vector, rotation_degrees)
        _, X_rot, _ = axis.rotate_crystal(None, X, L)
        expected_X_rot = torch.tensor([[1, 1, 1]], dtype=torch.float32)  # Should remain the same as it's along the axis
        torch.testing.assert_close(X_rot, expected_X_rot, rtol=1e-6, atol=1e-6)
    
    def test_distance_preservation(self):
        """Ensures distances between points are preserved after rotation."""
        X = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        axis_vector = torch.tensor([0, 0, 1], dtype=torch.float32)
        rotation_degrees = 45
        axis = Axis(axis_vector, rotation_degrees)
        _, X_rot, _ = axis.rotate_crystal(None, X, L)
        dist_before = torch.norm(X[0] - X[1])
        dist_after = torch.norm(X_rot[0] - X_rot[1])
        self.assertTrue(torch.isclose(dist_before, dist_after, atol=1e-6))
    
    def test_empty_input(self):
        """Tests rotation on an empty input to ensure graceful handling."""
        X = torch.tensor([]).reshape(0, 3)
        L = torch.eye(3, dtype=torch.float32)
        axis_vector = torch.tensor([0, 0, 1], dtype=torch.float32)
        rotation_degrees = 90
        axis = Axis(axis_vector, rotation_degrees)
        _, X_rot, _ = axis.rotate_crystal(None, X, L)
        self.assertEqual(X_rot.shape, (0, 3))  # Should return an empty array
    
    def test_single_point_rotation(self):
        """Tests rotating a single point to verify correctness."""
        X = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        axis_vector = torch.tensor([0, 1, 0], dtype=torch.float32)
        rotation_degrees = 180
        axis = Axis(axis_vector, rotation_degrees)
        _, X_rot, _ = axis.rotate_crystal(None, X, L)
        expected_X_rot = torch.tensor([[-1, 2, -3]], dtype=torch.float32)
        torch.testing.assert_close(X_rot, expected_X_rot, rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
