import torch
import unittest
from models.augmentations.reflection import Plane

class Test_Reflection(unittest.TestCase):

    # ================== reflection_matrix ==========================
    def test_reflection_matrix_valid(self):
        """Test reflection matrix with a valid normal vector."""
        normal = torch.tensor([1, 0, 0], dtype=torch.float32)
        plane = Plane(normal)
        R = plane.reflection_matrix()
        expected = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        torch.testing.assert_close(R, expected, rtol=1e-6, atol=1e-6)

    def test_reflection_matrix_zero_vector(self):
        """Test reflection matrix with a zero normal vector (should fail)."""
        normal = torch.tensor([0, 0, 0], dtype=torch.float32)
        with self.assertRaises(ValueError):
            plane = Plane(normal)
            plane.reflection_matrix()

    def test_reflection_matrix_non_normalized_input(self):
        """Test reflection matrix with a non-normalized input vector."""
        normal = torch.tensor([2, 0, 0], dtype=torch.float32)
        plane = Plane(normal)
        R = plane.reflection_matrix()
        expected = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        torch.testing.assert_close(R, expected, rtol=1e-6, atol=1e-6)

    # ================== reflect_crystal ==========================
    def test_reflect_crystal_correct_input(self):
        """Test reflection of a simple crystal structure."""
        A = torch.tensor([[1], [2]], dtype=torch.float32)  # Features remain unchanged
        X = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)  # Atomic positions
        L = torch.eye(3, dtype=torch.float32)  # Lattice vectors
        normal = torch.tensor([1, 0, 0], dtype=torch.float32)
        
        plane = Plane(normal)
        A_reflected, X_reflected, L_reflected = plane.reflect_crystal(A, X, L)
        
        expected_X_reflected = torch.tensor([[-1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        
        torch.testing.assert_close(X_reflected, expected_X_reflected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(L_reflected, L, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(A_reflected, A, rtol=1e-6, atol=1e-6)

    def test_reflect_crystal_with_plane_not_through_origin(self):
        """Test reflection across a plane not passing through origin."""
        A = torch.tensor([[1], [2]], dtype=torch.float32)
        X = torch.tensor([[2, 0, 0], [0, 1, 0]], dtype=torch.float32)  # Atomic positions
        L = torch.eye(3, dtype=torch.float32)
        normal = torch.tensor([1, 0, 0], dtype=torch.float32)
        point = torch.tensor([1, 0, 0], dtype=torch.float32)  # Plane at x=1
        
        plane = Plane(normal, point)
        A_reflected, X_reflected, L_reflected = plane.reflect_crystal(A, X, L)
        
        expected_X_reflected = torch.tensor([[0, 0, 0], [2, 1, 0]], dtype=torch.float32)
        
        torch.testing.assert_close(X_reflected, expected_X_reflected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(L_reflected, L, rtol=1e-6, atol=1e-6)

    def test_reflect_crystal_invalid_normal_vector(self):
        """Test reflect_crystal with an invalid normal vector (zero vector)."""
        A = torch.tensor([[1], [2]], dtype=torch.float32)
        X = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        L = torch.eye(3, dtype=torch.float32)
        normal = torch.tensor([0, 0, 0], dtype=torch.float32)  # Invalid normal vector
        
        with self.assertRaises(ValueError):
            plane = Plane(normal)
            plane.reflect_crystal(A, X, L)

if __name__ == '__main__':
    unittest.main()
