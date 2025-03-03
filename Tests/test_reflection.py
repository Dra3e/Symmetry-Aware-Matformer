import numpy as np
import unittest
from components.reflection import reflection_matrix, reflect_crystal

class Test_Reflection(unittest.TestCase):

    # ================== reflect_matrix ==========================
    def test_reflection_matrix_valid(self):
        """Test reflection matrix with a valid normal vector."""
        normal = [1, 0, 0]
        R = reflection_matrix(normal)
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_almost_equal(R, expected, decimal=6)

    def test_reflection_matrix_zero_vector(self):
        """Test reflection matrix with a zero normal vector (should fail)."""
        normal = [0, 0, 0]
        with self.assertRaises(ValueError):
            reflection_matrix(normal)

    def test_reflection_matrix_non_normalized_input(self):
        """Test reflection matrix with a non-normalized input vector."""
        normal = [2, 0, 0]
        R = reflection_matrix(normal)
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_almost_equal(R, expected, decimal=6)

    # ================== reflect_crystal ==========================
    def test_reflect_crystal_correct_input(self):
        """Test reflection of a simple crystal structure."""
        A = np.array([[1], [2]])  # Features remain unchanged
        X = np.array([[1, 0, 0], [0, 1, 0]])  # Atomic positions
        L = np.eye(3)  # Lattice vectors
        normal = [1, 0, 0]
        
        A_reflected, X_reflected, L_reflected = reflect_crystal(A, X, L, normal)
        
        expected_X_reflected = np.array([[-1, 0, 0], [0, 1, 0]])
        expected_L_reflected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        np.testing.assert_almost_equal(X_reflected, expected_X_reflected, decimal=6)
        np.testing.assert_almost_equal(L_reflected, expected_L_reflected, decimal=6)
        np.testing.assert_array_equal(A_reflected, A)

    def test_reflect_crystal_with_plane_not_through_origin(self):
        """Test reflection across a plane not passing through origin."""
        A = np.array([[1], [2]])
        X = np.array([[2, 0, 0], [0, 1, 0]])  # Atomic positions
        L = np.eye(3)
        normal = [1, 0, 0]
        p = [1, 0, 0]  # Plane at x=1
        
        A_reflected, X_reflected, L_reflected = reflect_crystal(A, X, L, normal, p)
        
        expected_X_reflected = np.array([[0, 0, 0], [2, 1, 0]])
        expected_L_reflected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        np.testing.assert_almost_equal(X_reflected, expected_X_reflected, decimal=6)
        np.testing.assert_almost_equal(L_reflected, expected_L_reflected, decimal=6)

    def test_reflect_crystal_invalid_normal_vector(self):
        """Test reflect_crystal with an invalid normal vector (zero vector)."""
        A = np.array([[1], [2]])
        X = np.array([[1, 0, 0], [0, 1, 0]])
        L = np.eye(3)
        normal = [0, 0, 0]  # Invalid normal vector
        
        with self.assertRaises(ValueError):
            reflect_crystal(A, X, L, normal)

if __name__ == '__main__':
    unittest.main()
    