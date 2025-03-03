import numpy as np
import unittest
from components.rotation import rotate_3d_object  # Replace with the actual module name

class TestRotate3DObject(unittest.TestCase):
    def test_identity_rotation(self): # Tests with no rotation, which should result in the matrix being unchanged
        A = np.array([[1, 2], [3, 4]])  # Example atomic features (remains unchanged)
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Unit vectors
        L = np.eye(3)  # Identity matrix for lattice
        axis = [0, 0, 1]
        angle = 0  # No rotation
        A_rot, X_rot, L_rot = rotate_3d_object(A, X, L, axis, angle)
        np.testing.assert_array_equal(X, X_rot)
        np.testing.assert_array_equal(L, L_rot)
    
    def test_full_rotation(self): # Tests a full 360-degree rotation, which should return the original matrix
        X = np.array([[1, 0, 0]])
        L = np.eye(3)
        axis = [0, 0, 1]
        angle = 360 # Full rotation
        _, X_rot, L_rot = rotate_3d_object(None, X, L, axis, angle) 
        np.testing.assert_array_almost_equal(X, X_rot, decimal=6)
        np.testing.assert_array_almost_equal(L, L_rot, decimal=6)
    
    def test_rotation_90_degrees(self): # Tests a 90-degree rotation around the z-axis to check correctness
        X = np.array([[1, 0, 0]])
        L = np.eye(3)
        axis = [0, 0, 1]
        angle = 90
        _, X_rot, _ = rotate_3d_object(None, X, L, axis, angle)
        expected_X_rot = np.array([[0, 1, 0]])
        np.testing.assert_array_almost_equal(X_rot, expected_X_rot, decimal=6)
    
    def test_rotation_around_non_orthogonal_axis(self): # Tests rotation around a non-orthogonal axis to verify proper handling
        X = np.array([[1, 1, 1]])
        L = np.eye(3)
        axis = [1, 1, 1]
        angle = 120
        _, X_rot, _ = rotate_3d_object(None, X, L, axis, angle)
        expected_X_rot = np.array([[1, 1, 1]])  # Should remain the same as it's along the axis
        np.testing.assert_array_almost_equal(X_rot, expected_X_rot, decimal=6)
    
    def test_distance_preservation(self): # Ensures distances between points are preserved after rotation
        X = np.array([[1, 0, 0], [0, 1, 0]])
        L = np.eye(3)
        axis = [0, 0, 1]
        angle = 45
        _, X_rot, _ = rotate_3d_object(None, X, L, axis, angle)
        dist_before = np.linalg.norm(X[0] - X[1])
        dist_after = np.linalg.norm(X_rot[0] - X_rot[1])
        self.assertTrue(np.isclose(dist_before, dist_after, atol=1e-6))
    
    def test_empty_input(self): # Tests rotation on an empty input to ensure graceful handling
        X = np.array([]).reshape(0, 3)
        L = np.eye(3)
        axis = [0, 0, 1]
        angle = 90
        _, X_rot, _ = rotate_3d_object(None, X, L, axis, angle)
        self.assertEqual(X_rot.shape, (0, 3))  # Should return an empty array
    
    def test_single_point_rotation(self): # Tests rotating a single point to verify correctness
        X = np.array([[1, 2, 3]])
        L = np.eye(3)
        axis = [0, 1, 0]
        angle = 180
        _, X_rot, _ = rotate_3d_object(None, X, L, axis, angle)
        expected_X_rot = np.array([[-1, 2, -3]])
        np.testing.assert_array_almost_equal(X_rot, expected_X_rot, decimal=6)
        