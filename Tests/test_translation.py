import unittest
import numpy as np

from components.translation import TranslateRand

# Assuming TranslateRand function is defined in the same file or imported from another module

class TestTranslateRand(unittest.TestCase):

    def test_valid_translation(self):
        """Test the function with valid inputs and check the translation behavior."""
        A = np.array([[1, 2], [3, 4]])  # Atomic features
        X = np.array([[0, 0, 0], [1, 1, 1]])  # Atomic coordinates
        L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity matrix for lattice
        coefficients = np.array([1, 2, 3])  # Translation coefficients

        A_translated, X_translated, L_translated = TranslateRand(A, X, L, coefficients)

        # Verify that A and L remain unchanged
        np.testing.assert_array_equal(A, A_translated)
        np.testing.assert_array_equal(L, L_translated)

        # Verify the translated coordinates
        expected_translation_vector = np.dot(coefficients, L)
        expected_X_translated = X + expected_translation_vector
        np.testing.assert_array_equal(X_translated, expected_X_translated)

    def test_invalid_shape_X(self):
        """Test for invalid shape of the `X` input matrix."""
        A = np.array([[1, 2], [3, 4]])  # Atomic features
        X = np.array([[0, 0]])  # Invalid shape (1x2 instead of Nx3)
        L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coefficients = np.array([1, 2, 3])

        with self.assertRaises(ValueError) as context:
            TranslateRand(A, X, L, coefficients)
        self.assertEqual(str(context.exception), "X-matrix is not Nx3")

    def test_invalid_shape_L(self):
        """Test for invalid shape of the `L` lattice matrix."""
        A = np.array([[1, 2], [3, 4]])
        X = np.array([[0, 0, 0], [1, 1, 1]])
        L = np.array([[1, 0], [0, 1], [0, 0]])  # Invalid shape (3x2 instead of 3x3)
        coefficients = np.array([1, 2, 3])

        with self.assertRaises(ValueError) as context:
            TranslateRand(A, X, L, coefficients)
        self.assertEqual(str(context.exception), "Lattice matrix L must be 3x3")

    def test_invalid_coefficients_length(self):
        """Test for invalid length of the coefficients array."""
        A = np.array([[1, 2], [3, 4]])
        X = np.array([[0, 0, 0], [1, 1, 1]])
        L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coefficients = np.array([1, 2])  # Invalid length

        with self.assertRaises(ValueError) as context:
            TranslateRand(A, X, L, coefficients)
        self.assertEqual(str(context.exception), "Translation coefficients must be a 3-element integer array")

    def test_invalid_coefficients_type(self):
        """Test for invalid data type of coefficients."""
        A = np.array([[1, 2], [3, 4]])
        X = np.array([[0, 0, 0], [1, 1, 1]])
        L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coefficients = np.array([1.0, 2.0, 3.0])  # Must be integers

        with self.assertRaises(ValueError) as context:
            TranslateRand(A, X, L, coefficients)
        self.assertEqual(str(context.exception), "Translation coefficients must be a 3-element integer array")

    def test_invalid_input_type(self):
        """Test for invalid input types."""
        A = None  # Not a valid numpy array
        X = np.array([[0, 0, 0], [1, 1, 1]])
        L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coefficients = np.array([1, 2, 3])

        with self.assertRaises(ValueError) as context:
            TranslateRand(A, X, L, coefficients)
        self.assertEqual(str(context.exception), "A can't be none")

if __name__ == '__main__':
    unittest.main()  # Run the unit tests