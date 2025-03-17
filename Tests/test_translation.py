import unittest
import torch
from models.components.crystal import Crystal

class Test_Translation(unittest.TestCase):

    def test_valid_translation(self):
        """Test the function with valid inputs and check the translation behavior."""
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # Atomic features
        X = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)  # Atomic coordinates
        L = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)  # Identity matrix for lattice
        coefficients = torch.tensor([1, 2, 3], dtype=torch.int32)  # Translation coefficients

        crystal = Crystal(A, X, L)
        A_translated, X_translated, L_translated = crystal.translate_crystal(coefficients)

        # Verify that A and L remain unchanged
        torch.testing.assert_close(A, A_translated, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(L, L_translated, rtol=1e-6, atol=1e-6)

        # Verify the translated coordinates
        expected_translation_vector = torch.matmul(coefficients.float(), L)
        expected_X_translated = X + expected_translation_vector
        torch.testing.assert_close(X_translated, expected_X_translated, rtol=1e-6, atol=1e-6)

    def test_invalid_coefficients_length(self):
        """Test for invalid length of the coefficients array."""
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        X = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        L = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        coefficients = torch.tensor([1, 2], dtype=torch.int32)  # Invalid length

        crystal = Crystal(A, X, L)
        with self.assertRaises(ValueError) as context:
            crystal.translate_crystal(coefficients)
        self.assertEqual(str(context.exception), "Translation coefficients must be a 3-element integer tensor")

    def test_invalid_coefficients_type(self):
        """Test for invalid data type of coefficients."""
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        X = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        L = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        coefficients = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # Must be integers

        crystal = Crystal(A, X, L)
        with self.assertRaises(ValueError) as context:
            crystal.translate_crystal(coefficients)
        self.assertEqual(str(context.exception), "Translation coefficients must be a 3-element integer tensor")

if __name__ == '__main__':
    unittest.main()  # Run the unit tests