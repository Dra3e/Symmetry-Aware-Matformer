import numpy as np
import unittest
from components.translation import TranslateRand

class Test_Translation(unittest.TestCase):
    def test_Translation_CorrectInput(self):
        A_input = np.array([[1,1],[1,1]])
        X_input = np.array([[1,1,1],[1,1,1],[1,1,1]])
        L_input = np.array([[1,0,0],[0,1,0],[0,0,1]])
        direction = np.array([[1,1,1]])

        A_output,X_output,L_output = TranslateRand(A_input,X_input,L_input,direction)

        self.assertEqual(A_input.shape,A_output.shape)
        self.assertEqual(X_input.shape,X_output.shape)
        self.assertEqual(L_input.shape,L_output.shape)
        self.assertTrue((A_input == A_output).all())
        self.assertTrue((X_input != X_output).all())
        self.assertTrue((L_input == L_output).all())
    
    def test_Translation_WrongMatrixSize(self):
        A_input = np.array([[1,1],[1,1]])
        X_input = np.array([[1,1],[1,1],[1,1]])
        L_input = np.array([[1,0,0],[0,1,0],[0,0,1]])

        with self.assertRaises(Exception):
            A_output,X_output,L_output = TranslateRand(A_input,X_input,L_input)
    
    def test_Translation_WrongInput(self):
        A_input = np.array([[1,1],[1,1]])
        X_input = "fail"
        L_input = np.array([[1,0,0],[0,1,0],[0,0,1]])

        with self.assertRaises(Exception):
            A_output,X_output,L_output = TranslateRand("fail")
            