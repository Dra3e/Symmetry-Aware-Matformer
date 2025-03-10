import numpy as np
import unittest
from components.loss_functions import LossProperty, LossContrastive, LossFactor, LossTotal

class Test_Loss_Functions(unittest.TestCase):
    def test_LossProperty_CorrectInput(self):
        Y_input = np.array([1,1,1,1])
        Y_predicted = np.array([1,1,1,1])
        expected_loss = 0

        calculated_loss = LossProperty(Y_input, Y_predicted)

        self.assertEqual(expected_loss, calculated_loss)

    def test_LossProperty_WrongSize(self):
        Y_input = np.array([1,1,1,1])
        Y_predicted = np.array([1,1,1])
        
        with self.assertRaises(Exception):
            calculated_loss = LossProperty(Y_input, Y_predicted)
    
    def test_LossProperty_WrongDimension(self):
        Y_input = np.array([1,1,1,1])
        Y_predicted = np.array([[1,1],[1,1]])
        
        with self.assertRaises(Exception):
            calculated_loss = LossProperty(Y_input, Y_predicted)
    
    def test_LossContrastive_CorrectInput(self):
        F_input = np.array([[1,1,1,1]])
        F_predicted = np.array([[1,1,1,1]])
        margin = 3
        expected_loss = 0

        calculated_loss = LossContrastive(F_input, F_predicted, margin, 1)

        self.assertEqual(expected_loss, calculated_loss)
    
    def test_LossFactor_CorrectInput(self):
        factor_loss = 0
        factor = 1
        expected_loss = 0

        calculated_loss = LossFactor(factor_loss, factor)

        self.assertEqual(expected_loss, calculated_loss)

    def test_LossTotal_CorrectInput(self):
        losses = np.array([[1,1,1,1]])
        factors = np.array([[1,1,1,1]])
        expected_loss = 4

        calculated_loss = LossTotal(losses, factors)

        self.assertEqual(expected_loss, calculated_loss)