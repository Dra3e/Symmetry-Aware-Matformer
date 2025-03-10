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

    def test_LossProperty_WrongInput(self):
        Y_input = np.array([1,1,1,1])
        wrong_dimension_predicted = np.array([[1,1],[1,1]])
        wrong_size_predicted = np.array([1,1,1])
        
        with self.assertRaises(Exception):
            calculated_loss = LossProperty(Y_input, wrong_dimension_predicted)
        with self.assertRaises(Exception):
            calculated_loss = LossProperty(Y_input, wrong_size_predicted)
    
    def test_LossContrastive_CorrectInput(self):
        F_input = np.array([[1,1,1,1],[1,1,1,1]])
        F_predicted = np.array([[1,1,1,1],[3,3,3,3]])
        Augmentation_Label = np.array([0,1])
        margin = 3
        expected_loss = 0

        calculated_loss = LossContrastive(F_input, F_predicted, Augmentation_Label, margin)

        self.assertEqual(expected_loss, calculated_loss)
    
    def test_LossContrastive_WrongInput(self):
        F_input = np.array([[1,1,1,1],[1,1,1,1]])
        wrong_dimension_predicted = np.array([1,1,1,1,3,3,3,3])
        wrong_size_predicted = np.array([[1,1,1],[3,3,3]])
        Augmentation_Label = np.array([0,1])
        margin = 3

        with self.assertRaises(Exception):
            calculated_loss = LossContrastive(F_input, wrong_dimension_predicted, Augmentation_Label, margin)
        with self.assertRaises(Exception):
            calculated_loss = LossContrastive(F_input, wrong_size_predicted, Augmentation_Label, margin)
    
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