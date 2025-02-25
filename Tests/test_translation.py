import numpy
import unittest
from Components.Translation import TranslateRand

class Test_Translation(unittest.TestCase):
    def test_Translation_CorrectInput(self):
        baseMatrix = numpy.array([[1,1,1],[1,1,1],[1,1,1]])
        testMatrix = TranslateRand(baseMatrix)
        self.assertEqual(baseMatrix.shape,testMatrix.shape)
        self.assertTrue((baseMatrix != testMatrix).all())
    
    def test_Translation_WrongMatrixSize(self):
        baseMatrix = numpy.array([[1,1],[1,1],[1,1]])
        with self.assertRaises(Exception):
            testMatrix = TranslateRand(baseMatrix)
    
    def test_Translation_WrongInput(self):
        with self.assertRaises(Exception):
            testMatrix = TranslateRand("fail")