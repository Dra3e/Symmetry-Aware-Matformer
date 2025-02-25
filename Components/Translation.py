import numpy
import random

def TranslateRand(matrix):
    try:
        if(matrix.shape[1]!=3):
            raise Exception("Matrix is not Nx3")
    except AttributeError:
        raise Exception("Invalide Input")
    else:
        newMatrix = numpy.empty([matrix.shape[0],3])
        randVector = numpy.array([[(random.randrange(-9999,9999) / 10000),
                                   (random.randrange(-9999,9999) / 10000),
                                   (random.randrange(-9999,9999) / 10000)]])
        for r in range(matrix.shape[0]):
            newMatrix[r] = numpy.add(matrix[r],randVector)
        return newMatrix