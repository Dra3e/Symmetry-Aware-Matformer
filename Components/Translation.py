import numpy
import random

def TranslateRand(matrix):
    try:
        newMatrix = numpy.empty([matrix.shape[0],3])
    except:
        raise Exception("Invalide Input")
    else:
        randVector = numpy.array([[(random.randrange(-9999,9999) / 10000),(random.randrange(-9999,9999) / 10000),(random.randrange(-9999,9999) / 10000)]])
        for r in range(matrix.shape[0]):
            newMatrix[r] = numpy.add(newMatrix[r],randVector)
        return newMatrix