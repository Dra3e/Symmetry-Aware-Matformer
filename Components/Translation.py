import numpy
import random

def TranslateRand(matrix):
    try:
        randMatrix = numpy.empty([matrix.shape[0],matrix.shape[1]])
    except:
        raise Exception("Invalide Input")
    else:
        for r in range((matrix.shape[0] - 1)):
            for c in range((matrix.shape[1] - 1)):
                randInt = random.randrange(-9999,9999)
                randNr = randInt / 10000
                randMatrix[r][c] = randNr
        return numpy.add(matrix,randMatrix)