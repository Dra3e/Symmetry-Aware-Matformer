import numpy as np
from components.translation import TranslateRand

if __name__ == "__main__":
    baseMatrix = np.array([[1,1,1],[1,1,1],[1,1,1]])
    testMatrix = TranslateRand(baseMatrix)
    for vector in testMatrix:
        print(vector)
