import numpy as np
from numpy.linalg import norm

def LossProperty(actual_Y, predicted_Y):
    if len(actual_Y) == len(predicted_Y) and actual_Y.shape == predicted_Y.shape:
        loss = 0
        for i in actual_Y:
            loss += (actual_Y[i] - predicted_Y[i]) ** 2
        loss = loss / len(actual_Y)
        return loss
    else:
        raise Exception("Inputs are not the same size!")

def LossContrastive(actual_F, predicted_F, flag_list, margin):
    if len(actual_F) == len(predicted_F) == len(flag_list) and actual_F.shape == predicted_F.shape:
        loss = 0
        for i in range(actual_F.shape[0]):
            loss += ((1 - flag_list[i]) * (np.linalg.norm(actual_F[i] - predicted_F[i]) ** 2)) + (flag_list[i] * (max(0, margin - np.linalg.norm(actual_F[i] - predicted_F[i])) ** 2))
        return loss
    else:
        raise Exception("Invalid distance function!")

def LossFactor(loss, factor):
    raise Exception("Not Implemented")

def LossTotal(loss, factor):
    raise Exception("Not Implemented")