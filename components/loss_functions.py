import numpy as np
import math as m
from numpy.linalg import norm

def LossProperty(actual_Y, predicted_Y):
    """
    Calculates the loss using Mean Square Error function.

    Keyword arguments:
    - actual_Y: list of every property vector produced by the neural network.
    - predicted_Y: list of every expected property vector.
    
    Returns:
    - loss: the calculated loss.
    """
    if len(actual_Y) == len(predicted_Y) and actual_Y.shape == predicted_Y.shape:
        loss = 0
        for i in actual_Y:
            loss += (actual_Y[i] - predicted_Y[i]) ** 2
        loss = loss / len(actual_Y)
        return loss
    else:
        raise Exception("Inputs are not the same size!")

def LossContrastive(actual_F, predicted_F, flag_list, margin):
    """
    Calculates the contrastive loss. Uses Euclidian distance to calculate distance in latent space.

    Keyword arguments:
    - actual_F: list of every latent space vector produced by the neural network.
    - predicted_F: list of every expected latent space vector.
    - flag_list: list of flags, denoting whether the actual_F has correct symmetry (0) or not (1).
    - margin: a value representing the minimum distance in latent space that points need to be appart to count as being seperate clusters.
    
    Returns:
    - loss: the calculated loss.
    """
    if len(actual_F) == len(predicted_F) == len(flag_list) and actual_F.shape == predicted_F.shape:
        loss = 0
        for i in range(actual_F.shape[0]):
            loss += ((1 - flag_list[i]) * (np.linalg.norm(actual_F[i] - predicted_F[i]) ** 2)) + (flag_list[i] * (max(0, margin - np.linalg.norm(actual_F[i] - predicted_F[i])) ** 2))
        return loss
    else:
        raise Exception("Inputs are not the same size!")

def LossFactor(loss, factor):
    """
    Calculates the loss of a factor used in a learnable multi-task loss function.

    Keyword arguments:
    - loss: the calulated loss of the loss function associated with the factor.
    - factor: the weight parameter used with the loss function.
    
    Returns:
    - loss: the calculated loss.
    """
    if factor != 0:
        return (loss / (factor ** 3)) + (1 / factor)
    else:
        raise Exception("Cannot divide by zero!")

def LossTotal(losses, factors):
    """
    Calculates the total loss of a neural network using learnable multi-task loss function.

    Keyword arguments:
    - losses: list of the losses calculated by the different loss functions used in the learnable multi-task loss function.
    - factors: list of weight parameters for every loss function used in the learnable multi-task loss function.
    
    Returns:
    - loss: the calculated total loss.
    """
    if len(losses) == len(factors) and losses.shape == factors.shape and np.prod(factors) != 0:
        total_loss = 0
        for i in losses:
            total_loss += losses[i] / (2 * (factors[i] ** 2))
        total_loss += m.log10(np.prod(factors))
        return total_loss
    else:
        raise Exception("Inputs are not the same size!")