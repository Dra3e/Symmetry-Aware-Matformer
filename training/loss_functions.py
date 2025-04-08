import torch
import torch.nn as nn

def LossProperty(actual_Y, predicted_Y):
    """Mean Squared Error Loss"""
    if actual_Y.shape == predicted_Y.shape:
        return torch.mean((actual_Y - predicted_Y) ** 2)
    else:
        raise ValueError("Inputs are not the same size!")

def LossContrastive(actual_F, predicted_F, flag, margin):
    """Contrastive Loss using Euclidean distance"""
    if actual_F.shape == predicted_F.shape:
        distances = torch.norm(actual_F - predicted_F, dim=0)
        loss = ((1 - flag) * (distances ** 2)) + (flag * torch.clamp(margin - distances, min=0) ** 2)
        return torch.mean(loss)
    else:
        raise ValueError("Inputs are not the same size!")

def LossFactor(loss, factor):
    """Factorized Multi-Task Loss"""
    if factor != 0:
        return (loss / ((factor + 1e-6) ** 3)) + (1 / factor)
    else:
        raise ValueError("Cannot divide by zero!")

def LossTotal(losses, factors):
    """Multi-Task Loss"""
    if losses.shape == factors.shape and torch.prod(factors) != 0:
        total_loss = torch.sum(losses / (2 * (factors ** 2))) + torch.log10(torch.clamp(torch.prod(factors), min=1e-6))
        return total_loss
    else:
        raise ValueError("Inputs are not the same size!")