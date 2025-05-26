import torch
import torch.nn as nn
import torch.nn.functional as F

class AutomaticWeightedLoss(nn.Module):
    """
    Multi-task loss function based on task uncertainty.
    Each task is weighted by a learnable log variance (log σ²).
    This allows the model to learn an appropriate weighting for each task during training.
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018).

    Attributes:
        log_vars (torch.nn.Parameter): Learnable parameters representing the log variance for each task.
    """
    def __init__(self, n_tasks: int = 2):
        """
        Initializes the Automatic Weighted Loss.
        
        Args:
            n_tasks (int): The number of loss function that will be executed / how many parameters that should be initialized.
        """
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *task_losses):
        """
        Performs an automattically weighted loss calculation.
        
        loss = 1/2 ( SUM_i( (task_loss_i / log_sigma_i) + ln(1 + log_sigma_i) ) )
        
        Args:
            *task_losses (torch.Tensor): Variable number of scalar losses, one per task.
            
        Returns:
            torch.Tensor: A single scalar tensor representing the combined weighted loss.
        """
        assert len(task_losses) == len(self.log_sigma)          # Ensure there is a log_sigma for each task_loss.
        loss_sum = 0.0
        for i, L in enumerate(task_losses):                     # Calculate AWL for all inputted task_losses.
            sigma2 = torch.exp(self.log_sigma[i]) ** 2          # Upgrade to the exponential sigma.
            loss_sum += 0.5 * L / sigma2 + torch.log1p(sigma2)  # Calculate the loss and add it to the sum.
        return loss_sum


def LossProperty(actual_Y, predicted_Y):
    """
    Computes the Mean Squared Error (MSE) loss between actual and predicted values.

    loss = SUM( (actual - predicted)^2 )

    Args:
        actual_Y (torch.Tensor): Ground truth values.
        predicted_Y (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Tensor of shape (1, 1) containing the MSE loss.
    """
    if actual_Y.shape == predicted_Y.shape:
        assert not torch.isnan(actual_Y).any()
        assert not torch.isnan(predicted_Y).any()
        return torch.mean((actual_Y - predicted_Y) ** 2)
    else:
        raise ValueError("Inputs are not the same size!")

def LossContrastive_Vectorized(latent_ref, latents, flags, margin: float, tau: float = 0.1):
    """
    Computes the contrastive loss using Euclidean distance in a vectorized manner.

    loss = SUM( flag * Dist(reference, positive)^2 + flag * Max(0, margin - Dist(reference, negative))^2  )

    Args:
        latent_ref (torch.Tensor): Reference latent vector.
        latents (torch.Tensor): Latent vectors to compare against.
        flags (torch.Tensor): Binary tensor indicating positive (1) or negative (0) pairs.
        margin (float): Minimum distance between positive and negative pairs in latent space.
        tau (float): A scale used to approximate Softplus towards Max. Smaller value equals closer to Max.
    
    Returns:
        torch.Tensor: Tensor of shape (1, 1) containing the contrastive loss.
    
    """
    if latents.numel() == 0:                       # edge case: batch had only the ref graph
        return latent_ref.new_tensor(1e-6)

    pos = latents[flags == 1] # Get all validly augmented graphs
    neg = latents[flags == 0] # Get all invalidly augmented graphs

    loss_pos = 0.0
    if pos.numel():
        loss_pos = (pos - latent_ref).pow(2).sum(dim=1).mean() # Calculate positive side loss on valid graphs

    loss_neg = 0.0
    if neg.numel():
        distance = (neg - latent_ref).pow(2).sum(dim=1).sqrt() # Calculate euclidean distance between reference and invalid graph 
        loss_neg = F.softplus((margin - distance) / tau).mean() # Calculate negative side loss on valid graphs

    return loss_pos + loss_neg + 1e-6 # Get the complete contrastive loss