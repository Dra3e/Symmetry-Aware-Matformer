import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Automatic Weighted Loss (AWL)  – always ≥ 0, learns σ_i
# ------------------------------------------------------------------
class AutomaticWeightedLoss(nn.Module):
    """
    Two-task homoscedastic-uncertainty weighting:
        ½·L_i / σ_i²  +  log(1 + σ_i²)
    The log1p term keeps the total loss non-negative and stops σ_i→0 collapse.
    """
    def __init__(self, n_tasks: int = 2):
        super().__init__()
        # log σ_i   (init σ_i = 1  ⇒  log σ_i = 0)
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *task_losses):
        assert len(task_losses) == len(self.log_sigma)
        loss_sum = 0.0
        for i, L in enumerate(task_losses):
            sigma2 = torch.exp(self.log_sigma[i]) ** 2          # σ_i² ≥ 0
            loss_sum += 0.5 * L / sigma2 + torch.log1p(sigma2)  # ½·L/σ² + log(1+σ²)
        return loss_sum


# ------------------------------------------------------------------
# 2. Property MSE (unchanged)
# ------------------------------------------------------------------
def LossProperty(actual_Y, predicted_Y):
    """
    Computes the Mean Squared Error (MSE) loss between actual and predicted values.

    Args:
        actual_Y (torch.Tensor): Ground truth values.
        predicted_Y (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: The MSE loss.
    """
    if actual_Y.shape == predicted_Y.shape:
        assert not torch.isnan(actual_Y).any()
        assert not torch.isnan(predicted_Y).any()
        return torch.mean((actual_Y - predicted_Y) ** 2)
    else:
        raise ValueError("Inputs are not the same size!")


# ------------------------------------------------------------------
# 3. Contrastive loss with safe ε-floor + soft margin
# ------------------------------------------------------------------
def LossContrastive_Vectorized(latent_ref, latents, flags, margin: float, tau: float = 0.1):
    """
    flags == 1  →  valid   (distance should be *small*)
    flags == 0  →  invalid (distance should be > margin)
    Softplus gives a smooth, strictly-positive hinge; add ε so L_con > 0.
    """
    if latents.numel() == 0:                       # edge case: batch had only the ref graph
        return latent_ref.new_tensor(1e-6)

    pos = latents[flags == 1]
    neg = latents[flags == 0]

    loss_pos = 0.0
    if pos.numel():
        loss_pos = (pos - latent_ref).pow(2).sum(dim=1).mean()

    loss_neg = 0.0
    if neg.numel():
        d = (neg - latent_ref).pow(2).sum(dim=1).sqrt()       # Euclidean
        loss_neg = F.softplus((margin - d) / tau).mean() # > 0

    return loss_pos + loss_neg + 1e-6    # ε-floor keeps σ_con well behaved