import torch
import torch.nn.functional as F

class MultiTaskLoss(torch.nn.Module):
    """
    Multi-task loss function based on task uncertainty.
    Each task is weighted by a learnable log variance (log σ²).
    This allows the model to learn an appropriate weighting for each task during training.
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018).
    """
    def __init__(self, n_tasks: int = 2):
        """
        Initializes the MultiTaskLoss module.

        Args:
            n_tasks (int): Number of tasks to balance. Default is 2.
        """
        super().__init__()
        # Initialize log σ² for each task to 0 (σ = 1), giving equal influence to all tasks initially.
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks, requires_grad=True))

    def forward(self, *task_losses: torch.Tensor) -> torch.Tensor:
        """
        Computes the total loss by combining individual task losses with learnable weights.

        Args:
            *task_losses (torch.Tensor): Losses for each task.

        Returns:
            torch.Tensor: The combined loss.
        """
        losses = torch.stack(task_losses)          # [T]
        
        # Ensure no NaN or Inf values in task losses
        assert not torch.isnan(losses).any(), "NaN detected in task losses!"
        assert not torch.isinf(losses).any(), "Inf detected in task losses!"

        # Clamp log_vars to prevent extreme values
        clamped_vars = torch.clamp(self.log_vars, min=-20, max=20)
        
        precision = torch.exp(-clamped_vars)     # e^{-s_i}

        # Ensure no NaN or Inf values in precision
        assert not torch.isnan(precision).any(), "NaN detected in precision!"
        assert not torch.isinf(precision).any(), "Inf detected in precision!"
        
        total = 0.5 * torch.sum(precision * losses + clamped_vars)

        # Ensure no NaN or Inf values in the total loss
        assert not torch.isnan(total).any(), "NaN detected in total loss!"
        assert not torch.isinf(total).any(), "Inf detected in total loss!"

        return total
    
    @staticmethod
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

    @staticmethod
    def LossContrastive(actual_F, predicted_F, flag, threshold):
        """
        Computes the contrastive loss using Euclidean distance.

        Args:
            actual_F (torch.Tensor): Ground truth feature vectors.
            predicted_F (torch.Tensor): Predicted feature vectors.
            flag (torch.Tensor): Binary tensor indicating positive (1) or negative (0) pairs.
            threshold (float): Margin for negative pairs.

        Returns:
            torch.Tensor: The contrastive loss.
        """
        assert actual_F.shape == predicted_F.shape, "Inputs are not the same size!"
        # assert actual_F.ndim == 2, "Inputs must be 2D tensors!"
        # assert flag.ndim == 1, "Flag must be a 1D tensor!"
        # assert actual_F.size(0) == flag.size(0), "Flag size must match the batch size!"

        distances = torch.norm(actual_F - predicted_F, dim=0) # Compute Euclidean distances between actual and predicted feature vectors

        # Ensure no NaN or Inf values in distances
        assert not torch.isnan(distances).any(), "NaN detected in distances!"
        assert not torch.isinf(distances).any(), "Inf detected in distances!"

        # Compute positive and negative losses
        positive_loss = flag * (distances ** 2)
        negative_loss = (1 - flag) * torch.clamp(threshold - distances, min=0.0) ** 2

        loss = positive_loss + negative_loss 

        return loss.mean()
    
    @staticmethod
    def LossContrastive_Vectorized(latent_ref, latents, flags, margin):
        if latents.numel() == 0:                      # nothing to compare with
            # Return *zero* so the other task still back‑propagates
            return torch.zeros(1, device=latent_ref.device, requires_grad=True)

        pos = latents[flags == 1]
        neg = latents[flags == 0]

        dist_pos = torch.norm(pos - latent_ref, dim=1) if pos.numel() else None
        dist_neg = torch.norm(neg - latent_ref, dim=1) if neg.numel() else None

        loss_pos = dist_pos.pow(2).mean() if dist_pos is not None else 0.0
        loss_neg = torch.clamp(margin - dist_neg, min=0).pow(2).mean() \
                if dist_neg is not None else 0.0

        return (loss_pos + loss_neg)        # always finite; gradients flow
