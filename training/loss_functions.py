import torch

class MultiTaskLoss(torch.nn.Module):
    """
    Kendall et al., CVPR 2018 — learnable uncertainty weights
        L = Σ (½·e^{-s_i}·L_i + ½·s_i)
      We store s_i = log σ_i²  to keep σ_i > 0 automatically.
    """
    def __init__(self, n_tasks: int = 2):
        super().__init__()
        # initialise each log σ² at 0  →  σ = 1  →  equal influence
        # self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks, requires_grad=True))

    def forward(self, *task_losses: torch.Tensor) -> torch.Tensor:
        losses = torch.stack(task_losses)          # [T]
        
        assert not torch.isnan(losses).any(), "NaN detected in task losses!"
        assert not torch.isinf(losses).any(), "Inf detected in task losses!"

        clamped_vars = torch.clamp(self.log_vars, min=-20, max=20)
        
        precision = torch.exp(-clamped_vars)     # e^{-s_i}

        assert not torch.isnan(precision).any(), "NaN detected in precision!"
        assert not torch.isinf(precision).any(), "Inf detected in precision!"
        
        total = 0.5 * torch.sum(precision * losses + clamped_vars)

        assert not torch.isnan(total).any(), "NaN detected in total loss!"
        assert not torch.isinf(total).any(), "Inf detected in total loss!"

        return total
    
    @staticmethod
    def LossProperty(actual_Y, predicted_Y):
        """Mean Squared Error Loss"""
        if actual_Y.shape == predicted_Y.shape:
            return torch.mean((actual_Y - predicted_Y) ** 2)
        else:
            raise ValueError("Inputs are not the same size!")

    @staticmethod
    def LossContrastive(actual_F, predicted_F, flag, threshold):
        """Contrastive Loss using Euclidean distance"""
        assert actual_F.shape == predicted_F.shape, "Inputs are not the same size!"
        # assert actual_F.ndim == 2, "Inputs must be 2D tensors!"
        # assert flag.ndim == 1, "Flag must be a 1D tensor!"
        # assert actual_F.size(0) == flag.size(0), "Flag size must match the batch size!"

        distances = torch.norm(actual_F - predicted_F, dim=0)  # Compute distances per sample

        assert not torch.isnan(distances).any(), "NaN detected in distances!"
        assert not torch.isinf(distances).any(), "Inf detected in distances!"
        # Compute contrastive loss
        positive_loss = flag * (distances ** 2)
        negative_loss = (1 - flag) * torch.clamp(threshold - distances, min=0.0) ** 2
        loss = positive_loss + negative_loss

        return loss.mean()
        
        # if actual_F.shape == predicted_F.shape:
        #     print("Flag:", flag)
        #     distances = torch.norm(actual_F - predicted_F, dim=0)
            
        #     assert not torch.isnan(distances).any(), "NaN detected in distances!"
        #     assert not torch.isinf(distances).any(), "Inf detected in distances!"
            
        #     return (flag * (distances ** 2)) + ((1 - flag) * torch.clamp(threshold - distances, min=0.0) ** 2)
        # else:
        #     raise ValueError("Inputs are not the same size!")

# def LossFactor(loss, factor):
#     """Factorized Multi-Task Loss"""
#     if factor != 0:
#         return (loss / ((factor + 1e-6) ** 3)) + (1 / factor)
#     else:
#         raise ValueError("Cannot divide by zero!")

# def LossTotal(losses, factors):
#     """Multi-Task Loss"""
#     if losses.shape == factors.shape:
#         factors = torch.clamp(factors, min=1e-3, max=1e3)
#         losses = torch.clamp(losses, min=1e-4, max=1e2)
#         prod_factors = torch.clamp(torch.prod(factors), min=1e-3)
#         total_loss = torch.sum(losses / (2 * (factors ** 2))) + torch.log10(prod_factors)
#         return total_loss
#     else:
#         raise ValueError("Inputs are not the same size!")