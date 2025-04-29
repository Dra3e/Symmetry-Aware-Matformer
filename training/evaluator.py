import torch
from training.loss_functions import LossContrastive, LossProperty, LossTotal

def compute_loss(model, eval_loader, factors, device, threshold):
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for crystals, target_values in eval_loader:
            # Move crystals and targets to device
            crystals = [c.to(device) for c in crystals]  # Move each crystal to the device
            target_values = target_values.to(device)

            # Forward pass
            results = [model(c) for c in crystals]
            batch_property_losses = []
            batch_contrastive_losses = []

            for i, (predictions, latents, graphs) in enumerate(results):
                # Get the target value for this crystal
                target = target_values[i]  # scalar

                crystal_property_losses = []
                crystal_contrastive_losses = []

                # Use first graph as reference
                latent_ref = latents[0]
                for j in range(1, len(graphs)):  # skip index 0 (reference)
                    latent_j = latents[j]
                    pred_j = predictions[j]
                    target = target.expand_as(pred_j)

                    m_valid = torch.tensor(graphs[j].m_valid, dtype=torch.float32, device=device)

                    loss_property = LossProperty(target, pred_j)
                    loss_contrastive = LossContrastive(latent_ref, latent_j, m_valid, threshold)

                    crystal_property_losses.append(loss_property)
                    crystal_contrastive_losses.append(loss_contrastive)

                # Average per-crystal loss (mean over graphs)
                mean_loss_property = torch.stack(crystal_property_losses).mean()
                mean_loss_contrastive = torch.stack(crystal_contrastive_losses).mean()

                batch_property_losses.append(mean_loss_property)
                batch_contrastive_losses.append(mean_loss_contrastive)

            # Average over the batch (mean over crystals)
            loss_property_batch = torch.stack(batch_property_losses).mean()
            loss_contrastive_batch = torch.stack(batch_contrastive_losses).mean()

            # Combine losses
            both_losses = torch.stack([loss_property_batch, loss_contrastive_batch])
            total = LossTotal(both_losses, factors)

            total_loss += total.item()
            sample_count += 1

    return total_loss / sample_count if sample_count > 0 else float('inf')