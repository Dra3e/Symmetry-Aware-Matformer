import torch

from training.loss_functions import MultiTaskLoss

def compute_loss(model, eval_loader, multi_loss, device, threshold):
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for graphs, target_values in eval_loader:
            # Move graphs and targets to device
            graphs = [[g.to(device) for g in g_list] for g_list in graphs]  # Move each graph to the device
            target_values = target_values.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # Forward pass
                results = [model(g_list) for g_list in graphs]
                batch_property_losses = []
                batch_contrastive_losses = []

                for i, (predictions, latents, graphs) in enumerate(results):
                    # Get the target value for this crystal
                    target = target_values[i]  # scalar

                    crystal_property_losses = []
                    crystal_contrastive_losses = []

                    # Use last graph as reference
                    latent_ref = latents[-1]
                    assert graphs[-1].m_valid == 1, "Reference graph must be valid!"
                    for j in range(len(graphs) - 1):  # skip reference graph
                        latent_j = latents[j]
                        pred_j = predictions[j]
                        target = target.expand_as(pred_j)

                        m_valid = torch.tensor(graphs[j].m_valid, dtype=torch.float32, device=device)

                        loss_property = MultiTaskLoss.LossProperty(target, pred_j)
                        loss_contrastive = MultiTaskLoss.LossContrastive(latent_ref, latent_j, m_valid, threshold)

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
                total = multi_loss(loss_property_batch, loss_contrastive_batch)

                total_loss += total.item()
                sample_count += 1

    return total_loss / sample_count if sample_count > 0 else float('inf')