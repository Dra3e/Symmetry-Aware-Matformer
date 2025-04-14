import torch
from training.loss_functions import LossContrastive, LossProperty, LossTotal

def compute_loss(model, eval_loader, factors, device, threshold):
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for crystals, target_values in eval_loader:
            target_values = target_values.to(device)

            results = [model(c) for c in crystals]
            predictions, latentspace_f, graphs_batch = zip(*results)
            predictions = torch.stack(predictions)
            latentspace_f = torch.stack(latentspace_f)
            graphs_batch = list(graphs_batch)

            all_loss_property = []
            all_loss_contrastive = []

            for i in range(1, latentspace_f.shape[1]):  # Skip reference (index 0)
                latent0 = latentspace_f[:, 0, :]       # [B, D]
                latenti = latentspace_f[:, i, :]       # [B, D]
                pred_i = predictions[:, i].view(-1)             # [B]

                m_valid = torch.tensor([g[i].m_valid for g in graphs_batch], dtype=torch.float32, device=device)

                loss_property = LossProperty(target_values, pred_i)
                loss_contrastive = LossContrastive(latent0, latenti, m_valid, threshold)

                all_loss_property.append(loss_property)
                all_loss_contrastive.append(loss_contrastive)

            loss_property_batch = torch.stack(all_loss_property).mean()
            loss_contrastive_batch = torch.stack(all_loss_contrastive).mean()

            both_losses = torch.tensor([loss_property_batch, loss_contrastive_batch], device=device)
            total = LossTotal(both_losses, factors)

            total_loss += total.item()
            sample_count += 1

    return total_loss / sample_count if sample_count > 0 else float('inf')