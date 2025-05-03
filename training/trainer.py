import os
import torch
import datetime

from torch.utils.data import DataLoader
from models.Transformer import Transformer  
from training.loss_functions import MultiTaskLoss
from training.dataset import get_training_data_from_pkl, CrystalGraphDataset
from training.evaluator import compute_loss
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

def graph_collate(batch):
    graphs, targets = zip(*batch)          # lists of length B
    return list(graphs), default_collate(targets)

def run(property, training_path: str, validation_path: str, num_epochs = 100, batch_size = 25, learning_rate = 1e-4, threshold = 100.0):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    num_workers = max(2, os.cpu_count() // 2) if torch.cuda.is_available() else 0

    # Initialize model, optimizer, and loss funciton
    model = Transformer().to(device) # Assuming Transformer is defined in models.Transformer
    multi_loss = MultiTaskLoss(n_tasks=2).to(device)
    model_optimizer = torch.optim.Adam(list(model.parameters()) + list(multi_loss.parameters()), lr=learning_rate) # Optimizer for the model

    # Create dataset
    training_crystals, training_target_properties = get_training_data_from_pkl(property, training_path) # Assuming get_training_data() returns a dataset and property value lit compatible with the model
    assert all(c is not None for c in training_crystals), "❌ 'training_crystals' contains None values!"
    assert all(tp is not None for tp in training_target_properties), "❌ 'target_properties' contains None values!"
    eval_crystals, eval_target_properties = get_training_data_from_pkl(property, validation_path)
    assert all(c is not None for c in eval_crystals), "❌ 'eval_crystals' contains None values!"
    assert all(tp is not None for tp in eval_target_properties), "❌ 'eval_target_properties' contains None values!"
    training_graphs = CrystalGraphDataset(training_crystals, training_target_properties)
    training_dataloader = DataLoader(training_graphs,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=graph_collate,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
    eval_graphs = CrystalGraphDataset(eval_crystals, eval_target_properties)
    eval_dataloader = DataLoader(eval_graphs,
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=graph_collate,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
    best_eval_loss = float('inf')
    best_model_state = None

    # Training loop
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            epoch_loss = 0.0 # Initialize epoch loss

            for graphs, target_values in tqdm(training_dataloader, desc="Training on crystals", unit="batch"):
                print() # newline
                # Move crystals and targets to device
                graphs = [[g.to(device) for g in g_list] for g_list in graphs]
                target_values = target_values.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    # Forward pass
                    results = [model(g_list) for g_list in graphs]
                    batch_property_losses = []
                    batch_contrastive_losses = []

                    for i, (predictions, latents, graph_list) in enumerate(results):
                        # Get the target value for this crystal
                        target = target_values[i]  # scalar

                        crystal_contrastive_losses = []

                        # Use last graph as reference
                        latent_ref = latents[-1]
                        assert graph_list[-1].m_valid == 1, "Reference graph must be valid!"
                        for j in range(len(graph_list) - 1):
                            assert graph_list[j].m_valid == 0, f"Graph {j} should be invalid!"
                            latent_j = latents[j]
                            m_valid = torch.tensor(graph_list[j].m_valid, dtype=torch.float32, device=device)
                            
                            loss_contrastive = MultiTaskLoss.LossContrastive(latent_ref, latent_j, m_valid, threshold)
                            crystal_contrastive_losses.append(loss_contrastive)

                        # Average per-crystal loss (mean over graphs)
                        loss_property = MultiTaskLoss.LossProperty(target.expand_as(predictions[0]), predictions[0])
                        mean_loss_contrastive = torch.stack(crystal_contrastive_losses).sum()

                        batch_property_losses.append(loss_property)
                        batch_contrastive_losses.append(mean_loss_contrastive)

                    # Average over the batch (mean over crystals)
                    loss_property_batch = torch.stack(batch_property_losses).mean()
                    loss_contrastive_batch = torch.stack(batch_contrastive_losses).mean()

                    # Combine losses
                    model_loss = multi_loss(loss_property_batch, loss_contrastive_batch)

                # Backprop and optimize
                model_optimizer.zero_grad()
                if model_loss.requires_grad and model_loss.grad_fn is not None:
                    print("LossProperty:", loss_property_batch.item())
                    print("LossContrastive:", loss_contrastive_batch.item())
                    print("log_vars:", multi_loss.log_vars.detach().cpu().numpy())
                    print("model_loss:", model_loss.item())
                    print()
                    scaler.scale(model_loss).backward()
                    scaler.step(model_optimizer)
                    scaler.update()
                else:
                    print("⚠️ Skipping batch: No gradient flow detected.")

                epoch_loss += model_loss.item()

        
            # Evaluate validation loss
            eval_loss = compute_loss(model, eval_dataloader, multi_loss, device, threshold)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(training_crystals):.4f}, Eval Loss: {eval_loss:.4f}")

            # Save the best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = model.state_dict().copy()

    # Save the best model
    if best_model_state:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        name = f"training/results/{property}_{date}.pth"
        torch.save({
            'threshold': threshold,
            'property': property,
            'model_state_dict': best_model_state,
            "multi_loss_state_dict": multi_loss.state_dict()
        }, name)
        print(f"Best model saved with Val Loss: {best_eval_loss:.4f}")

    print("Training Complete!")