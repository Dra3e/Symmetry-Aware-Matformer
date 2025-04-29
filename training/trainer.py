import os
import torch
import datetime
from torch.utils.data import DataLoader
from models.Transformer import Transformer  
from training.loss_functions import LossContrastive, LossFactor, LossProperty, LossTotal
from training.dataset import get_training_data_from_pkl
from training.evaluator import compute_loss
from tqdm import tqdm 

def custom_collate_fn(batch):
    crystals, target_values = zip(*batch)
    return list(crystals), torch.tensor(target_values, dtype=torch.float32)

def run(property, training_path: str, validation_path: str, num_epochs = 100, batch_size = 25, learning_rate = 1e-4, threshold = 100.0):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    num_workers = max(2, os.cpu_count() // 2) if torch.cuda.is_available() else 0

    # Initialize learnable loss factors
    factors = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))  # 2 loss functions

    # Initialize model and optimizer
    model = Transformer().to(device) # Assuming Transformer is defined in models.Transformer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer for the model
    factor_optimizer = torch.optim.Adam([factors], lr=1e-5) # Optimizer for the loss factors

    # Create dataset
    dataset, target_properties = get_training_data_from_pkl(property, training_path) # Assuming get_training_data() returns a dataset and property value lit compatible with the model
    assert all(c is not None for c in dataset), "❌ 'dataset' contains None values!"
    assert all(tp is not None for tp in target_properties), "❌ 'dataset' contains None values!"
    eval_dataset, eval_target_properties = get_training_data_from_pkl(property, validation_path)
    assert all(c is not None for c in eval_dataset), "❌ 'eval_dataset' contains None values!"
    data_loader = DataLoader(
        list(zip(dataset, target_properties)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    eval_loader = DataLoader(
        list(zip(eval_dataset, eval_target_properties)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    best_eval_loss = float('inf')
    best_model_state = None

    # Training loop
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            epoch_loss = 0.0 # Initialize epoch loss

            for crystals, target_values in tqdm(data_loader, desc="Training on crystals", unit="crystal batches"):
                # Move crystals and targets to device
                crystals = [c.to(device) for c in crystals]
                target_values = target_values.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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
                    model_loss = LossTotal(both_losses, factors)

                # Backprop and optimize
                model_optimizer.zero_grad()
                if model_loss.requires_grad and model_loss.grad_fn is not None:
                    scaler.scale(model_loss).backward()
                    scaler.step(model_optimizer)
                    scaler.update()
                else:
                    print("⚠️ Skipping batch: No gradient flow detected.")

                # Update factors
                factor_optimizer.zero_grad()
                loss_factor_property = LossFactor(loss_property_batch.detach(), factors[0])
                loss_factor_contrastive = LossFactor(loss_contrastive_batch.detach(), factors[1])
                loss_factor_property.backward(retain_graph=True)
                loss_factor_contrastive.backward()
                factor_optimizer.step()

                epoch_loss += model_loss.item()
        
            # Evaluate validation loss
            eval_loss = compute_loss(model, eval_loader, factors, device, threshold)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(dataset):.4f}, Eval Loss: {eval_loss:.4f}")

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
            'factors': factors.detach().cpu()
        }, name)
        print(f"Best model saved with Val Loss: {best_eval_loss:.4f}")

    print("Training Complete!")