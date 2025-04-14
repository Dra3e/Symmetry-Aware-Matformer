import torch
import datetime
from torch.utils.data import DataLoader
from models.Transformer import Transformer  
from training.loss_functions import LossContrastive, LossFactor, LossProperty, LossTotal
from training.dataset import get_training_data
from training.evaluator import compute_loss
from tqdm import tqdm 

def custom_collate_fn(batch):
    crystals, target_values = zip(*batch)
    return list(crystals), torch.tensor(target_values, dtype=torch.float32)

def run(property, training_path: str, validation_path: str, num_epochs = 100, batch_size = 25, learning_rate = 1e-4, threshold = 100.0):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize learnable loss factors
    factors = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))  # 2 loss functions

    # Initialize model and optimizer
    model = Transformer().to(device) # Assuming Transformer is defined in models.Transformer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer for the model
    factor_optimizer = torch.optim.Adam([factors], lr=1e-5) # Optimizer for the loss factors

    # Create dataset
    dataset, target_properties = get_training_data(property, training_path) # Assuming get_training_data() returns a dataset and property value lit compatible with the model
    eval_dataset, eval_target_properties = get_training_data(property, validation_path)
    data_loader = DataLoader(list(zip(dataset, target_properties)), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(list(zip(eval_dataset, eval_target_properties)), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    best_eval_loss = float('inf')
    best_model_state = None

    # Training loop
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            epoch_loss = 0.0 # Initialize epoch loss

            for crystals, target_values in tqdm(data_loader, desc="Training on crystals", unit="crystal"):
                # Move targets to device
                target_values = target_values.to(device)

                # Forward pass
                results = [model(c) for c in crystals]
                predictions, latentspace_f, graphs_batch = zip(*results)
                predictions = torch.stack(predictions)         # [B, N]
                latentspace_f = torch.stack(latentspace_f)     # [B, N, D]
                graphs_batch = list(graphs_batch)              # keep as list if needed individually


                # Initialize accumulators
                all_loss_property = []
                all_loss_contrastive = []

                # We'll stack all per-graph losses into one big batch operation
                for i in range(1, latentspace_f.shape[1]):  # Skip index 0 (reference)
                    latent0 = latentspace_f[:, 0, :]                  # [B, D]
                    latenti = latentspace_f[:, i, :]                  # [B, D]
                    pred_i = predictions[:, i].view(-1)              # [B]

                    m_valid = torch.tensor([g[i].m_valid for g in graphs_batch], dtype=torch.float32, device=device)

                    loss_property = LossProperty(target_values, pred_i)               # [B]
                    loss_contrastive = LossContrastive(latent0, latenti, m_valid, threshold)  # [B]

                    all_loss_property.append(loss_property)
                    all_loss_contrastive.append(loss_contrastive)

                # Stack and mean over all comparisons
                loss_property_batch = torch.stack(all_loss_property).mean()
                loss_contrastive_batch = torch.stack(all_loss_contrastive).mean()

                # Combine losses
                both_losses = torch.tensor([loss_property_batch, loss_contrastive_batch], device=device)
                model_loss = LossTotal(both_losses, factors)

                # Backprop and optimize
                model_optimizer.zero_grad()
                model_loss.backward()
                model_optimizer.step()

                # Update factors
                factor_optimizer.zero_grad()
                LossFactor(loss_property_batch.detach(), factors[0]).backward(retain_graph=True)
                LossFactor(loss_contrastive_batch.detach(), factors[1]).backward()
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