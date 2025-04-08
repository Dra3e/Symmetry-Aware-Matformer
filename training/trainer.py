import torch
# from torch.utils.data import DataLoader
from models.Transformer import Transformer  
from training.loss_functions import LossContrastive, LossFactor, LossProperty, LossTotal
from training.dataset import get_training_data

def compute_val_loss(model, val_dataset, target_property, factors, device, margin):
    # Initialize model
    model.eval()
    total_loss = 0.0
    sample_count = 0

    # Runs the model on each crystal in the dataset
    with torch.no_grad(): # Tells torch that these weights should not be remembered for training
        for crystal in val_dataset:
            predictions, latenspace_f, graphs = model(crystal)
            for i in range(len(graphs)):
                target = torch.tensor(target_property, dtype=torch.float32, device=device)

                loss_property = LossProperty(target, predictions[i].squeeze())
                loss_contrastive = LossContrastive(latenspace_f[0], latenspace_f[i], graphs[i].m_valid, margin)
                both_losses = torch.stack([loss_property, loss_contrastive])
                total = LossTotal(both_losses, factors)

                total_loss += total.item()
                sample_count += 1

    return total_loss / sample_count if sample_count > 0 else float('inf')

def run(property, training_path: str, validation_path: str, num_epochs = 100, learning_rate = 1e-4, margin = 1.0):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize learnable loss factors
    factors = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))  # 2 loss functions

    # Initialize model and optimizer
    model = Transformer().to(device) # Assuming Transformer is defined in models.Transformer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer for the model
    factor_optimizer = torch.optim.Adam([factors], lr=1e-5) # Optimizer for the loss factors

    # Create dataset
    dataset, target_property = get_training_data(property, training_path) # Assuming get_training_data() returns a dataset compatible with the model
    val_dataset, val_target_property = get_training_data(property, validation_path)

    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            epoch_loss = 0.0 # Initialize epoch loss
            
            for crystal in dataset:  # Iterate over individual crystals since model only supports single input
                predictions, latenspace_f, graphs = model(crystal) # Get prediction and flag from model
                for i in range(0, len(graphs)-1):
                    model_optimizer.zero_grad() # Zero gradients for model optimizer
                    factor_optimizer.zero_grad() # Zero gradients for factor optimizer
                    
                    # Forward pass through the model
                    target = torch.tensor(target_property, dtype=torch.float32, device=device)

                    # Compute loss values
                    latent0 = latenspace_f[0].clone() # .backward() complains when a parameter is reused without cloning
                    latenti = latenspace_f[i].clone()

                    loss_property = LossProperty(target, predictions[i].squeeze()) 
                    loss_contrastive = LossContrastive(latent0, latenti, graphs[i].m_valid, margin)
                    both_losses = torch.stack([loss_property, loss_contrastive])
                    model_loss = LossTotal(both_losses, factors)
                    model_loss = loss_property
                    
                    # Backpropagation and optimization for model
                    model_loss.backward()
                    model_optimizer.step()
                    
                    # Compute and update loss factors separately
                    loss_factor_property = LossFactor(loss_property.detach(), factors[0])
                    loss_factor_contrastive = LossFactor(loss_contrastive.detach(), factors[1])
                    
                    # Backpropagate each loss separately to update only respective factors
                    loss_factor_property.backward(retain_graph=True)  # Updates only factors[0]
                    loss_factor_contrastive.backward()  # Updates only factors[1]
                    
                    # factor_optimizer.step() # Update loss factors
                    
                    epoch_loss += model_loss.item() # Accumulate epoch loss
            
            # Evaluate validation loss
            val_loss = compute_val_loss(model, val_dataset, val_target_property, factors, device, margin)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(dataset):.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

    # Save the best model
    if best_model_state:
        torch.save(best_model_state, "transformer_model_best.pth")
        print(f"Best model saved with Val Loss: {best_val_loss:.4f}")

    print("Training Complete!")