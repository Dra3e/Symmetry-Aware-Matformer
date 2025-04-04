import torch
from torch.utils.data import DataLoader
from models.Transformer import Transformer  
from training.loss_functions import LossContrastive, LossFactor, LossProperty, LossTotal
from training.dataset import get_training_data

def run(property, num_epochs, json_path: str):
    # Hyperparameters
    learning_rate = 1e-4 # Initial learning rate for the model
    num_epochs = 100 # Number of epochs for training
    batch_size = 32 # Batch size for training
    margin = 1.0  # Contrastive loss margin
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

    # Initialize learnable loss factors
    factors = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))  # 2 loss functions

    # Initialize model and optimizer
    model = Transformer().to(device) # Assuming Transformer is defined in models.Transformer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer for the model
    factor_optimizer = torch.optim.Adam([factors], lr=1e-5) # Optimizer for the loss factors

    # Create dataset
    dataset, target_property = get_training_data(property, json_path) # Assuming get_training_data() returns a dataset compatible with the model

    # Training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0 # Initialize epoch loss
        
        for crystal in dataset:  # Iterate over individual crystals since model only supports single input
            predictions, graphs = model(crystal) # Get prediction and flag from model
            for i in range(len(graphs)):
                model_optimizer.zero_grad() # Zero gradients for model optimizer
                factor_optimizer.zero_grad() # Zero gradients for factor optimizer
                
                # Forward pass through the model
                target = target_property # Get target property from crystal
                
                # Compute loss values
                loss_property = LossProperty(predictions[i], target) 
                loss_contrastive = LossContrastive(predictions[i], target, graphs[i].m_valid, margin)
                both_losses = torch.stack([loss_property, loss_contrastive])
                model_loss = LossTotal(both_losses, factors)
                
                # Backpropagation and optimization for model
                model_loss.backward()
                model_optimizer.step()
                
                # Compute and update loss factors separately
                loss_factor_property = LossFactor(loss_property, factors[0])
                loss_factor_contrastive = LossFactor(loss_contrastive, factors[1])
                
                # Backpropagate each loss separately to update only respective factors
                loss_factor_property.backward(retain_graph=True)  # Updates only factors[0]
                loss_factor_contrastive.backward()  # Updates only factors[1]
                
                factor_optimizer.step() # Update loss factors
                
                epoch_loss += model_loss.item() # Accumulate epoch loss
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataset):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Training Complete!")