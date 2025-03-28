import torch
from torch.utils.data import DataLoader
from models.Transformer import Transformer  
from training.loss_functions import LossContrastive, LossFactor, LossProperty, LossTotal

# Hyperparameters
learning_rate = 1e-4
num_epochs = 100
batch_size = 32
margin = 1.0  # Contrastive loss margin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize learnable loss factors
factors = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))  # 2 loss functions

# Initialize model and optimizer
model = Transformer().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + [factors], lr=learning_rate)

# Create dataset and dataloader
dataset = CrystalDataset()  # Load your dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for crystals, targets, flags in dataloader:
        crystals, targets, flags = crystals.to(device), targets.to(device), flags.to(device).float()  # Ensure float32

        # Forward pass
        predictions = model(crystals)

        # Compute losses
        loss_property = LossProperty(predictions, targets)
        loss_contrastive = LossContrastive(predictions, targets, flags, margin)
        both_losses = torch.stack([loss_property, loss_contrastive]) 

        # Compute factors
        factor_property = LossFactor(loss_property, factors[0])
        factor_contrastive = LossFactor(loss_contrastive, factors[1]) 
        both_factors = torch.stack([factor_property, factor_contrastive])  

        # Compute total loss
        loss = LossTotal(both_losses, both_factors)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "transformer_model.pth") # transformer_model.pth = folder path where the model is saved
print("Training Complete!")