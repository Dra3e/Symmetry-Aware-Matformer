import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.mp import CrystalDataset
from models.Transformer import Transformer  
from training.loss_functions import LossContrastive, LossFactor, LossProperty, LossTotal

# Hyperparameters
learning_rate = 1e-4
num_epochs = 100
batch_size = 32
margin = 1.0  # Contrastive loss margin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and optimizer
model = Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize learnable loss factors
factors = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))  # 2 loss functions

# Create dataset and dataloader
dataset = CrystalDataset()  # Load your dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for crystals, targets, flags in dataloader:
        crystals, targets, flags = crystals.to(device), targets.to(device), flags.to(device)

        # Forward pass
        predictions = model(crystals)

        # Compute losses
        loss_property = LossProperty(predictions, targets)
        loss_contrastive = LossContrastive(predictions, targets, flags, margin)
        losses = torch.tensor([loss_property, loss_contrastive], device=device)

        # Compute factors
        factor_property = LossFactor(loss_property, factors[0])
        factor_contrastive = LossFactor(loss_contrastive, factors[1])
        factors = torch.tensor([factor_property, factor_contrastive], device=device)

        # Compute total loss
        loss = LossTotal(losses, factors)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

print("Training Complete!")