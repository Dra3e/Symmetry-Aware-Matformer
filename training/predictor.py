import torch

from models.Transformer import Transformer  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer()  # Inference
model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
model.to(device)  # Move to GPU if needed
model.eval()

new_crystal = ...  # Prepare new input (same format as training data)
new_crystal = new_crystal.to(device)  # Move to GPU if needed

with torch.no_grad():  # No need to track gradients
    prediction = model(new_crystal)

print("Predicted Property:", prediction.item())