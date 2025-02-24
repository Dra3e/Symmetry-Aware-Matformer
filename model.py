import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pymatgen.ext.matproj import MPRester


# API Key & Material ID
api_key = "KzsI815OLqRu9u7MZwj9vuizzTCcUUon"
material_id = "mp-5409"


# 1. Function to fetch crystal structure data from Materials Project API
def fetch_material_data(material_id, api_key):
    mpr = MPRester(api_key)  # Initialize API
    # Use the new method to fetch structure data
    material_data = mpr.get_structure_by_material_id(material_id)

    if material_data is None:
        raise ValueError("Failed to fetch material data. Check API key or material ID.")
    # Extract atomic features and positions
    A = np.array([[site.specie.Z] for site in material_data.sites])  # Atomic number as feature
    X = np.array([site.coords for site in material_data.sites])  # 3D positions
    L = material_data.lattice.matrix  # Lattice (3x3)

    return A, X, L

# 2. Define Augmentation Functions


# 3. Function to convert atomic structure into 3D voxel grid
def create_voxel_grid_with_features(A, X, L, grid_size=32):
    n, f = A.shape  # Number of atoms, features per atom

    # Normalize atomic positions to fit within a grid
    min_coords = np.min(X, axis=0)
    max_coords = np.max(X, axis=0)
    X_norm = (X - min_coords) / (max_coords - min_coords + 1e-6)  #  # Avoid division by zero
    
    # Scale positions to the voxel grid
    X_voxel = (X_norm * (grid_size - 1)).astype(int) 

    # Initialize a 3D voxel grid with zeros
    voxel_grid = np.zeros((f, grid_size, grid_size, grid_size))
    for i, (x, y, z) in enumerate(X_voxel):
        voxel_grid[:, x, y, z] = A[i]   # Assign atomic features to grid positions

    return torch.tensor(voxel_grid, dtype=torch.float) # Convert to tensor


# 4. 3D CNN Model
class Crystal3DCNN(nn.Module):
    def __init__(self, num_features):
        super(Crystal3DCNN, self).__init__()
        # Define 3D convolutional layers
        self.conv1 = nn.Conv3d(num_features, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)  # Adjust dimensions accordingly
        self.fc2 = nn.Linear(128, 1)  # Predict single value (e.g., energy)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First convolution + ReLU activation
        x = F.max_pool3d(x, 2)  # Max pooling to reduce dimensions
        x = F.relu(self.conv2(x))  # Second convolution + ReLU activation
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3(x))  # Third convolution + ReLU activation
        x = F.max_pool3d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        return self.fc2(x)  # Final prediction


# 5. Fetch material data
A, X, L = fetch_material_data(material_id, api_key)

# Convert atomic structure to a 3D voxel representation
grid_size = 32
original_voxel_data = create_voxel_grid_with_features(A, X, L, grid_size)

# 6. Perform Augmentation

# 7. Train the model and Initialize 3D CNN model
model = Crystal3DCNN(num_features=A.shape[1])  # Initialize model with input feature size

# 8. Define Loss and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# 9. Dummy target (e.g., energy)
target = torch.tensor([[1.0]])  # Ensure shape matches model output

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()  # Reset gradients
    output = model(original_voxel_data.unsqueeze(0))  # Add batch dimension
    loss = criterion(output, target)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
