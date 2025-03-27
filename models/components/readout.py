import torch.nn as nn
from models.scripts.update import Update
from models.scripts.global_pooling import GlobalPooling
from models.scripts.predicted_property import PredictedProperty

class Readout(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.update = Update(hidden_dim, hidden_dim)  
        self.global_pooling = GlobalPooling()  
        self.predicted_property = PredictedProperty(hidden_dim, output_dim)  

    def forward(self, aggregated_input, atom_features):
        updated_features = self.update.forward(aggregated_input, atom_features)
        pooled_features = self.global_pooling.forward(updated_features)
        predicted = self.predicted_property.forward(pooled_features)
        return predicted
    
    def batch_forward(self, aggregated_input, atom_features):
        """
        Perform the forward pass for batched inputs.

        Parameters:
        - aggregated_input: Aggregated features (batch_size, 1)
        - atom_features: Node features (batch_size, num_nodes, feature_dim)
        - hidden_dim: Hidden feature dimension

        Returns:
        - predicted: Predicted properties for each graph in the batch (batch_size, hidden_dim)
        """
        # Reshape m_i to match the expected shape of aggregated_input
        expanded_aggregated_input = aggregated_input.unsqueeze(1).expand(-1, atom_features.shape[1], self.hidden_dim)  # Shape: (batch_size, num_nodes, hidden_dim)

        updated_features = self.update.batch_forward(expanded_aggregated_input, atom_features) # Shape: (batch_size, num_nodes, hidden_dim)
        pooled_features = self.global_pooling.batch_forward(updated_features) # Shape: (batch_size, hidden_dim)
        predicted = self.predicted_property.batch_forward(pooled_features) # Shape: (batch_size, hidden_dim)
        return predicted