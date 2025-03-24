import torch.nn as nn
from models.scripts.update import Update
from models.scripts.global_pooling import GlobalPooling
from models.scripts.predicted_property import PredictedProperty

class Readout(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.update = Update(input_dim, hidden_dim)  
        self.global_pooling = GlobalPooling()  
        self.predicted_property = PredictedProperty(input_dim, hidden_dim)  

    def forward(self, aggregated_input, atom_features):
        updated_features = self.update.forward(aggregated_input, atom_features)
        pooled_features = self.global_pooling.forward(updated_features)
        predicted = self.predicted_property.forward(pooled_features)
        return predicted