import torch.nn as nn

from models.scripts.update import Update
from models.scripts.global_pooling import GlobalPooling
from models.scripts.predicted_property import PredictedProperty

class Readout(nn.Module):
    """
    Class that pools information about a crystal to predict its properties.

    Attributes:
        update (Update): Instance of the Update class.
        global_pooling (GlobalPooling): Instance of the GlobalPooling class.
        predicted_property (PredictedProperty): Instance of the PredictedProperty class.
    """
    def __init__(self, hidden_dim, output_dim):
        """
        Initilize the readout steps hidden dimensions for linear transformation.

        Args:
            hidden_dim (int): expected dimension of inputs.
            output_dim (int): how many properties to predict (most often 1).
        """
        super().__init__()
        self.update = Update(hidden_dim, hidden_dim)  
        self.global_pooling = GlobalPooling()  
        self.predicted_property = PredictedProperty(hidden_dim, output_dim)  

    def forward(self, aggregated_input, atom_features):
        """
        Pool the attention and features of each of the atoms to determine the predicted properties of the crystal.
        
        Args:
            aggregated_input (torch.tensor): An ordered list of length n with the attention of each of the atoms in the crystal.
            atom_features (torch.tensor): An ordered matrix of size n x 120, where the placement of each row matches the atom in the aggregated_input list.

        Returns:
            prediction: The value of the predicted property of the crystal.
            pooled_features: The pooled features of the crystal.
        """
        updated_features = self.update.forward(aggregated_input, atom_features)
        pooled_features = self.global_pooling.forward(updated_features)
        prediction = self.predicted_property.forward(pooled_features)
        return prediction, pooled_features
    
    def to(self, device):
        """
        Move the Readout instance to the specified device.
        
        Args:
            device (torch.device): The device to move the instance to.

        Returns:
            Readout: instance of the Readout class moved to the specified device. 
        """
        self.update = self.update.to(device)
        self.predicted_property = self.predicted_property.to(device)
        return self