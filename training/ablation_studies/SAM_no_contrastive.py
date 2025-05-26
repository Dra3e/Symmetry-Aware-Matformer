import os

from models.components.ACG import ACG
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine

def NoContrastive(crystal_property, learning_rate, training_data_file, validation_data_file, num_epochs, batch_size, margin):
    """
    Trains the SAM model without contrastive loss component.

    Args:
        crystal_property (str): The property to be predicted.
        learning_rate (float): Learning rate for the optimizer.
        training_data_file (str): Path to the training data file.
        validation_data_file (str): Path to the validation data file.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        margin (float): Margin for the loss function.
    
    Returns:
        float: The best loss achieved during training.
    """
    acg = ACG() # Initialize the ACG component

    # Train the model using MSE loss insted of contrastive loss
    return SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=acg.build_graphs, # Use ACG to build graphs
        loss_type="mse", # Use MSE loss instead of contrastive loss
        evaluation_loss_type="mse", # Evaluation also uses MSE
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )