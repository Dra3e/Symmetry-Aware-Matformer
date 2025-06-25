import os

from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine


def NoACG(crystal_property, learning_rate, training_data_file, validation_data_file, num_epochs, batch_size, margin):
    """
    Trains the SAM model without thte ACG component.

    Args:
        crystal_property (str): The property to be predicted.
        learning_rate (float): The learning rate for the optimizer.
        training_data_file (str): The path to the training data file.
        validation_data_file (str): The path to the validation data file.
        num_epochs (int): The number of epochs to train for.
        batch_size (int): The batch size for training.
        margin (float): The margin for the loss function.

    Returns:
        float: The best loss achieved during training.
    """
    return SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=build_graphs, # Uses a simplified graph builder
        loss_type="mse", # Mean Squared Error
        evaluation_loss_type="mse", # Evaluation also use MSE
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

def build_graphs(C: Crystal):
    """
    Build a graph from the crystal structure without using the ACG component.

    Args:
        C (Crystal): The crystal structure.

    Returns:
        g_valid (list): A list of graphs.
    """
    # Generate a valid crystal graph using the input crystal
    g_valid = CrystalGraph.init_from_crystal(
        crystal=C,
        m_valid=1, # Indicated a valid crystal graph
        cutoff=3.0, # Default cutoff configuration
    )
    return [g_valid]