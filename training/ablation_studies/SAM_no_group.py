import os
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from models.scripts.invalid_augmentations import InvalidAugmenter
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine

def NoGroupTheory(crystal_property, learning_rate, training_data_file, validation_data_file, num_epochs, batch_size, margin):
    """
    Trains the SAM model without group theory component.

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
    # Initialize the graph builder without group theory
    graph_builder = NoGroupTheory()
    # TRain the SAM model using the specified parameters
    return SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=graph_builder.build_graphs,
        loss_type="mse",
        evaluation_loss_type="mse",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

class NoGroupTheory:
    """
    Class to build graphs for the SAM model without group theory component.
    """
    def __init__(self):
        self.invalid_aug = InvalidAugmenter()

    def build_graphs(self, C: Crystal):
        valid_syms = C.get_symmetry_operations()
        g_actual = CrystalGraph.init_from_crystal(
            crystal=C,
            m_valid=1,
            config=3.0,
        )

        aug_valid = C.apply_augmentations(valid_syms)
        g_valid = CrystalGraph.create_batch(
            crystals=aug_valid,
            m_valid=1
        )

        inv_syms = self.invalid_aug.forward(valid_syms)
        aug_inv = C.apply_augmentations(inv_syms)
        g_inv = CrystalGraph.create_batch(
            crystals=aug_inv,
            m_valid=0
        )

        return g_inv + g_valid + [g_actual]