import os
from models.components.ACG import ACG
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine


def NoContrastive(crystal_property, learning_rate, training_data_file, validation_data_file, num_epochs, batch_size, margin):
    acg = ACG()
    return SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=acg.build_graphs,
        loss_type="mse",
        evaluation_loss_type="mse",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )