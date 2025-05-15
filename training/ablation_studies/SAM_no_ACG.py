import os
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine


def NoACG(crystal_property, learning_rate, training_data_file, validation_data_file, num_epochs, batch_size, margin):
    return SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=build_graphs,
        loss_type="mse",
        evaluation_loss_type="mse",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

def build_graphs(C: Crystal):
    g_valid = CrystalGraph.init_from_crystal(
        crystal=C,
        m_valid=1,
        config=3.0,
    )
    return [g_valid]