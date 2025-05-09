import os
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from models.scripts.invalid_augmentations import InvalidAugmenter
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine


def NoGroupTheory(crystal_property, learning_rate, training_data_file, validation_data_file, num_epochs, batch_size, margin):
    return SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property_mapping=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=build_graphs,
        loss_type="mse",
        evaluation_loss_type="mse",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

def build_graphs(self, C: Crystal):
    cutoff = 3.0
    invalid_aug = InvalidAugmenter()

    valid_syms, valid_trans = C.get_symmetry_operations()
    aug_valid = C.apply_augmentations(valid_syms)
    g_valid = CrystalGraph.create_batch(
        C, aug_valid, valid_trans, m_valid=1, config=cutoff
    )

    inv_syms = invalid_aug.forward(valid_syms)
    aug_inv = C.apply_augmentations(inv_syms)
    g_inv = CrystalGraph.create_batch(aug_inv, valid_trans, m_valid=0)

    return g_inv + g_valid