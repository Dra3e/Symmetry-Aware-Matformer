import os
from models.components.ACG import ACG
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine
from training.ablation_studies.SAM_no_ACG import NoACG
from training.ablation_studies.SAM_no_contrastive import NoContrastive


def study():
    hull_energy = "e_hull"
    band_gap = "gap pbe"
    magnetic_moment = "mu_b"
    elastic_anisotropy = "elastic anisotropy"
    bulk_modulus = "bulk modulus"
    shear_modulus = "shear modulus"
    formation_energy = "e_form"

    learning_rate = 1e-4
    training_data_file = "bulk_megnet_train.pkl"
    validation_data_file = "bulk_megnet_val.pkl"
    num_epochs = 150
    batch_size = 32
    margin = 20.0

    acg = ACG()

    original = SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=formation_energy,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=acg.build_graphs,
        loss_type="multi",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    no_acg_best_loss = NoACG(
        crystal_property=formation_energy,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    no_contrastive_best_loss = NoContrastive(
        crystal_property=formation_energy,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

 

    print()
    print("Ablation Study Results:")
    print("===================================")
    print("Original Model: ", original)
    print("No ACG Model: ", no_acg_best_loss)
    print("No Contrastive Model: ", no_contrastive_best_loss)