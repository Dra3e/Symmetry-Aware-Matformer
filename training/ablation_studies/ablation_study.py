import os
from models.components import ACG
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine
from training.ablation_studies.SAM_no_ACG import NoACG
from training.ablation_studies.SAM_no_contrastive import NoContrastive
from training.ablation_studies.SAM_no_group import NoGroupTheory


def study():
    learning_rate = 1e-2
    crystal_property = "e_form"
    training_data_file = "bulk_megnet_train.pkl"
    validation_data_file = "bulk_megnet_val.pkl"
    num_epochs = 50
    batch_size = 32
    margin = 20.0

    acg = ACG()
    
    original = SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property_mapping=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=acg.build_graphs,
        loss_type="multi",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    no_acg = NoACG(
        crystal_property=crystal_property,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    no_contrastive = NoContrastive(
        crystal_property=crystal_property,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    no_group_theory = NoGroupTheory(
        crystal_property=crystal_property,
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
    print("No ACG Model: ", no_acg)
    print("No Contrastive Model: ", no_contrastive)
    print("No Group Theory Model: ", no_group_theory)