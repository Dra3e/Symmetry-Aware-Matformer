import os

from models.components.ACG import ACG
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine
from training.ablation_studies.SAM_no_ACG import NoACG
from training.ablation_studies.SAM_no_contrastive import NoContrastive
from training.ablation_studies.SAM_no_group import NoGroupTheory


def study():
    """
    Conducts an ablation study to evaluate the impact of different components pm the perfomance of the SAM model.
    The study compares the original model with three variation that excule specific components:
    1. No ACG: Excludes the ACG component.
    2. No Contrastive: Excludes the contrastive loss.
    3. No Group Theory: Excludes the group theory component.
    The performance of each model is evaluated using the same dataset and hyperparameters.
    """
    # Hyperparameters for the training process
    learning_rate = 1e-4
    crystal_property = "e_form"
    training_data_file = "bulk_megnet_train.pkl"
    validation_data_file = "bulk_megnet_val.pkl"
    num_epochs = 50
    batch_size = 32
    margin = 20.0

    acg = ACG() # Intitialize the ACG(Automatic Crystal Graph) component
    
    # Train the original model and the three variations
    original_best_loss = SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=crystal_property,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=acg.build_graphs,
        loss_type="multi",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    # Train the model without the ACG component
    no_acg_best_loss = NoACG(
        crystal_property=crystal_property,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    # Train the model without the contrastive loss component
    no_contrastive_best_loss = NoContrastive(
        crystal_property=crystal_property,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    # no_group_theory_best_loss = NoGroupTheory(
    #     crystal_property=crystal_property,
    #     learning_rate=learning_rate,
    #     training_data_file=training_data_file,
    #     validation_data_file=validation_data_file,
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )

    # Print the results of the ablation study
    print()
    print("Ablation Study Results:")
    print("===================================")
    print("Original Model: ", original_best_loss)
    print("No ACG Model: ", no_acg_best_loss)
    print("No Contrastive Model: ", no_contrastive_best_loss)
    print("No Group Theory Model: ", no_group_theory_best_loss)