import os

from models.components.ACG import ACG
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine
from training.ablation_studies.SAM_no_ACG import NoACG
from training.ablation_studies.SAM_no_contrastive import NoContrastive


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
    hull_energy = "e_hull"
    band_gap = "gap pbe"
    magnetic_moment = "mu_b"
    elastic_anisotropy = "elastic anisotropy"
    bulk_modulus = "bulk modulus"
    shear_modulus = "shear modulus"
    formation_energy = "e_form"

    learning_rate = 1e-4
    training_data_file = "bulk_megnet_train.pkl" # Path to the traning dataset
    validation_data_file = "bulk_megnet_val.pkl" # Path to the validation dataset
    num_epochs = 150 # Number of epochs for training
    batch_size = 32 # Batch size for training
    margin = 20.0 # Margin for the contrastive loss

    acg = ACG() # Intitialize the ACG(Automatic Crystal Graph) component
    
    # Train the original model and the three variations
    original_best_loss = SAM_TrainingEngine(
        learning_rate=learning_rate, # Learning rate for the optimizer
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

    # Train the model without the ACG component
    no_acg_best_loss = NoACG(
        crystal_property=formation_energy,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    # Train the model without the contrastive loss component
    no_contrastive_best_loss = NoContrastive(
        crystal_property=formation_energy,
        learning_rate=learning_rate,
        training_data_file=training_data_file,
        validation_data_file=validation_data_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

    # Print the results of the ablation study
    print()
    print("Ablation Study Results:")
    print("===================================")
    print("Original Model: ", original_best_loss)
    print("No ACG Model: ", no_acg_best_loss)
    print("No Contrastive Model: ", no_contrastive_best_loss)