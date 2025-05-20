import os
from models.components.ACG import ACG
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine
from training.ablation_studies.SAM_no_ACG import NoACG
from training.ablation_studies.SAM_no_contrastive import NoContrastive
from training.ablation_studies.SAM_no_group import NoGroupTheory


def study():
    learning_rate = 1e-4


    hull_energy = "e_hull"
    band_gap = "gap pbe"
    magnetic_moment = "mu_b"
    elastic_anisotropy = "elastic anisotropy"
    bulk_modulus = "bulk modulus"
    shear_modulus = "shear modulus"
    formation_energy = "e_form"


    training_data_file = "bulk_megnet_train.pkl"
    validation_data_file = "bulk_megnet_val.pkl"
    num_epochs = 150
    batch_size = 32
    margin = 20.0

    acg = ACG()
    
    hull_energy_loss = SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=hull_energy,
        training_path=os.path.join(DATA_PATH, training_data_file),
        validation_path=os.path.join(DATA_PATH, validation_data_file),
        graph_builder=acg.build_graphs,
        loss_type="multi",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )
    # band_gap_loss = SAM_TrainingEngine(
    #     learning_rate=learning_rate,
    # ).fit(
    #     property=band_gap,
    #     training_path=os.path.join(DATA_PATH, training_data_file),
    #     validation_path=os.path.join(DATA_PATH, validation_data_file),
    #     graph_builder=acg.build_graphs,
    #     loss_type="multi",
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )
    # magnetic_moment_loss = SAM_TrainingEngine(
    #     learning_rate=learning_rate,
    # ).fit(
    #     property=magnetic_moment,
    #     training_path=os.path.join(DATA_PATH, training_data_file),
    #     validation_path=os.path.join(DATA_PATH, validation_data_file),
    #     graph_builder=acg.build_graphs,
    #     loss_type="multi",
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )
    # elastic_anisotropy_loss = SAM_TrainingEngine(
    #     learning_rate=learning_rate,
    # ).fit(
    #     property=elastic_anisotropy,
    #     training_path=os.path.join(DATA_PATH, training_data_file),
    #     validation_path=os.path.join(DATA_PATH, validation_data_file),
    #     graph_builder=acg.build_graphs,
    #     loss_type="multi",
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )
    # bulk_modulus_loss = SAM_TrainingEngine(
    #     learning_rate=learning_rate,
    # ).fit(
    #     property=bulk_modulus,
    #     training_path=os.path.join(DATA_PATH, training_data_file),
    #     validation_path=os.path.join(DATA_PATH, validation_data_file),
    #     graph_builder=acg.build_graphs,
    #     loss_type="multi",
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )
    # shear_modulus_loss = SAM_TrainingEngine(
    #     learning_rate=learning_rate,
    # ).fit(
    #     property=shear_modulus,
    #     training_path=os.path.join(DATA_PATH, training_data_file),
    #     validation_path=os.path.join(DATA_PATH, validation_data_file),
    #     graph_builder=acg.build_graphs,
    #     loss_type="multi",
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )
    formation_energy_loss = SAM_TrainingEngine(
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

    # no_group_theory_best_loss = NoGroupTheory(
    #     crystal_property=crystal_property,
    #     learning_rate=learning_rate,
    #     training_data_file=training_data_file,
    #     validation_data_file=validation_data_file,
    #     num_epochs=num_epochs,
    #     batch_size=batch_size,
    #     margin=margin
    # )

    print()
    print("Training Results:")
    print("===================================")
    print("Hull Energy Loss: ", hull_energy_loss)
    # print("Band Gap Loss: ", band_gap_loss)
    # print("Magnetic Moment Loss: ", magnetic_moment_loss)
    # print("Elastic Anisotropy Loss: ", elastic_anisotropy_loss)
    # print("Bulk Modulus Loss: ", bulk_modulus_loss)
    # print("Shear Modulus Loss: ", shear_modulus_loss)
    print("Formation Energy Loss: ", formation_energy_loss)

    print()
    print("Ablation Study Results:")
    print("===================================")
    print("Original Model: ", formation_energy_loss)
    print("No ACG Model: ", no_acg_best_loss)
    print("No Contrastive Model: ", no_contrastive_best_loss)
    # print("No Group Theory Model: ", no_group_theory_best_loss)