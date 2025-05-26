# run.py  – top‑level script
import os
from models.components.ACG import ACG
from setup import DATA_PATH
from training.engine import SAM_TrainingEngine
from training.ablation_studies.ablation_study import study   # just the import

# -----------------------------------------------------------------
if __name__ == "__main__":             # ← the “entry‑point guard”
    import multiprocessing as mp       # Windows needs this extra line
    mp.freeze_support()                # safe even when you’re *not* freezing an exe
                             # everything that kicks off training
    
    hull_energy = "e_hull"
    band_gap = "gap pbe"
    magnetic_moment = "mu_b"
    elastic_anisotropy = "elastic anisotropy"
    bulk_modulus = "bulk modulus"
    shear_modulus = "shear modulus"
    formation_energy = "e_form"
    training_data_file = os.path.join(DATA_PATH, "bulk_megnet_train.pkl")
    validation_data_file = os.path.join(DATA_PATH, "bulk_megnet_val.pkl")

    learning_rate = 1e-4
    num_epochs = 5
    batch_size = 64
    margin = 20.0

    acg = ACG()

    formation_energy_loss = SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property="formation_energy_peratom",
        training_path=training_data_file,
        validation_path=validation_data_file,
        graph_builder=acg.build_graphs,
        loss_type="multi",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )
    hull_energy_loss = SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property="ehull",
        training_path=training_data_file,
        validation_path=validation_data_file,
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
    
    print()
    print("Training Results:")
    print("===================================")
    print("Hull Energy Loss: ", hull_energy_loss)
    print("Formation Energy Loss: ", formation_energy_loss)
    # print("Band Gap Loss: ", band_gap_loss)
    # print("Magnetic Moment Loss: ", magnetic_moment_loss)
    # print("Elastic Anisotropy Loss: ", elastic_anisotropy_loss)
    # print("Bulk Modulus Loss: ", bulk_modulus_loss)
    # print("Shear Modulus Loss: ", shear_modulus_loss)

    study()   

# -----------------------------------------------------------------