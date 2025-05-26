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

    jarvis_props = [
        "formation_energy_peratom",
        "optb88vdw_bandgap",
        "bulk_modulus_kv",
        "shear_modulus_gv",
        "mbj_bandgap",
        "slme",
        "magmom_oszicar",
        "spillage",
        "kpoint_length_unit",
        "encut",
        "optb88vdw_total_energy",
        "epsx",
        "epsy",
        "epsz",
        "mepsx",
        "mepsy",
        "mepsz",
        "max_ir_mode",
        "min_ir_mode",
        "n-Seebeck",
        "p-Seebeck",
        "n-powerfact",
        "p-powerfact",
        "ncond",
        "pcond",
        "nkappa",
        "pkappa",
        "ehull",
        "exfoliation_energy",
        "dfpt_piezo_max_dielectric",
        "dfpt_piezo_max_eij",
        "dfpt_piezo_max_dij",
    ]
    
    hull_energy = "e_hull"
    band_gap = "gap pbe"
    magnetic_moment = "mu_b"
    elastic_anisotropy = "elastic anisotropy"
    bulk_modulus = "bulk modulus"
    shear_modulus = "shear modulus"
    formation_energy = "e_form"

    training_data_file = "jarvis_train"
    validation_data_file = "jarvis_val"
    # training_data_file = os.path.join(DATA_PATH, "bulk_megnet_train.pkl")
    # validation_data_file = os.path.join(DATA_PATH, "bulk_megnet_val.pkl")

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

    
    print()
    print("Training Results:")
    print("===================================")
    print("Hull Energy Loss: ", hull_energy_loss)
    print("Formation Energy Loss: ", formation_energy_loss)

    study()   

# -----------------------------------------------------------------