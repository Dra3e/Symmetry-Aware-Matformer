import os
import torch

from torch.utils.data import DataLoader
from models.components.ACG import ACG
from training.engine import SAM_TrainingEngine
from training.data_getter import DataGetter
from models.Transformer import Transformer

# Paths to data and weights directories
DATA_PATH = "data"
WEIGHTS_PATH = "training/results"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging CUDA errors

def main():
    """
    Main entry point for the program.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = os.path.dirname(os.path.abspath(__file__))  # Path to the base directory

    # Check if the data and results folders exist, if not create them
    data_folder = os.path.join(base_path, "data")
    results_folder = os.path.join(base_path, "training", "results")

    for folder in [data_folder, results_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created missing folder: {folder}")

    # Check again if the data folder is empty (in case it was created just now)
    if not os.listdir(data_folder):
        print("The 'data' folder is empty. Please add some data files before running the program.")
        return

    # Get available data and weight files
    data_files = get_files_from_directory(DATA_PATH)
    weight_files = get_files_from_directory(WEIGHTS_PATH)

    if not data_files:
        print("No files found in the data directory. Please add some data files.")
        return

    while True:
        print(
            "\nHello World!\n"
            "Welcome to SAM (Symmetry Aware Matformer)\n"
            "A transformer model for predicting the properties of crystals\n"
        )
        print("Available options:")
        print("[0] Terminate program")
        print("[1] Train a new model")
        print("[2] Run an existing model")
        
        try:
            choice = int(input("Please enter your choice > "))
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 2.")
            continue

        if choice == 0:
            print("Terminating program. Goodbye!")
            break
        elif choice == 1:
            setup_training(data_files=data_files)
            weight_files = get_files_from_directory(WEIGHTS_PATH)
        elif choice == 2:
            setup_run(device=device, data_files=data_files, weight_files=weight_files)
        else:
            print("Invalid choice. Please enter a number between 0 and 2.")

def setup_training(data_files):
    """
    Sets up the training process.
    
    Args:
        data_files (list[str]): List of available dataset filenames.
    """
    data_file_list = list_files(data_files)
    while(1): # Prompt user to select training dataset
        print("Please choose the training dataset:")
        print(data_file_list)
        test_choice = int(input("> "))
        if(test_choice >= 0 and test_choice < len(data_files)):
            break
    
    while(1): # Prompt user to select validation dataset
        print("Please choose the validation dataset:")
        print(data_file_list)
        validation_choice = int(input("> "))
        if(validation_choice >= 0 and validation_choice < len(data_files)):
            break
    
    property_mapping = { # Available properties mapped to their internal keys
        "hull energy": "e_hull",
        "band gap": "gap pbe",
        "magnetic moment": "mu_b",
        "elastic anisotropy": "elastic anisotropy",
        "bulk modulus": "bulk modulus",
        "shear modulus": "shear modulus",
        "formation energy": "e_form"
    }
    property_names = list(property_mapping.keys()) 
    property_list = list_files(property_names)
    while True: # Prompt user to select which property to predict
        print("Please choose a property to predict:")
        print(property_list)
        chosen_property = int(input("> "))
        if 0 <= chosen_property < len(property_list):
            break
        else:
            print("Invalid choice. Please select a valid index.")


    # Default training parameters
    DEFAULT_NUM_EPOCHS = 150
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 3e-4
    DEFAULT_MARGIN = 20.0

    num_epochs = DEFAULT_NUM_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LEARNING_RATE
    margin = DEFAULT_MARGIN

    print(
        f"Default settings:\n"
        f"Number of epochs: {num_epochs}\n"
        f"Batch size: {batch_size}\n"
        f"Learning rate: {learning_rate}\n"
        f"Margin for separating valid and invalid crystals: {margin}\n"
    )
    default = input("Run using default settings?\n (Y/N): ").strip().upper()
    
    if(default == "N"):
        try:
            num_epochs = int(input(f"Number of epochs (Default: {DEFAULT_NUM_EPOCHS}): > "))
            batch_size = int(input(f"Batch size (Default: {DEFAULT_BATCH_SIZE}): > "))
            learning_rate = float(input(f"Learning rate (Default: {DEFAULT_LEARNING_RATE}): > "))
            margin = float(input(f"Margin (Default: {DEFAULT_MARGIN}): > "))
        except ValueError:
            print("Invalid input. Using default settings.")

    print("Training...")
    acg = ACG()
    SAM_TrainingEngine(
        learning_rate=learning_rate,
    ).fit(
        property=property_mapping[property_names[chosen_property]],
        training_path=os.path.join(DATA_PATH, data_files[test_choice]),
        validation_path=os.path.join(DATA_PATH, data_files[validation_choice]),
        graph_builder=acg.build_graphs,
        loss_type="multi",
        num_epochs=num_epochs,
        batch_size=batch_size,
        margin=margin
    )

def setup_run(device, data_files, weight_files):
    """
    Sets up and runs an existing model.
    
    Args:
        device (torch.device): The device to load the model on (CPU or GPU).
        data_files (list): List of available datasets.
        weight_files (list): List of available model weight files.
    """
    weight_file_list = list_files(weight_files)
    data_file_list = list_files(data_files)

    while True: # Prompt user to select saved model
        print("Please choose the constructed model to use:")
        print(weight_file_list)
        model_choice = int(input("> "))
        if 0 <= model_choice < len(weight_files):
            break
        else:
            print("Invalid choice. Please select a valid index.")

    while True: # Prompt user to select validation dataset
        print("Please choose the validation dataset:")
        print(data_file_list)
        data_choice = int(input("> "))
        if 0 <= data_choice < len(data_files):
            break
        else:
            print("Invalid choice. Please select a valid index.")
    
    checkpoint = torch.load(os.path.join(WEIGHTS_PATH, weight_files[model_choice]), map_location=device)
    acg = ACG()
    engine = SAM_TrainingEngine()
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.multi_loss.load_state_dict(checkpoint['multi_loss_state_dict'])

    eval_dataloader = DataGetter(
            path           = os.path.join(DATA_PATH, data_files[data_choice]),
            property_name  = checkpoint['property'],
            graph_builder  = acg.build_graphs,
            batch_size     = 32,
            shuffle        = False
    ).loader

    print("Computing loss...")
    eval_loss = engine.run(dataloader=eval_dataloader, loss_type="mse", margin=checkpoint['margin'])
    print(f"The MSA = {eval_loss}")
    input("Press Enter to continue > ")        

def list_files(files):
    """
    Formats a list of filenames into a numbered list string.

    Args:
        files (list[str]): List of file names.

    Returns:
        str: Formatted string with indexed file names.
    """
    file_list = []
    for i, filename in enumerate(files):
        file_list.append(f"[{i}] {filename}")
    return "\n".join(file_list)

def get_files_from_directory(directory):
    """
    Retrieves all file names from a given directory (non-recursively).

    Args:
        directory (str): Path to the target directory.

    Returns:
        list[str]: List of filenames in the directory.
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == "__main__":
    main()