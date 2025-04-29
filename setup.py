import os
import torch
from torch.utils.data import DataLoader
from training.trainer import run, graph_collate
from training.evaluator import compute_loss
from models.Transformer import Transformer
from training.dataset import get_training_data_from_pkl, CrystalGraphDataset

DATA_PATH = "data"
WEIGHTS_PATH = "training/results"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    """Main entry point for the program."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_path = os.path.dirname(os.path.abspath(__file__))  # Path to the base directory

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

    run_program = True
    while(run_program):
        print(
            "Hello World!\n"
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
            run_program = False
        elif choice == 1:
            setup_training(data_files=data_files)
            weight_files = get_files_from_directory(WEIGHTS_PATH)
        elif choice == 2:
            setup_run(device=device, data_files=data_files, weight_files=weight_files)
        else:
            print("Invalid choice. Please enter a number between 0 and 2.")

def setup_training(data_files):
    """Sets up the training process."""
    data_file_list = list_files(data_files)
    while(1):
        print("Please choose the training dataset:")
        print(data_file_list)
        test_choice = int(input("> "))
        if(test_choice >= 0 and test_choice < len(data_files)):
            break
    
    while(1):
        print("Please choose the validation dataset:")
        print(data_file_list)
        validation_choice = int(input("> "))
        if(validation_choice >= 0 and validation_choice < len(data_files)):
            break
    
    property_mapping = {
        "formation energy": "e_hull",
        "band gap": "gap pbe",
        "magnetic moment": "mu_b",
        "elastic anisotropy": "elastic anisotropy",
        "bulk modulus": "bulk modulus",
        "shear modulus": "shear modulus"
    }
    property_names = list(property_mapping.keys())
    property_list = list_files(property_names)
    while True:
        print("Please choose a property to predict:")
        print(property_list)
        chosen_property = int(input("> "))
        if 0 <= chosen_property < len(property_list):
            break
        else:
            print("Invalid choice. Please select a valid index.")


    # Default training parameters
    DEFAULT_NUM_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 1e-2
    DEFAULT_THRESHOLD = 100.0

    num_epochs = DEFAULT_NUM_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LEARNING_RATE
    threshold = DEFAULT_THRESHOLD

    print(
        f"Default settings:\n"
        f"Number of epochs: {num_epochs}\n"
        f"Batch size: {batch_size}\n"
        f"Learning rate: {learning_rate}\n"
        f"Threshold for separating valid and invalid crystals: {threshold}\n"
    )
    default = input("Run using default settings?\n (Y/N): ").strip().upper()
    
    if(default == "N"):
        try:
            num_epochs = int(input(f"Number of epochs (Default: {DEFAULT_NUM_EPOCHS}):"))
            batch_size = int(input(f"Batch size (Default: {DEFAULT_BATCH_SIZE});"))
            learning_rate = float(input(f"Learning rate (Default: {DEFAULT_LEARNING_RATE}):"))
            threshold = float(input(f"Threshold (Default: {DEFAULT_THRESHOLD}):"))
        except ValueError:
            print("Invalid input. Using default settings.")

    print("Training...")
    run(
        property_mapping[property_names[chosen_property]],
        training_path=os.path.join(DATA_PATH, data_files[test_choice]),
        validation_path=os.path.join(DATA_PATH, data_files[validation_choice]),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold=threshold)


def setup_run(device, data_files, weight_files):
    """Sets up and runs an existing model.
    Args:
        device (torch.device): The device to load the model on (CPU or GPU).
        data_files (list): List of available datasets.
        weight_files (list): List of available model weight files."""
    weight_file_list = list_files(weight_files)
    data_file_list = list_files(data_files)

    while True:
        print("Please choose the constructed model to use:")
        print(weight_file_list)
        model_choice = int(input("> "))
        if 0 <= model_choice < len(weight_files):
            break
        else:
            print("Invalid choice. Please select a valid index.")

    while True:
        print("Please choose the validation dataset:")
        print(data_file_list)
        data_choice = int(input("> "))
        if 0 <= data_choice < len(data_files):
            break
        else:
            print("Invalid choice. Please select a valid index.")
    
    checkpoint = torch.load(os.path.join(WEIGHTS_PATH, weight_files[model_choice]), map_location=device)
    model = Transformer().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model

    # Re-wrap the saved tensor as a Parameter
    factors = checkpoint['factors'].to(device)
    
    target_property = checkpoint['property']
    threshold = checkpoint['threshold']
    print("Loading model...")
    dataset, target_properties = get_training_data_from_pkl(target_property, os.path.join(DATA_PATH, data_files[data_choice]))
    graphs = CrystalGraphDataset(dataset, target_properties)
    dataloader = DataLoader(graphs,
                      batch_size=25,
                      shuffle=False,
                      collate_fn=graph_collate,
                      num_workers=4, pin_memory=True)

    print("Computing loss...")
    eval_loss = compute_loss(model, dataloader, factors, device, threshold)
    print(f"The MSA = {eval_loss}")
    input("Press Enter to continue > ")        

def list_files(files):
    file_list = []
    for i, filename in enumerate(files):
        file_list.append(f"[{i}] {filename}")
    return "\n".join(file_list)

def get_files_from_directory(directory):
    """Helper function to list files in a directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == "__main__":
    main()