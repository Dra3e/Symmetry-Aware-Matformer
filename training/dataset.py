import pickle
import torch
from models.components.crystal import Crystal
from pymatgen.core import Structure
from pathlib import Path

def extract_crystals_from_pkl(pkl_path: str, nr_of_crystals):
    """
    Extracts a number of crystals from a .pkl file and saves it as a .pkl file.

    Args:
        pkl_path (str): Path to the .pkl file.
        nr_of_crystals (int): The number of crystals to extract.
    """
    pkl_file = Path(pkl_path)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Extract the specific crystals
    crystals = []
    i = 0
    for crystal in data:
        crystals.append(crystal)
        i += 1
        if i >= nr_of_crystals:
            break

    # Save the crystals to a pkl file
    with open('data/sample_crystals.pkl', 'wb') as file:
        pickle.dump(crystals, file) 

    print("Saved crystals to sample_crystals.pkl")

def get_training_data_from_pkl(property_type, pkl_path: str):
    """
    Processes multiple crystals from a JSON file, extracts their property values,
    and stores all valid crystals in one list and their property values in another.

    Args:
        property_type (str): The name of the property to extract (e.g., 'gap pbe').
        json_path (str): Path to the JSON file containing crystal data.

    Returns:
        tuple: A tuple containing:
            - List of Crystal objects.
            - List of property values.
    """
    with open(pkl_path, 'rb') as f:
        crystal_data_list = pickle.load(f)

    crystals = []  # List to store valid Crystal objects
    property_values = []  # List to store valid property values

    for crystal_data in crystal_data_list:
        property_value = crystal_data.get(property_type, None)
        if property_value is None:
            print(f"Warning: {property_type} not found in entry {crystal_data.get('id', 'unknown')}")
            continue

        # Extract atoms section
        atoms = crystal_data["atoms"]
        try:
            # Build pymatgen Structure object
            structure = Structure(
                lattice=atoms["lattice_mat"],
                species=atoms["elements"],
                coords=atoms["coords"],
                coords_are_cartesian=atoms.get("cartesian", False)
            )

            # Convert to tensors
            A = torch.tensor([[site.specie.number] for site in structure.sites], dtype=torch.int64)
            X = torch.tensor(structure.cart_coords, dtype=torch.float32)
            L = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

            # Create Crystal object
            crystal_instance = Crystal(A, X, L)

            if crystal_instance is None:
                print(f"⚠️ Crystal object is None for ID {crystal_data.get('id', 'unknown')}")
                continue
            
            # Append to lists
            crystals.append(crystal_instance)
            property_values.append(property_value)

        except Exception as e:
            print(f"❌ Failed to process crystal ID {crystal_data.get('id', 'unknown')}: {e}")
            continue

    return crystals, property_values