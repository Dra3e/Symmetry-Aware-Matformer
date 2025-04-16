import json
import pickle
import torch
from models.components.crystal import Crystal
from pymatgen.core import Structure
from pathlib import Path

def convert_pkl_to_json(pkl_path: str, json_path: str):
    """
    Converts a .pkl file to a .json file.

    Args:
        pkl_path (str): Path to the .pkl file.
        json_path (str): Path where the .json file should be saved.
    """
    # Convert string paths to Path objects
    pkl_file = Path(pkl_path)
    json_file = Path(json_path)

    # Load the .pkl file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Save the data as a .json file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4, default=str)  # default=str helps with non-serializable objects

    print(f"Saved JSON to {json_file}")

import pickle
import json
from pathlib import Path

def extract_crystal_from_pkl(pkl_path: str, json_path: str, crystal_id):
    """
    Extracts a single crystal from a .pkl file and saves it as a .json file.

    Args:
        pkl_path (str): Path to the .pkl file.
        json_path (str): Path where the .json file should be saved.
        crystal_id (int or str): Index or key of the crystal to extract.
    """
    pkl_file = Path(pkl_path)
    json_file = Path(json_path)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Extract the specific crystal
    try:
        crystal = data[crystal_id]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Could not extract crystal '{crystal_id}' from the data: {e}")

    # Save that one crystal to a JSON file
    with open(json_file, 'w') as f:
        json.dump(crystal, f, indent=4, default=str)

    print(f"Saved crystal '{crystal_id}' to {json_file}")

def get_training_data(property_type, json_path: str):
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
    with open(json_path, "r") as f:
        crystal_data_list = json.load(f)  # Assuming the JSON file contains a list of crystals

    crystals = []  # List to store valid Crystal objects
    property_values = []  # List to store valid property values

    for crystal_data in crystal_data_list:
        property_value = crystal_data.get(property_type, None)
        if property_value is None:
            print(f"Warning: {property_type} not found in entry {crystal_data.get('id', 'unknown')}")
            continue  # Skip this crystal if the property is not found

        # Extract atoms section
        atoms = crystal_data["atoms"]

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

        # Append to lists
        crystals.append(crystal_instance)
        property_values.append(property_value)

    return crystals, property_values

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
            continue  # Skip this crystal if the property is not found

        # Extract atoms section
        atoms = crystal_data["atoms"]

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

        # Append to lists
        crystals.append(crystal_instance)
        property_values.append(property_value)

    return crystals, property_values