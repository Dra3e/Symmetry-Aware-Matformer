import json
import pickle
import random
import torch
# from mp_api.client import MPRester
from models.components.crystal import Crystal
from pymatgen.core import Structure
from pathlib import Path

API_KEY = "exOmUg204h1fbQsVK65z9U5YLGXK453p"

# def fetch_crystals_from_mp(crystals):
#     """
#     Fetches a specified number of crystal structures from the Materials Project database using the MPRester API.
#     Filters the crystals based on certain criteria (e.g., stable crystals with 2-5 elements) and saves them into JSON files.
#     """
#     total_crystals = crystals  # Total number of crystals to fetch from Materials Project
#     batch_size = 100  # Number of crystals to fetch in each API call (batch size)

#     all_crystals = []  # List to store all fetched crystals
#     num_fetched = 0  # Counter to track how many crystals have been fetched so far

#     # Using the MPRester class to interact with Materials Project API
#     with MPRester(API_KEY) as mpr:
#         while num_fetched < total_crystals:
#             # Search for crystals based on certain criteria (e.g., stable crystals with 2-5 elements)
#             docs = mpr.materials.search(
#                 fields = [  # Fields to retrieve for each material
#                     'builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 
#                     'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 
#                     'symmetry', 'material_id', 'structure', 'deprecated', 'deprecation_reasons', 'initial_structures', 
#                     'task_ids', 'deprecated_tasks', 'calc_types', 'last_updated', 'created_at', 'origins', 'warnings', 
#                     'task_types', 'run_types', 'entries'
#                 ],
#                 # energy_above_hull = (0, 0.1),  # Filter for stable crystals (energy above hull <= 0.1)
#                 # num_elements = (2, 5),  # Only fetch crystals with 2 to 5 elements
#                 chunk_size = batch_size,  # Number of crystals per request (batch size)
#                 # offset = num_fetched  # Offset to keep track of how many have been fetched so far
#             )

#             if not docs:
#                 break  # If no more results are returned, exit the loop

#             # Extend the list of all crystals with the newly fetched docs
#             all_crystals.extend(docs)
#             num_fetched += len(docs)  # Update the counter for the number of crystals fetched
#             print(f"Fetched {num_fetched} crystals so far...")  # Print progress

#     print(f"Total crystals fetched: {len(all_crystals)}")  # Print total number of fetched crystals

#     random.shuffle(all_crystals)  # Shuffle the list to ensure randomness in the dataset

#     # Split the data into training and evaluation sets (80% for training, 20% for evaluation)
#     split_index = int(0.8 * len(all_crystals))  # Index for splitting the dataset
#     train_data = all_crystals[:split_index]  # First 80% for training
#     eval_data = all_crystals[split_index:]  # Remaining 20% for evaluation

#     # Save the training and evaluation data into separate JSON files
#     with open("train_crystals.json", "w") as f:
#         json.dump(train_data, f, indent=4)  # Write training data to a JSON file

#     with open("eval_crystals.json", "w") as f:
#         json.dump(eval_data, f, indent=4)  # Write evaluation data to a JSON file

#     # Print the sizes of the datasets
#     print(f"Training set: {len(train_data)} crystals")
#     print(f"Evaluation set: {len(eval_data)} crystals")

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
    Takes a single crystal dictionary (with 'atoms' format), extracts the specified property,
    and returns a list with one tuple: (Crystal object, property value).

    Args:
        property_type (str): The name of the property to extract (e.g., 'gap pbe').
        crystal_data (dict): A single crystal entry in the custom format.

    Returns:
        list: A list containing one tuple: (Crystal, property_value)
    """
    with open(json_path, "r") as f:
        crystal_data = json.load(f)

    # Get the property value
    property_value = crystal_data.get(property_type, None)
    if property_value is None:
        print(f"Warning: {property_type} not found in entry {crystal_data.get('id', 'unknown')}")
        return []

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
    crystal_list = []
    crystal_list.append(crystal_instance)

    return crystal_list, property_value

def get_crystals_and_properties(property_type, json_path: str):
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
        # Get the property value
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