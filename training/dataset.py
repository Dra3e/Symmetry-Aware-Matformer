import json
import random
import torch
from mp_api.client import MPRester
from models.components.crystal import Crystal
from pymatgen.core import Structure

API_KEY = "exOmUg204h1fbQsVK65z9U5YLGXK453p"

def fetch_crystals_from_mp():
    """
    Fetches a specified number of crystal structures from the Materials Project database using the MPRester API.
    Filters the crystals based on certain criteria (e.g., stable crystals with 2-5 elements) and saves them into JSON files.
    """
    TOTAL_CRYSTALS = 1000  # Total number of crystals to fetch from Materials Project
    BATCH_SIZE = 100  # Number of crystals to fetch in each API call (batch size)

    all_crystals = []  # List to store all fetched crystals
    num_fetched = 0  # Counter to track how many crystals have been fetched so far

    # Using the MPRester class to interact with Materials Project API
    with MPRester(API_KEY) as mpr:
        while num_fetched < TOTAL_CRYSTALS:
            # Search for crystals based on certain criteria (e.g., stable crystals with 2-5 elements)
            docs = mpr.materials.search(
                fields = [  # Fields to retrieve for each material
                    "material_id", "structure", "band_gap", "density", "formation_energy_per_atom",
                    "energy_above_hull", "volume", "spacegroup", "lattice", "elasticity", "dielectric_constant",
                    "bulk_modulus", "shear_modulus", "poisson_ratio", "fermi_energy", "specific_heat",
                    "optical_gap", "elastic_modulus", "magnetization", "thermoelectric_power_factor", "vibration_modes",
                    "critical_temperature"
                ],
                energy_above_hull = (0, 0.1),  # Filter for stable crystals (energy above hull <= 0.1)
                num_elements = (2, 5),  # Only fetch crystals with 2 to 5 elements
                chunk_size = BATCH_SIZE,  # Number of crystals per request (batch size)
                offset = num_fetched  # Offset to keep track of how many have been fetched so far
            )

            if not docs:
                break  # If no more results are returned, exit the loop

            # Extend the list of all crystals with the newly fetched docs
            all_crystals.extend(docs)
            num_fetched += len(docs)  # Update the counter for the number of crystals fetched
            print(f"Fetched {num_fetched} crystals so far...")  # Print progress

    print(f"Total crystals fetched: {len(all_crystals)}")  # Print total number of fetched crystals

    random.shuffle(all_crystals)  # Shuffle the list to ensure randomness in the dataset

    # Split the data into training and evaluation sets (80% for training, 20% for evaluation)
    split_index = int(0.8 * len(all_crystals))  # Index for splitting the dataset
    train_data = all_crystals[:split_index]  # First 80% for training
    eval_data = all_crystals[split_index:]  # Remaining 20% for evaluation

    # Save the training and evaluation data into separate JSON files
    with open("train_crystals.json", "w") as f:
        json.dump(train_data, f, indent=4)  # Write training data to a JSON file

    with open("eval_crystals.json", "w") as f:
        json.dump(eval_data, f, indent=4)  # Write evaluation data to a JSON file

    # Print the sizes of the datasets
    print(f"Training set: {len(train_data)} crystals")
    print(f"Evaluation set: {len(eval_data)} crystals")


def get_training_data(property_type):
    """
    Loads training data from the 'train_crystals.json' file, extracts a specified property for each crystal,
    and returns a list of tuples containing a Crystal object and its corresponding property value.

    Args:
        property_type (str): The name of the property to extract for each crystal (e.g., 'band_gap', 'density').

    Returns:
        list: A list of tuples, where each tuple contains:
              - A Crystal object representing the atomic features, positions, and lattice of the crystal.
              - The value of the specified property for that crystal.
    """
    
    # Load the training data from the 'train_crystals.json' file
    with open("train_crystals.json", "r") as f:
        train_data = json.load(f)

    output = []  # List to store the final output of Crystal objects and their properties

    # Loop through each crystal in the training data
    for crystal in train_data:
        # Extract the relevant property for each crystal, return None if property is not found
        property_value = crystal.get(property_type, None)

        # Check if the property exists for the current crystal
        if property_value is None:
            # Print a warning message if the property is not found
            print(f"Warning: {property_type} not found for crystal {crystal['material_id']}")
            continue  # Skip the current crystal if the property is missing

        # Extract atomic features (n x f matrix, where f=1 for atomic numbers)
        structure = Structure.from_dict(crystal["structure"])  # Convert the structure from dictionary format
        A = torch.tensor([[site.specie.number] for site in structure.sites], dtype=torch.float32)  # Atomic numbers

        # Extract atomic positions (n x 3 matrix) in Cartesian coordinates
        X = torch.tensor(structure.cart_coords, dtype=torch.float32)

        # Extract lattice vectors (3 x 3 matrix) describing the crystal's unit cell
        L = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

        # Create a Crystal object using the extracted atomic features, positions, and lattice
        crystal_instance = Crystal(A, X, L)

        # Append the Crystal object and its corresponding property value as a tuple to the output list
        output.append((crystal_instance, property_value))

    return output  # Return the list of tuples containing Crystal objects and their properties