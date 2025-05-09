from __future__ import annotations
import os
from pathlib import Path
import pickle
from typing import Callable, Sequence
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from pymatgen.core import Structure
from models.components.crystal import Crystal

class DataGetter(Dataset):
    """
    Class for loading and processing crystal data from a PKL file.
    This class handles the loading of crystal structures and their associated properties,
    and provides a DataLoader for batching and shuffling the data.
    """
    def __init__(self, *, path: str | Path, property_name: str, graph_builder: Callable[[Crystal], list], batch_size=32, shuffle=True, collate_fn: Callable | None = None) -> None:
        """
        Initializes the DataGetter

        Args:
            path (str | Path): Path to the PKL file containing crystal data.
            property_name (str): The key in the PKL file that contains the target property.
            graph_builder (Callable[[Crystal], list]): Function to convert a Crystal object into a graph representation.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            collate_fn (Callable | None, optional): Custom collate function for DataLoader. Defaults to None.
        """
        super().__init__()

        # Load crystals and properties from the PKL file
        self.crystals, self.targets = self.load_crystals(path, property_name)
        assert self.crystals, (
            "No valid crystals found – check your PKL path and property key."
        )
        self.num_data_points = len(self.crystals) 

        self.graph_builder = graph_builder
        self._graphs: list | None = [None] * len(self.crystals)  # On‑demand cache

        # Initilizes a data loader for batching and shuffling the data
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(2, os.cpu_count() // 2) if torch.cuda.is_available() else 0, # Use multiple workers if CUDA is available
            collate_fn=collate_fn or DataGetter._default_collate, # Use custom or default collate function
            pin_memory=torch.cuda.is_available(), # Pin memory for faster data transfer to GPU
        )
    
    @staticmethod
    def load_crystals(pkl_path: str | Path, property_name: str) -> tuple[list[Crystal], list[float | int | torch.Tensor]]:
        """
        Inline replacement for get_training_data_from_pkl.
        Loads crystal data and their associated properties from a pickle file.

        Args:
            pkl_path (str | Path): Path to the pickle file.
            property_name (str): The key for the target property in the data.

        Returns:
            tuple: A tuple containing a list of Crystal objects and their corresponding properties.
        """
        with open(pkl_path, "rb") as f:
            raw: Sequence[dict] = pickle.load(f) # Load serialized data

        crystals, props = [], []
        for entry in raw:
            value = entry.get(property_name) # Get the target property
            if value is None:
                print(f"⚠️  {property_name} missing for id={entry.get('id', 'NA')}")
                continue

            atoms = entry["atoms"] # Extract atomic data
            try:
                # Create a pymatgen Structure object
                struct = Structure(
                    lattice=atoms["lattice_mat"],
                    species=atoms["elements"],
                    coords=atoms["coords"],
                    coords_are_cartesian=atoms.get("cartesian", False),
                )
                # Convert structure data to tensors
                A = torch.tensor(
                    [[s.specie.number] for s in struct.sites], dtype=torch.int64
                )
                X = torch.tensor(struct.cart_coords, dtype=torch.float32)
                L = torch.tensor(struct.lattice.matrix, dtype=torch.float32)

                crystals.append(Crystal(A, X, L))
                props.append(value)

            except Exception as exc:
                print(f"❌ failed to parse id={entry.get('id', 'NA')}: {exc}")

        return crystals, props
    
    @staticmethod
    def _default_collate(batch):
        """
        Default collate function for DataLoader.
        Keeps the graph-list objects as plain Python lists.

        Args:
            batch: A batch of data.

        Returns:
            tuple: A tuple containing a list of graphs and a tensor of targets.
        """
        graphs, targets = zip(*batch) # Separate graphs and targets
        return list(graphs), default_collate(targets)

    def __len__(self) -> int:
        """
        Returns the total number of crystals.

        Returns:
            int: Number of crystals.
        """
        return len(self.crystals)

    def __getitem__(self, idx: int):
        """
        Retrieves a single graph by index.

        Args:
            idx (int): Index of the graph.

        Returns:
            tuple: A tuple containing the graph representation and the target property.
        """
        if self._graphs[idx] is None: # If graph is not cached, build it
            self._graphs[idx] = self.graph_builder(self.crystals[idx])
        return self._graphs[idx], self.targets[idx]

    # Convenience wrappers
    def __iter__(self):
        """
        Returns an iterator over the DataLoader.

        Returns:
            iterator: An iterator over the DataLoader.
        """
        return iter(self.loader)

    def __call__(self):
        """
        Returns the DataLoader instance.

        Returns:
            DataLoader: The DataLoader for this dataset.
        """
        return self.loader