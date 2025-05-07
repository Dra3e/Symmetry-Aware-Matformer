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
    Dataset + ready‑made DataLoader that produces exactly the same
    (graphs_list, target) pairs your old pipeline did.

    Pass in **your** `graph_builder(crystal) -> list[Graph]`
    (e.g. the old `CrystalGraphDataset._build_graph_list`).

    Example
    -------
    train_dl = DataGetter(
        path="data/train.pkl",
        property_name="gap pbe",
        graph_builder=my_build_graph_list,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    ).loader
    """
    def __init__(self, *, path: str | Path, property_name: str, graph_builder: Callable[[Crystal], list], batch_size=32, shuffle=True, collate_fn: Callable | None = None) -> None:
        super().__init__()

        self.crystals, self.targets = self.load_crystals(path, property_name)
        assert self.crystals, (
            "No valid crystals found – check your PKL path and property key."
        )
        self.num_data_points = len(self.crystals)

        self.graph_builder = graph_builder
        self._graphs: list | None = [None] * len(self.crystals)  # on‑demand cache

        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(2, os.cpu_count() // 2) if torch.cuda.is_available() else 0,
            collate_fn=collate_fn or DataGetter._default_collate,
            pin_memory=torch.cuda.is_available(),
        )
    
    @staticmethod
    def load_crystals(pkl_path: str | Path, property_name: str) -> tuple[list[Crystal], list[float | int | torch.Tensor]]:
        """
        Inline replacement for get_training_data_from_pkl.
        """
        with open(pkl_path, "rb") as f:
            raw: Sequence[dict] = pickle.load(f)

        crystals, props = [], []
        for entry in raw:
            value = entry.get(property_name)
            if value is None:
                print(f"⚠️  {property_name} missing for id={entry.get('id', 'NA')}")
                continue

            atoms = entry["atoms"]
            try:
                struct = Structure(
                    lattice=atoms["lattice_mat"],
                    species=atoms["elements"],
                    coords=atoms["coords"],
                    coords_are_cartesian=atoms.get("cartesian", False),
                )
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
        """Keep the graph-list objects as plain Python lists."""
        graphs, targets = zip(*batch)          # lists of length B
        return list(graphs), default_collate(targets)

    def __len__(self) -> int:
        return len(self.crystals)

    def __getitem__(self, idx: int):
        if self._graphs[idx] is None:               # construct once, then reuse
            self._graphs[idx] = self.graph_builder(self.crystals[idx])
        return self._graphs[idx], self.targets[idx]

    # Convenience wrappers
    def __iter__(self):
        return iter(self.loader)

    def __call__(self):
        return self.loader