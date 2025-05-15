from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from models.scripts.invalid_augmentations import InvalidAugmenter


class ACG():
    def __init__(self, cutoff=3.0):
        self.cutoff = cutoff
        self.invalid_aug = InvalidAugmenter()

    def build_graphs(self, C: Crystal):
        valid_syms = C.get_symmetry_operations()
        aug_valid = C.apply_augmentations(valid_syms)
        g_valid = CrystalGraph.init_from_augmented_crystal(
            original_crystal=C, 
            augmented_crystals=aug_valid,
            m_valid=1,
            config=self.cutoff
        )

        inv_syms = self.invalid_aug.forward(valid_syms)
        aug_inv = C.apply_augmentations(inv_syms)
        g_inv = CrystalGraph.create_batch(
            crystals=aug_inv,
            m_valid=0
        )

        return g_inv + [g_valid]