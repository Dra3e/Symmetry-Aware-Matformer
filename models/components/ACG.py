from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from models.scripts.invalid_augmentations import InvalidAugmenter


class ACG():
    """
    ACG class for generating a list of crystal graphs based on valid and invalid augmentations of a crystal.
    
    Attributes:
        cutoff (float): The maximum distance two atoms can have from each other and still be considered to have influence upon each other.
        invalid_aug (InvalidAugmenter): An instance of the InvalidAugmenter class, used to get the invalid augmentations of the crystal.
    """
    def __init__(self, cutoff=3.0):
        """
        Initializes the ACG class with a cutoff distance for edge connections and an instance of the InvalidAugmenter class.

        Args:
            cutoff (float): The cutoff distance for edge connections. Default is 3.0.
            invalid_aug (InvalidAugmenter): An instance of the InvalidAugmenter class that creates the invalid augmentations.
        """
        self.cutoff = cutoff
        self.invalid_aug = InvalidAugmenter()

    def build_graphs(self, C: Crystal):
        """
        Generates a list of crystal graphs, with one valid and several invalid augmentations.
        The valid crystal representing graph is located at the end of the outputted list; [-1].

        Args:
            C (Crystal): The input crystal object, which that graphs are generated from.

        Returns:
            crystal_graph_list: A list containing all the graph representations of the crystal.
        """
        valid_syms = C.get_symmetry_operations()        # Get valid symmetry operations from the crystal
        aug_valid = C.apply_augmentations(valid_syms)   # Create valid augmentations of the original crystal
        
        # Construct a single crystal graph from all the valid augmented crystals
        g_valid = CrystalGraph.init_from_augmented_crystal(
            original_crystal=C, 
            augmented_crystals=aug_valid,
            m_valid=1,
            cutoff=self.cutoff
        )
        
        inv_syms = self.invalid_aug.forward(valid_syms) # Get some invalid symmetry operations from the crystal
        aug_inv = C.apply_augmentations(inv_syms)       # Create the invalid augmentations of the original crystal

        # Generate a list of crystal graphs, one for each invalid augmentation.
        g_inv = CrystalGraph.create_batch(
            crystals=aug_inv,
            m_valid=0
        )
        return g_inv + [g_valid]                        # Return the combined list of crystal graphs.