import torch
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

class CrystalSymmetry:
    def __init__(self, crystal):
        """
        Initializes the CrystalSymmetry class using a crystal instance.
        
        Parameters:
        - crystal: An instance of a Crystal class containing A, X, L.
        """
        self.crystal = crystal  # Store the crystal instance
        
        # Convert the crystal data into a pymatgen Structure
        self.structure = self.create_structure()

        # Extract symmetry elements
        self.rotation_axes_tensor, self.reflection_planes_tensor = self.extract_symmetry_elements()
    
    def create_structure(self):
        """Constructs a pymatgen Structure from the given crystal instance."""
        lattice = Lattice(self.crystal.L)  # Create Lattice object
        structure = Structure(lattice, self.crystal.A, self.crystal.X)  # Create Structure object
        return structure
    
    def extract_symmetry_elements(self):
        """
        Extracts valid rotation axes and reflection planes from the crystal's symmetry operations.
        
        Returns:
        - rotation_axes_tensor (torch.Tensor): Tensor containing valid rotation matrices.
        - reflection_planes_tensor (torch.Tensor): Tensor containing valid reflection matrices.
        """
        analyzer = SpacegroupAnalyzer(self.structure)
        symmetry_ops = analyzer.get_symmetry_operations()
        
        rotation_axes = []
        reflection_planes = []
        
        for op in symmetry_ops:
            matrix = torch.tensor(op.rotation_matrix, dtype=torch.float32)
            
            # Rotation Axes: det = 1
            if torch.det(matrix).item() == 1:
                rotation_axes.append(matrix)
            
            # Reflection Planes: det = -1
            elif torch.det(matrix).item() == -1:
                reflection_planes.append(matrix)

        # Convert lists to PyTorch tensors
        rotation_axes_tensor = torch.stack(rotation_axes) if rotation_axes else torch.empty(0)
        reflection_planes_tensor = torch.stack(reflection_planes) if reflection_planes else torch.empty(0)
        
        return rotation_axes_tensor, reflection_planes_tensor

    def get_rotation_axes(self):
        """Returns the rotation axes as a PyTorch tensor."""
        return self.rotation_axes_tensor

    def get_reflection_planes(self):
        """Returns the reflection planes as a PyTorch tensor."""
        return self.reflection_planes_tensor

# Example Usage:
# Assuming you already have a crystal instance with A, P, L
crystal_symmetry = CrystalSymmetry(crystal)

# Access tensors
print("Rotation Axes Tensor Shape:", crystal_symmetry.get_rotation_axes().shape)
print("Reflection Planes Tensor Shape:", crystal_symmetry.get_reflection_planes().shape)
