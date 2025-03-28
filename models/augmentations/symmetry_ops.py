import torch
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.ext.matproj import MPRester

from models.components.crystal import Crystal


class CrystalSymmetry:
    def __init__(self, structure):
        """
        Initializes the CrystalSymmetry class using a pymatgen Structure object.
        
        Parameters:
        - structure: A pymatgen Structure object.
        """
        self.structure = structure
        self.rotation_axes_tensor, self.rotation_angles_list, self.reflection_planes_tensor = self.extract_symmetry_elements()

    def extract_symmetry_elements(self):
        """
        Extracts valid rotation axes, their corresponding rotation angles, and reflection planes from the crystal's symmetry operations.

        Returns:
        - rotation_axes_tensor (torch.Tensor): Tensor containing valid rotation axes as 3D unit vectors.
        - rotation_angles_list (list of list of float): List of valid rotation angles (in degrees) for each axis.
        - reflection_planes_tensor (torch.Tensor): Tensor containing valid reflection plane normals as 3D unit vectors.
        """
        analyzer = SpacegroupAnalyzer(self.structure)
        symmetry_ops = analyzer.get_symmetry_operations()

        conv_structure = analyzer.get_conventional_standard_structure()
        analyzer_conv = SpacegroupAnalyzer(conv_structure, symprec=1e-5)
        symmetry_dataset_conv = analyzer_conv.get_symmetry_dataset()
        print(f"Number of Symmetry Operations (Conventional Cell): {len(symmetry_dataset_conv.rotations)}")  


        rotation_axes = []
        rotation_angles = []
        reflection_planes = []

        for op in symmetry_ops:
            matrix = np.array(op.rotation_matrix, dtype=np.float32)

            if np.linalg.det(matrix) == 1:  # Rotation
                eigvals, eigvecs = np.linalg.eig(matrix)
                for i in range(3):
                    if np.isclose(eigvals[i], 1, atol=1e-6):  # Stricter tolerance
                        axis = eigvecs[:, i].real
                        if np.linalg.norm(axis) > 1e-6:
                            axis /= np.linalg.norm(axis) # Normalize to unit vector

                        # Check for duplicates
                        if not any(np.allclose(axis, existing_axis, atol=1e-6) for existing_axis in rotation_axes):  # Relaxed tolerance
                            rotation_axes.append(axis)

                            # Determine the rotation angle
                            trace = np.trace(matrix)  # Trace of the rotation matrix
                            cos_theta = (trace - 1) / 2
                            #cos_theta = np.clip(cos_theta, -1, 1)  
                            #angle = np.arccos(cos_theta)  
                            cos_theta = np.clip((trace - 1) / 2, -1, 1) # Ensure cos_theta is in valid range [-1, 1]
                            angle = np.arccos(cos_theta) # Rotation angle in radians


                            if np.isclose(angle, 0, atol=1e-6):  # Skip identity rotations (angle = 0)
                                continue

                            order = int(round(2 * np.pi / angle))  # Determine the order of the rotation axis
                            if order <= 0:  # Skip invalid orders
                                continue

                            angles = [(360 / order) * i for i in range(1, order)]  # Generate all rotation angles
                            rotation_angles.append(angles)
                        break  # Only one valid axis per matrix

            elif np.linalg.det(matrix) == -1:  # Reflection
                eigvals, eigvecs = np.linalg.eig(matrix)
                for i in range(3):
                    if np.isclose(eigvals[i], -1, atol=1e-6):  # Stricter tolerance
                        plane_normal = eigvecs[:, i].real
                        plane_normal /= np.linalg.norm(plane_normal)  # Normalize to unit vector

                        # Check for duplicates
                        if not any(np.allclose(plane_normal, existing_plane, atol=1e-6) for existing_plane in reflection_planes):  # Relaxed tolerance
                            reflection_planes.append(plane_normal)
                        break  # Only one valid plane per matrix

        # Convert to PyTorch tensors
        rotation_axes_tensorF = torch.tensor(rotation_axes, dtype=torch.float32) if rotation_axes else torch.empty((0, 3))
        reflection_planes_tensorF = torch.tensor(reflection_planes, dtype=torch.float32) if reflection_planes else torch.empty((0, 3))
        rotation_axes_tensor = torch.tensor(rotation_axes, dtype=torch.float32) if rotation_axes else torch.empty((0, 3), dtype=torch.float32)
        reflection_planes_tensor = torch.tensor(reflection_planes, dtype=torch.float32) if reflection_planes else torch.empty((0, 3), dtype=torch.float32)


        return rotation_axes_tensor, rotation_angles, reflection_planes_tensor

    def get_rotation_axes(self):
        """Returns the rotation axes as a PyTorch tensor (N, 3)."""
        return self.rotation_axes_tensor

    def get_rotation_angles(self):
        """Returns the valid rotation angles (in degrees) for each axis."""
        return self.rotation_angles_list

    def get_reflection_planes(self):
        """Returns the reflection plane normals as a PyTorch tensor (M, 3)."""
        return self.reflection_planes_tensor


# To run write "python -m models.augmentations.symmetry_ops"

# === STEP 1: Get a Crystal from Materials Project ===

API_KEY = "exOmUg204h1fbQsVK65z9U5YLGXK453p"  # Replace with your new API key
material_id = "mp-22862"  # Replace with your desired material ID

with MPRester(api_key=API_KEY) as mpr:
    structure = mpr.get_structure_by_material_id(material_id)

# Extract atomic features (n x f matrix, where f=1 for atomic numbers)
A = torch.tensor([[site.specie.number] for site in structure.sites], dtype=torch.float32)
# Extract atomic positions (n x 3 matrix)
X = torch.tensor(structure.cart_coords, dtype=torch.float32)
# Extract lattice vectors (3 x 3 matrix)
L = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

crystal = Crystal(A, X, L)

symmetry_ops = crystal.get_symmetry_operations()
# print(result)
print("Number:", len(result))

# === STEP 2: Extract Symmetry from the Structure ===
# crystal_symmetry = CrystalSymmetry(structure)

# # === STEP 3: Get Rotation Axes, Rotation Angles & Reflection Planes ===
# rotation_axes_tensor = crystal_symmetry.get_rotation_axes()
# rotation_angles_list = crystal_symmetry.get_rotation_angles()
# reflection_planes_tensor = crystal_symmetry.get_reflection_planes()

# # === STEP 4: Output Results ===
# print(f"Space Group: {structure.get_space_group_info()[0]}")
# print(f"Number of Rotation Axes: {rotation_axes_tensor.shape[0]}")
# print(f"Number of Reflection Planes: {reflection_planes_tensor.shape[0]}")
# print("Rotation Axes and Angles:")
# for i, (axis, angles) in enumerate(zip(rotation_axes_tensor, rotation_angles_list)):
#     print(f"Axis {i + 1}: {axis.numpy()}, Angles: {angles}")
# print("Reflection Plane Normals:\n", reflection_planes_tensor)

# # pip install --upgrade pymatgen