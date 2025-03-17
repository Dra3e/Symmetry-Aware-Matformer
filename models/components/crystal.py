import torch



class Crystal:
    def __init__(self, A, X, L):
        # Initialize the crystal with atomic features (A), atomic coordinates (X), and lattice vectors (L)
        self.A = A.clone().detach() if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
        self.X = X.clone().detach() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.L = L.clone().detach() if isinstance(L, torch.Tensor) else torch.tensor(L, dtype=torch.float32)

    def apply_augmentations(self, axes, planes, directions):
        """
        Generates valid augmentations for a crystal structure based on the given axes and planes.
        Keyword arguments:
        - axes: List of Axis objects representing rotation axes.
        - planes: List of Plane objects representing reflection planes.
        """
        augmentations = []

        # Apply translation
        for translation in directions:
            augmented_crystal = Crystal(self.A, self.X, self.L)
            if translation is not None:
                # Convert translation to a PyTorch tensor with integer dtype
                translation_tensor = torch.tensor(translation, dtype=torch.int32)
                augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = augmented_crystal.translate_crystal(
                    self.A, self.X, self.L, translation_tensor
                )
            augmentations.append(augmented_crystal)
            # print(f"Translation: {translation}, Augmentations: {len(augmentations)}")

        # Apply rotations
        for axis in axes:
            for angle in [90, 180, 270]:
                augmented_crystal = Crystal(self.A, self.X, self.L)
                augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = axis.rotate_crystal(
                    self.A, self.X, self.L, axis.V, angle
                )
                augmentations.append(augmented_crystal)
                # print(f"Rotation: Axis {axis.V}, Angle {angle}, Augmentations: {len(augmentations)}")

        # Apply reflections
        for plane in planes:
            augmented_crystal = Crystal(self.A, self.X, self.L)
            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = plane.reflect_crystal(
                self.A, self.X, self.L, plane.N
            )
            augmentations.append(augmented_crystal)
            # print(f"Reflection: Plane {plane.N}, Augmentations: {len(augmentations)}")

        # Apply combinations of transformations

        # Translation + Rotation
        for translation in directions:
            for axis in axes:
                for angle in [90, 180, 270]:
                    augmented_crystal = Crystal(self.A, self.X, self.L)
                    if translation is not None:
                        translation_tensor = torch.tensor(translation, dtype=torch.int32)
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = augmented_crystal.translate_crystal(
                            self.A, self.X, self.L, translation_tensor
                        )
                    augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = axis.rotate_crystal(
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, axis.V, angle
                    )
                    augmentations.append(augmented_crystal)
                    # print(f"Translation + Rotation: {translation}, Axis {axis.V}, Angle {angle}, Augmentations: {len(augmentations)}")
        
        # Translation + Reflection
        for translation in directions:
            for plane in planes:
                augmented_crystal = Crystal(self.A, self.X, self.L)
                if translation is not None:
                    translation_tensor = torch.tensor(translation, dtype=torch.int32)
                    augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = augmented_crystal.translate_crystal(
                        self.A, self.X, self.L, translation_tensor
                    )
                augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = plane.reflect_crystal(
                    augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, plane.N
                )
                augmentations.append(augmented_crystal)
                # print(f"Translation + Reflection: {translation}, Plane {plane.N}, Augmentations: {len(augmentations)}")
        
        # Rotation + Reflection
        for axis in axes:
            for angle in [90, 180, 270]:
                for plane in planes:
                    augmented_crystal = Crystal(self.A, self.X, self.L)
                    augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = axis.rotate_crystal(
                        self.A, self.X, self.L, axis.V, angle
                    )
                    augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = plane.reflect_crystal(
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, plane.N
                    )
                    augmentations.append(augmented_crystal)
                    # print(f"Rotation + Reflection: Axis {axis.V}, Angle {angle}, Plane {plane.N}, Augmentations: {len(augmentations)}")

        # Translation + Rotation + Reflection
        for translation in directions: 
            for axis in axes:
                for angle in [90, 180, 270]:
                    for plane in planes:
                        augmented_crystal = Crystal(self.A, self.X, self.L)
                        if translation is not None:
                            translation_tensor = torch.tensor(translation, dtype=torch.int32)
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = augmented_crystal.translate_crystal(
                                self.A, self.X, self.L, translation_tensor
                            )
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = axis.rotate_crystal(
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, axis.V, angle
                        )
                        augmented_crystal.A, augmented_crystal.X, augmented_crystal.L = plane.reflect_crystal(
                            augmented_crystal.A, augmented_crystal.X, augmented_crystal.L, plane.N
                        )
                        augmentations.append(augmented_crystal)
                        # print(f"Translation + Rotation + Reflection: {translation}, Axis {axis.V}, Angle {angle}, Plane {plane.N}, Augmentations: {len(augmentations)}")

        return augmentations

    def translate_crystal(self, coefficients): 
        """
        Move a crystal structure through 3D space by a translation vector that is a multiple of the lattice vectors.

        Parameters:
        - coefficients: Tensor of shape (3,) containing integer coefficients for the lattice translation.

        Returns:
        - A_translated: Same as A.
        - X_translated: Tensor of shape (n, 3) with atomic coordinates shifted by a lattice-based translation vector.
        - L_translated: Same as L.
        """
        try:
            # Ensure coefficients is a tensor of integers
            if not isinstance(coefficients, torch.Tensor):
                coefficients = torch.tensor(coefficients, dtype=torch.int32)
            if len(coefficients) != 3 or torch.is_floating_point(coefficients):
                raise ValueError("Translation coefficients must be a 3-element integer tensor")
        except AttributeError:
            raise ValueError("Invalid Input")
        
        # Compute translation vector as an integer combination of lattice vectors
        translation_vector = torch.matmul(coefficients.float(), self.L)

        # Apply translation
        X_translated = self.X + translation_vector

        return self.A, X_translated, self.L