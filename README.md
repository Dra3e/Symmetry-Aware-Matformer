# Symmetry-Aware Matformer (SAM)
Find new crystals with new properties using E(3) invariance rules to train a SSL neural network.

# Crystal Graph Neural Network (CGNN)

This project provides a framework for training a **Self-Supervised Learning (SSL)** neural network to predict properties of crystals using **E(3) invariance rules**. The codebase includes components for crystal graph generation, augmentation, and ablation studies to evaluate the impact of various model components.

----------------------------------------------------------------------------------------------

## Features

- **Crystal Graph Representation**: Converts crystal structures into graph representations for neural network training.
- **Augmented Crystal Generator (ACG)**: Generates valid and invalid augmented crystal graphs for training.
- **Ablation Studies**: Evaluates the impact of removing specific components (e.g., ACG, contrastive loss, group theory) on model performance.
- **Customizable Training Pipeline**: Supports various loss functions, hyperparameters, and graph builders.
- **E(3) Invariance**: Ensures the model respects symmetry and invariance rules for crystal structures.

----------------------------------------------------------------------------------------------

## Code Structure

### 1. **Models**
- **`models/components/ACG.py`**: Defines the `ACG` class, which generates a list of crystal graphs from valid and invalid symmetry-based augmentations of a given crystal.
- **`models/components/crystal_graph.py`**: Implements the `CrystalGraph` class for converting crystals into graph representations with atomic features and interatomic edges, supporting augmentation and batching.
- **`models/components/crystal.py`**: Implements the `Crystal` class representing a crystal structure with atomic features, positions, and lattice vectors, including methods for symmetry extraction, augmentation, and translation handling.
- **`models/components/readout.py`**: Defines the `Readout` module, which aggregates atomic features and predicts crystal properties using an update mechanism, global pooling, and a final prediction head.
- **`models/components/transformer_layer.py`**: Implements a multi-head `transformer layer` that performs attention-based message passing on atomic graphs using learned edge and node representations.

- **`models/scripts/attention_function.py`**: Implements a multi-headed attention mechanism for graph edges, incorporating symmetry-aware and validity-masked attention using node and edge features.
- **`models/scripts/crystal_graph_vectorize.py`**: Constructs edge lists for crystal graphs by computing pairwise distances and applying symmetry and validity masks within a cutoff radius.
- **`models/scripts/global_pooling.py`**: Defines a `global pooling` layer that aggregates node-level features into a crystal-level representation by summing atom features.
- **`models/scripts/graph_embedding.py`**: Implements the `GraphEmbedder` class, which transforms crystal structures into learned node and edge features using atomic embeddings (via CGCNN) and distance-based RBF transformations for edge construction.
- **`models/scripts/invalid_augmentations.py`**: Defines the `InvalidAugmenter` class that retrieves all symmetry operations from the spglib database and computes which ones are not valid for a given crystal structure, useful for contrastive or augmentation-based learning.
- **`models/scripts/predicted_property.py`**: Defines a lightweight neural network module for predicting crystal properties from pooled features using two linear layers and SiLU activation.
- **`models/scripts/update.py`**: Implements the feature `Update` step in the transformer, combining atom and message features with normalization and a feed-forward network for enriched atom representations.

### 2. **Training**
- **`training/engine.py`**: Core training engine for the SAM model.
- **`training/ablation_studies/`**: Contains scripts for ablation studies:
  - `SAM_no_ACG.py`: Trains the model without the ACG component.
  - `SAM_no_contrastive.py`: Trains the model without contrastive loss.
  - `SAM_no_group.py`: Trains the model without group theory.
- **`training/data_getter.py`**: Handles data loading and preprocessing.

### 3. **Setup**
- **`setup.py`**: Configuration file for paths and dependencies.

----------------------------------------------------------------------------------------------

## Ablation Studies

The ablation studies evaluate the importance of specific components in the SAM model:
1. **No ACG**: Excludes the Automatic Crystal Graph component.
2. **No Contrastive Loss**: Replaces contrastive loss with Mean Squared Error (MSE).
3. **No Group Theory**: Removes group theory-based symmetry operations.

Each study trains a modified version of the model and compares its performance to the original.

----------------------------------------------------------------------------------------------

## Installation

1. Clone the repository:
  ```bash
  git clone "main"
  cd 

2. Create folders
  **Data** Create data folder in the root directory and add your data files there.
  **Results** Create results folder in the /traning/ folder, or run the program and it will create the folder for you. 