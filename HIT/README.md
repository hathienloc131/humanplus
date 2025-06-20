# HIT - Humanoid Imitation with Transformers

This package implements the Humanoid Imitation with Transformers (HIT) framework for robotic imitation learning from human demonstrations.

## Files Overview

- `constants.py`: Contains configuration settings for various tasks, including dataset paths, camera setups, and observation/action dimensions.
- `policy.py`: Implements various policy architectures including HITPolicy, ACTPolicy, CNNMLPPolicy, and DiffusionPolicy.
- `model_util.py`: Utility functions for creating and configuring policy and optimizer instances.
- `utils.py`: Provides data handling utilities including dataset loading, normalization, and batch sampling.
- `imitate_episodes_h1_train.py`: Main training script for behavioral cloning of human demonstrations.

## Data Structure

The implementation supports various demonstration datasets stored in HDF5 format, including:
- Fold clothes tasks
- Object rearrangement tasks

## Usage

To train a model:

```bash
python imitate_episodes_h1_train.py --task data_rearrange_objects --policy_class HIT
```

## Requirements

- PyTorch
- NumPy
- OpenCV
- H5py
- DETR (Detection Transformer) dependencies
