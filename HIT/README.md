# HIT - Humanoid Imitation with Transformers

This package implements the Humanoid Imitation with Transformers (HIT) framework for robotic imitation learning from human demonstrations.

## Files Overview

- `constants.py`: Contains configuration settings for various tasks, including dataset paths, camera setups, and observation/action dimensions.
- `policy.py`: Implements various policy architectures including HITPolicy, ACTPolicy, CNNMLPPolicy, and DiffusionPolicy.
- `model_util.py`: Utility functions for creating and configuring policy and optimizer instances.
- `utils.py`: Provides data handling utilities including dataset loading, normalization, and batch sampling.
- `imitate_episodes_h1_train.py`: Main training script for behavioral cloning of human demonstrations.
- `realtime_inference.py`: Provides a wrapper for real-time sequential inference with trained HIT policies.
- `example_realtime_inference.py`: Example script demonstrating how to use the real-time inference model.

## Data Structure

The implementation supports various demonstration datasets stored in HDF5 format, including:
- Fold clothes tasks
- Object rearrangement tasks

## Usage

To train a model:

```bash
python imitate_episodes_h1_train.py --task data_rearrange_objects --policy_class HIT
```

To run real-time inference with a trained model:

```bash
python example_realtime_inference.py --policy_path /path/to/checkpoint.ckpt --norm_stats_path /path/to/stats.pkl
```

## Real-time Inference

The `RealtimeInferenceModel` provides a convenient wrapper for deploying trained HIT policies in real-time applications:

- Sequential processing of inputs and outputs
- Efficient preprocessing and postprocessing
- Performance monitoring
- Support for delayed actions
- Ability to maintain state between timesteps

Example usage:

```python
from HIT.realtime_inference import RealtimeInferenceModel

# Create model (see example_realtime_inference.py for complete example)
inference_model = RealtimeInferenceModel(
    policy=trained_policy,
    norm_stats=norm_stats,
    camera_names=['cam_high', 'cam_low'],
    observation_names=['qpos'],
    device="cuda"
)

# Run inference in a loop
while True:
    # Get observations from sensors
    images, observations = get_sensor_data()
    
    # Get action from model
    action = inference_model.step(images, observations)
    
    # Apply action to robot
    robot.apply_action(action)
```

## Requirements

- PyTorch
- NumPy
- OpenCV
- H5py
- DETR (Detection Transformer) dependencies
