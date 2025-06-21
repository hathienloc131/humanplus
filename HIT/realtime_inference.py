"""
Real-time inference model for HIT (Humanoid Imitation with Transformers) framework.

This module provides a wrapper for real-time sequential inference with HIT policies.
It handles preprocessing of inputs, maintaining state between inference steps,
and postprocessing of outputs for direct use in control applications.
"""

import torch
import numpy as np
import cv2
import time
import torchvision.transforms as transforms
from collections import deque

class RealtimeInferenceModel:
    """
    Real-time inference wrapper for HIT policies.
    
    This class provides utilities for running a HIT policy in real-time settings,
    handling preprocessing, inference, and postprocessing of inputs/outputs in a 
    sequential manner suitable for control applications.
    
    Args:
        policy: Trained HIT policy model
        norm_stats: Normalization statistics for inputs/outputs
        camera_names: List of camera names to use
        observation_names: List of observation field names to use
        chunk_size: Number of timesteps for action prediction
        width: Image width for preprocessing
        height: Image height for preprocessing
        device: Device to run inference on (cuda/cpu)
        action_delay: Number of steps to delay between receiving observation and returning action
        history_len: Length of history buffer for observations
    """
    def __init__(self, policy, norm_stats, camera_names, observation_names=[], 
                 chunk_size=8, width=224, height=224, device="cuda", 
                 action_delay=0, history_len=5):
        self.policy = policy
        self.policy.eval()  # Set policy to evaluation mode
        self.norm_stats = norm_stats
        self.camera_names = camera_names
        self.observation_names = observation_names
        self.chunk_size = chunk_size
        self.width = width
        self.height = height
        self.device = device
        self.action_delay = action_delay
        
        # Initialize buffers for sequential processing
        self.action_buffer = deque(maxlen=chunk_size)
        self.observation_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)
        
        # Image normalization for ResNet
        self.normalize_resnet = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Set up timer for performance monitoring
        self.inference_times = deque(maxlen=100)
        self.last_inference_time = None
        
        # Initialize model to device
        self.policy = self.policy.to(device)
        
    def preprocess_images(self, images):
        """
        Preprocesses images for model input.
        
        Args:
            images: Dictionary of images keyed by camera name
            
        Returns:
            Tensor of preprocessed images
        """
        all_cam_images = []
        for cam_name in self.camera_names:
            if cam_name in images:
                img = images[cam_name]
                # Resize if needed
                if self.width is not None and self.height is not None:
                    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
                all_cam_images.append(img)
            else:
                raise ValueError(f"Camera {cam_name} not found in input images")
                
        # Stack and convert to tensor
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images).float()
        
        # Convert to channel-first format and normalize
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        
        # Convert BGR to RGB
        image_data = image_data[:, [2, 1, 0], :, :]
        
        return image_data
    
    def preprocess_qpos(self, observations):
        """
        Preprocesses proprioceptive state for model input.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Tensor of preprocessed state
        """
        if len(self.observation_names) > 0:
            observation_data = []
            for name in self.observation_names:
                if name == 'imu_orn' and name in observations:
                    # Only take roll and pitch for IMU orientation
                    observation_data.append(observations[name][:2])
                elif name in observations:
                    observation_data.append(observations[name])
                else:
                    raise ValueError(f"Observation {name} not found in input observations")
            
            qpos = np.concatenate(observation_data, axis=-1)
        else:
            # If no observation names specified, use a default zero vector
            qpos = np.zeros(40)
            
        qpos_data = torch.from_numpy(qpos).float()
        
        # Normalize using dataset statistics
        qpos_data = (qpos_data - torch.from_numpy(self.norm_stats["qpos_mean"]).float()) / \
                    torch.from_numpy(self.norm_stats["qpos_std"]).float()
        
        return qpos_data
    
    def postprocess_action(self, action):
        """
        Postprocesses model output for execution.
        
        Args:
            action: Raw action from model
            
        Returns:
            Processed action ready for execution
        """
        # Denormalize
        action = action * torch.from_numpy(self.norm_stats["action_std"]).to(self.device) + \
                 torch.from_numpy(self.norm_stats["action_mean"]).to(self.device)
        
        return action.cpu().numpy()
    
    def step(self, images, observations):
        """
        Performs one step of inference with the policy.
        
        Args:
            images: Dictionary of images keyed by camera name
            observations: Dictionary of observations
            
        Returns:
            Action for the current timestep
        """
        start_time = time.time()
        
        # Preprocess inputs
        image_data = self.preprocess_images(images)
        qpos_data = self.preprocess_qpos(observations)
        
        # Move to device and add batch dimension
        image_data = image_data.unsqueeze(0).to(self.device)
        qpos_data = qpos_data.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            actions = self.policy.forward_inf(qpos_data, image_data)
            
        # Postprocess output
        actions = self.postprocess_action(actions)
        
        # Store in buffer
        self.action_buffer.append(actions[0])
        
        # Get action (with optional delay)
        if len(self.action_buffer) <= self.action_delay:
            # Not enough history yet, return zeros
            current_action = np.zeros_like(actions[0, 0])
        else:
            # Get delayed action
            current_action = self.action_buffer[0] if self.action_delay == 0 else list(self.action_buffer)[-(self.action_delay+1)]
        
        # Update observation and action history
        self.observation_history.append((images, observations))
        self.action_history.append(current_action)
        
        # Record inference time
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        self.last_inference_time = inference_time
        
        return current_action[0]  # Return first timestep of predicted action sequence
    
    def get_performance_stats(self):
        """
        Returns performance statistics for monitoring.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.inference_times:
            return {"mean_inference_time": 0, "max_inference_time": 0, "min_inference_time": 0}
        
        return {
            "mean_inference_time": np.mean(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "last_inference_time": self.last_inference_time,
            "fps": 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0
        }
    
    def reset(self):
        """
        Resets the model state between episodes.
        """
        self.action_buffer.clear()
        self.observation_history.clear()
        self.action_history.clear()


# Example usage
def load_inference_model(policy_path, norm_stats_path, camera_names, observation_names=[], device="cuda"):
    """
    Helper function to load a trained policy and create an inference model.
    
    Args:
        policy_path: Path to saved policy checkpoint
        norm_stats_path: Path to normalization statistics
        camera_names: List of camera names to use
        observation_names: List of observation names to use
        device: Device to run inference on
        
    Returns:
        Initialized RealtimeInferenceModel
    """
    import pickle
    from HIT.policy import HITPolicy
    
    # Load normalization statistics
    with open(norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Load policy
    checkpoint = torch.load(policy_path, map_location=device)
    policy_args = checkpoint['args']
    policy = HITPolicy(policy_args)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Create and return inference model
    inference_model = RealtimeInferenceModel(
        policy=policy,
        norm_stats=norm_stats,
        camera_names=camera_names,
        observation_names=observation_names,
        device=device
    )
    
    return inference_model
