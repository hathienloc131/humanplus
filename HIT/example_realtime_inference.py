"""
Example usage of the RealtimeInferenceModel for HIT policies.

This script demonstrates how to:
1. Load a trained HIT policy
2. Set up the real-time inference model
3. Run inference in a simulated real-time loop

You can use this as a starting point for integrating the model with actual hardware.
"""

import os
import numpy as np
import time
import cv2
import argparse
import pickle
import torch

from policy import HITPolicy
from realtime_inference import RealtimeInferenceModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, required=True, 
                        help='Path to the trained policy checkpoint')
    parser.add_argument('--norm_stats_path', type=str, required=True,
                        help='Path to the normalization statistics')
    parser.add_argument('--camera_names', type=str, nargs='+', default=['cam_high', 'cam_low'],
                        help='Names of cameras to use')
    parser.add_argument('--observation_names', type=str, nargs='+', default=['qpos'],
                        help='Names of observation fields to use')
    parser.add_argument('--width', type=int, default=224,
                        help='Width to resize images to')
    parser.add_argument('--height', type=int, default=224,
                        help='Height to resize images to')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--input_folder', type=str, default=None,
                        help='Folder with sample images for testing')
    return parser.parse_args()

def load_inference_model(args):
    """
    Loads a trained policy and creates a real-time inference model.
    """
    # Load normalization statistics
    with open(args.norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Load policy
    checkpoint = torch.load(args.policy_path, map_location=args.device)
    policy_args = checkpoint['args']
    policy = HITPolicy(policy_args)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Create inference model
    inference_model = RealtimeInferenceModel(
        policy=policy,
        norm_stats=norm_stats,
        camera_names=args.camera_names,
        observation_names=args.observation_names,
        width=args.width,
        height=args.height,
        device=args.device
    )
    
    return inference_model

def simulate_real_time_input(input_folder, camera_names, frame_idx):
    """
    Simulates real-time input by loading sample images and creating dummy observations.
    In a real application, this would be replaced with actual sensor readings.
    """
    images = {}
    for cam_name in camera_names:
        # In a real application, this would be replaced with camera capture
        image_path = os.path.join(input_folder, f'{cam_name}_{frame_idx:06d}.jpg')
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
        else:
            # Create a dummy image if no sample is available
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Frame {frame_idx}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        images[cam_name] = img
    
    # In a real application, this would be replaced with sensor readings
    observations = {
        'qpos': np.random.randn(40),  # Example qpos with 40 dimensions
        'imu_orn': np.random.randn(3)  # Example IMU data with 3 dimensions
    }
    
    return images, observations

def main():
    args = parse_args()
    
    # Load the inference model
    print("Loading inference model...")
    inference_model = load_inference_model(args)
    print("Model loaded successfully")
    
    # Simulate real-time inference
    frame_idx = 0
    target_fps = 30  # Target frames per second
    
    print(f"Starting real-time inference loop (target: {target_fps} FPS)")
    inference_model.reset()
    
    try:
        while True:
            cycle_start = time.time()
            
            # Get input (in a real application, this would be from sensors)
            images, observations = simulate_real_time_input(
                args.input_folder or '.', args.camera_names, frame_idx)
            
            # Run inference
            action = inference_model.step(images, observations)
            
            # Print results
            if frame_idx % 10 == 0:
                stats = inference_model.get_performance_stats()
                print(f"Frame {frame_idx}: Action[0:3] = {action[:3]}, "
                      f"FPS = {stats['fps']:.2f}, "
                      f"Inference time = {stats['last_inference_time']*1000:.2f}ms")
            
            # Simulate real-time by enforcing target FPS
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, 1.0/target_fps - cycle_time)
            time.sleep(sleep_time)
            
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("Inference loop stopped by user")
        
    finally:
        stats = inference_model.get_performance_stats()
        print("\nPerformance statistics:")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Average inference time: {stats['mean_inference_time']*1000:.2f}ms")
        print(f"Min inference time: {stats['min_inference_time']*1000:.2f}ms")
        print(f"Max inference time: {stats['max_inference_time']*1000:.2f}ms")

if __name__ == "__main__":
    main()
