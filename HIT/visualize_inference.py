"""
Visualization utility for the RealtimeInferenceModel.

This script provides visualization tools for debugging and monitoring
the real-time inference process with HIT policies.
"""

import os
import numpy as np
import cv2
import torch
import argparse
import pickle
import time
from matplotlib import pyplot as plt
from collections import deque

from HIT.policy import HITPolicy
from HIT.realtime_inference import RealtimeInferenceModel

class InferenceVisualizer:
    """
    Visualizes the inputs and outputs of the real-time inference model.
    
    This class provides utilities for:
    - Displaying camera images with overlaid action predictions
    - Plotting action trajectories over time
    - Monitoring performance metrics
    
    Args:
        camera_names: List of camera names to visualize
        action_dim: Dimensionality of the action space
        history_len: Number of timesteps to keep in history
        window_name: Name of the visualization window
    """
    def __init__(self, camera_names, action_dim=40, history_len=100, window_name="HIT Inference"):
        self.camera_names = camera_names
        self.action_dim = action_dim
        self.history_len = history_len
        self.window_name = window_name
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Initialize buffers
        self.action_history = deque(maxlen=history_len)
        self.inference_times = deque(maxlen=history_len)
        
        # Initialize plot for actions
        self.action_indices = list(range(min(8, action_dim)))  # Show first few dimensions
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.lines = [self.ax.plot([], [], label=f"Dim {i}")[0] for i in self.action_indices]
        self.ax.set_xlim(0, history_len)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlabel("Timestep")
        self.ax.set_ylabel("Action Value")
        self.ax.set_title("Action Trajectory")
        self.ax.legend()
        self.fig.tight_layout()
        
    def update(self, images, action, inference_time):
        """
        Updates the visualization with the latest data.
        
        Args:
            images: Dictionary of camera images
            action: Current action vector
            inference_time: Time taken for inference
        """
        # Update histories
        self.action_history.append(action)
        self.inference_times.append(inference_time)
        
        # Create combined image
        combined_img = self.create_combined_image(images)
        
        # Add performance overlay
        fps = 1.0 / np.mean(self.inference_times) if len(self.inference_times) > 0 else 0
        cv2.putText(combined_img, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_img, f"Inference: {inference_time*1000:.2f}ms", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow(self.window_name, combined_img)
        
        # Update action plot
        self.update_action_plot()
        
        # Process any CV2 events
        key = cv2.waitKey(1)
        return key
    
    def create_combined_image(self, images):
        """
        Creates a combined image from multiple camera views.
        
        Args:
            images: Dictionary of camera images
            
        Returns:
            Combined image for display
        """
        processed_images = []
        
        for cam_name in self.camera_names:
            if cam_name in images:
                img = images[cam_name].copy()
                # Add camera name as text
                cv2.putText(img, cam_name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                processed_images.append(img)
        
        # Combine images horizontally
        if len(processed_images) > 0:
            # Resize to same height
            height = min(img.shape[0] for img in processed_images)
            resized_images = []
            for img in processed_images:
                aspect = img.shape[1] / img.shape[0]
                width = int(height * aspect)
                resized_images.append(cv2.resize(img, (width, height)))
            
            # Combine horizontally
            combined_img = np.hstack(resized_images)
        else:
            combined_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(combined_img, "No images available", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined_img
    
    def update_action_plot(self):
        """
        Updates the action trajectory plot.
        """
        if len(self.action_history) > 0:
            action_array = np.array(self.action_history)
            x = np.arange(len(action_array))
            
            for i, idx in enumerate(self.action_indices):
                if idx < action_array.shape[1]:
                    self.lines[i].set_data(x, action_array[:, idx])
            
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Save plot as image
            self.fig.canvas.draw()
            plot_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            # Show in separate window
            cv2.imshow("Action Trajectory", cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
    
    def close(self):
        """
        Closes all visualization windows.
        """
        cv2.destroyAllWindows()
        plt.close(self.fig)

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
    parser.add_argument('--video_source', type=str, default=None,
                        help='Video file or camera ID for live testing (e.g., 0 for webcam)')
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

def simulate_real_time_input(args, video_sources, frame_idx):
    """
    Gets input from video sources or simulates input if not available.
    """
    images = {}
    
    # Try to get images from video sources first
    if video_sources:
        for i, cam_name in enumerate(args.camera_names):
            if i < len(video_sources) and video_sources[i] is not None:
                ret, frame = video_sources[i].read()
                if ret:
                    images[cam_name] = frame
                    continue
    
    # Fall back to sample images if video sources fail or don't exist
    for cam_name in args.camera_names:
        if cam_name not in images and args.input_folder:
            image_path = os.path.join(args.input_folder, f'{cam_name}_{frame_idx%100:06d}.jpg')
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                images[cam_name] = img
    
    # Create dummy images for any missing cameras
    for cam_name in args.camera_names:
        if cam_name not in images:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"{cam_name} - Frame {frame_idx}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            images[cam_name] = img
    
    # Create dummy observations
    observations = {
        'qpos': np.random.randn(40),  # Example qpos with 40 dimensions
        'imu_orn': np.random.randn(3)  # Example IMU data with 3 dimensions
    }
    
    return images, observations

def setup_video_sources(args):
    """
    Sets up video sources based on arguments.
    """
    video_sources = []
    
    if args.video_source is not None:
        # Try to parse as camera ID (integer)
        try:
            camera_id = int(args.video_source)
            # Use the same camera for all views
            for _ in args.camera_names:
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print(f"Warning: Could not open camera {camera_id}")
                    cap = None
                video_sources.append(cap)
                # Increment camera ID for next camera
                camera_id += 1
        except ValueError:
            # Try to parse as video file
            for _ in args.camera_names:
                cap = cv2.VideoCapture(args.video_source)
                if not cap.isOpened():
                    print(f"Warning: Could not open video file {args.video_source}")
                    cap = None
                video_sources.append(cap)
    else:
        video_sources = [None] * len(args.camera_names)
    
    return video_sources

def main():
    args = parse_args()
    
    # Load the inference model
    print("Loading inference model...")
    inference_model = load_inference_model(args)
    print("Model loaded successfully")
    
    # Set up visualization
    visualizer = InferenceVisualizer(args.camera_names)
    
    # Set up video sources
    video_sources = setup_video_sources(args)
    
    # Simulate real-time inference
    frame_idx = 0
    target_fps = 30  # Target frames per second
    
    print(f"Starting real-time inference visualization (target: {target_fps} FPS)")
    inference_model.reset()
    
    try:
        while True:
            cycle_start = time.time()
            
            # Get input
            images, observations = simulate_real_time_input(args, video_sources, frame_idx)
            
            # Run inference
            inference_start = time.time()
            action = inference_model.step(images, observations)
            inference_time = time.time() - inference_start
            
            # Update visualization
            key = visualizer.update(images, action, inference_time)
            if key == 27:  # ESC key
                break
            
            # Simulate real-time by enforcing target FPS
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, 1.0/target_fps - cycle_time)
            time.sleep(sleep_time)
            
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("Visualization stopped by user")
        
    finally:
        # Clean up
        visualizer.close()
        for source in video_sources:
            if source is not None:
                source.release()
        
        # Print performance stats
        stats = inference_model.get_performance_stats()
        print("\nPerformance statistics:")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Average inference time: {stats['mean_inference_time']*1000:.2f}ms")

if __name__ == "__main__":
    main()
