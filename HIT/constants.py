"""
Constants and configurations for the HIT (Humanoid Imitation with Transformers) framework.

This module defines task-specific configurations including dataset paths, camera setups,
observation and action dimensions, and masking for different body parts in the action space.
Each task configuration specifies:
- dataset_dir: Path to the dataset directory
- camera_names: List of camera views used for observations
- observation_name: List of observation components to include
- state_dim: Dimension of the state space
- action_dim: Dimension of the action space
- state_mask: Binary mask for state components
- action_mask: Binary mask for action components (typically segmented by body parts)
"""

import os

### Task parameters
DATA_DIR = f'{os.path.dirname(os.path.realpath(__file__))}/data' 

TASK_CONFIGS = {
    'data_fold_clothes':{
        'dataset_dir': DATA_DIR + '/data_fold_clothes',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action','wrist'],
        'state_dim':35,
        'action_dim':40,
        'state_mask': [0]*11 + [1]*24,
        'action_mask': [0]*11 + [1]*8 + [0]*5 + [1]*16 #11 for leg, 8 for arm, 5 for imu, 16 for gripper 
    },
    
    'data_rearrange_objects':{
        'dataset_dir': DATA_DIR + '/data_rearrange_objects',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'], # imu_orn -> only 0,1
        'state_dim':33,
        'action_dim':40,
        'state_mask': [0]*11 + [1]*22,
        'action_mask': [0]*10 + [0] + [1]*8 + [0]*5 + [1]*16, #10 for leg, 1 for waist, 8 for arm, 5 for imu, 16 for gripper 
    },
    
    'data_two_robot_greeting':{
        'dataset_dir': DATA_DIR + '/data_two_robot_greeting',
        'num_episodes': 100,
        'episode_len': 1000,
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'], # imu_orn -> only 0,1
        'state_dim':33,
        'action_dim':40,
        'state_mask': [0]*10 + [1]+ [1]*22,
        'action_mask': [0]*10 + [1] + [1]*8 + [0]*5 + [1]*16, #10 for leg, 1 for waist, 8 for arm, 5 for imu, 16 for gripper 
    },
    
    'data_warehouse':{
        'dataset_dir': DATA_DIR + '/data_warehouse',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'], # imu_orn -> only 0,1
        'state_dim':33,
        'action_dim':40,
        'state_mask': [1]*11 + [1]*22,
        'action_mask': [1]*10 + [1] + [1]*8 + [0]*5 + [1]*16, #10 for leg, 1 for waist, 8 for arm, 5 for imu, 16 for gripper 
    },
}

### Simulation envs fixed constants
DT = 0.04
FPS = 25