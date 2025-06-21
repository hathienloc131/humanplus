"""
Main training script for the HIT (Humanoid Imitation with Transformers) framework.

This script implements behavioral cloning (BC) from human demonstrations to train
policies for controlling humanoid robots. It supports various policy architectures
and training configurations.

Key components:
- Data loading and preprocessing
- Model training and validation loops
- Checkpoint saving and loading
- Performance monitoring and logging with wandb

The script can be run from the command line with various arguments to configure
the training process, including task selection, policy architecture, and hyperparameters.
"""

import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import json
import wandb

from utils import compute_dict_mean, set_seed, load_data, load_lerobot_data  # data functions
from constants import TASK_CONFIGS
from model_util import make_policy, make_optimizer


def forward_pass(data, policy):
    """
    Perform a forward pass through the policy model.

    Args:
        data (tuple): A tuple containing (image_data, qpos_data, action_data, is_pad)
            - image_data: Visual observations from cameras
            - qpos_data: Proprioceptive state information
            - action_data: Ground truth actions for training
            - is_pad: Boolean mask indicating padded timesteps
        policy (nn.Module): The policy model to evaluate

    Returns:
        dict: The output from the policy, typically containing loss values and predictions
    """
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    print(f"==>> image_data.shape: {image_data.shape}")
    print(f"==>> action_data.shape: {action_data.shape}")
    print(f"==>> is_pad.shape: {is_pad.shape}")
    print(f"==>> qpos_data.shape: {qpos_data.shape}")
    
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    """
    Train a policy using behavioral cloning.

    This function handles the main training loop, including:
    - Model initialization
    - Optimizer setup
    - Training and validation iterations
    - Checkpoint saving
    - Logging metrics

    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        config (dict): Configuration parameters including:
            - num_steps: Total number of training steps
            - ckpt_dir: Directory to save checkpoints
            - seed: Random seed for reproducibility
            - policy_class: Type of policy to train
            - policy_config: Configuration for the policy
            - validate_every: Steps between validation runs
            - save_every: Steps between checkpoint saves
            - load_pretrain: Whether to load pretrained weights
            - pretrained_path: Path to pretrained weights
            - resume_ckpt_path: Path to resume training from

    Returns:
        tuple: (policy, train_stats, val_stats) - The trained policy and statistics
    """
    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config["load_pretrain"]:
        loading_status = policy.deserialize(
            torch.load(
                f'{config["pretrained_path"]}/policy_last.ckpt', map_location="cuda"
            )
        )
        print(f"loaded! {loading_status}")
    if config["resume_ckpt_path"] is not None:
        loading_status = policy.deserialize(torch.load(config["resume_ckpt_path"]))
        print(
            f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}'
        )
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    if config["load_pretrain"]:
        optimizer.load_state_dict(
            torch.load(
                f'{config["pretrained_path"]}/optimizer_last.ckpt', map_location="cuda"
            )
        )

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps + 1)):
        if step % validate_every == 0:
            print("validating")

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f"val_{k}"] = validation_summary.pop(k)
            if config["wandb"]:
                wandb.log(validation_summary, step=step)
            print(f"Val loss:   {epoch_val_loss:.5f}")
            summary_string = ""
            for k, v in validation_summary.items():
                summary_string += f"{k}: {v.item():.3f} "
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict["loss"]
        loss.backward()
        optimizer.step()
        if config["wandb"]:
            wandb.log(forward_dict, step=step)  # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.serialize(), ckpt_path)
            # save optimizer state
            optimizer_ckpt_path = os.path.join(
                ckpt_dir, f"optimizer_step_{step}_seed_{seed}.ckpt"
            )
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
        if step % 2000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
            torch.save(policy.serialize(), ckpt_path)
            optimizer_ckpt_path = os.path.join(ckpt_dir, f"optimizer_last.ckpt")
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.serialize(), ckpt_path)
    optimizer_ckpt_path = os.path.join(ckpt_dir, f"optimizer_last.ckpt")
    torch.save(optimizer.state_dict(), optimizer_ckpt_path)

    return best_ckpt_info


def repeater(data_loader):
    """
    Creates an infinite data generator from a data loader.
    
    This function wraps a PyTorch DataLoader to provide an infinite stream
    of data batches by repeatedly cycling through the dataset. It keeps track
    of epochs and prints a message when a complete pass through the dataset
    is finished.
    
    Args:
        data_loader: PyTorch DataLoader object
        
    Yields:
        Data batches from the data loader, cycling infinitely
    """
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f"Epoch {epoch} done")
        epoch += 1


def main_train(args):
    """
    Main training function that configures and runs the humanoid imitation training process.
    
    This function:
    1. Sets up training parameters from command line arguments
    2. Configures the policy model based on the specified policy class
    3. Prepares data loaders for training and validation
    4. Initializes logging and checkpoint directories
    5. Calls the training function and saves the best model checkpoint
    
    Args:
        args (dict): Dictionary containing command line arguments and configuration including:
            - eval (bool): Whether to run in evaluation mode
            - ckpt_dir (str): Directory to save checkpoints
            - policy_class (str): Type of policy to train (e.g., 'ACT', 'HIT')
            - task_name (str): Name of the task to train on
            - batch_size (int): Batch size for training and validation
            - num_steps (int): Total number of training steps
            - backbone (str): Neural network backbone to use (e.g., 'resnet18')
            - same_backbones (bool): Whether to use the same backbone for all cameras
            - lr (float): Learning rate
            - chunk_size (int): Size of action chunks for processing
            - hidden_dim (int): Dimension of hidden layers
            - data_aug (bool): Whether to use data augmentation
            - feature_loss_weight (float): Weight for feature loss
            - width/height (int): Image dimensions for processing
            
    Returns:
        None: Results are saved to the specified checkpoint directory
    """
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    eval_every = args["eval_every"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]
    backbone = args["backbone"]
    same_backbones = args["same_backbones"]

    ckpt_dir = f"{ckpt_dir}_{task_name}_{policy_class}_{backbone}_{same_backbones}"
    if os.path.isdir(ckpt_dir):
        print(f"ckpt_dir {ckpt_dir} already exists, exiting")
        return
    args["ckpt_dir"] = ckpt_dir
    # get task parameters
    is_sim = task_name[:4] == "sim_"

    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)
    randomize_index = task_config.get("randomize_index", False)

    print(f"===========================START===========================:")
    print(f"{task_name}")
    print(f"===========================Config===========================:")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"policy_class: {policy_class}")
    # fixed parameters
    state_dim = task_config.get("state_dim", 40)
    action_dim = task_config.get("action_dim", 40)
    state_mask = task_config.get("state_mask", np.ones(state_dim))
    action_mask = task_config.get("action_mask", np.ones(action_dim))
    if args["use_mask"]:
        state_dim = sum(state_mask)
        action_dim = sum(action_mask)
        state_idx = np.where(state_mask)[0].tolist()
        action_idx = np.where(action_mask)[0].tolist()
    else:
        state_idx = np.arange(state_dim).tolist()
        action_idx = np.arange(action_dim).tolist()
    lr_backbone = 1e-5
    backbone = args["backbone"]
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "same_backbones": args["same_backbones"],
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "vq": False,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "no_encoder": args["no_encoder"],
            "state_idx": state_idx,
            "action_idx": action_idx,
            "state_mask": state_mask,
            "action_mask": action_mask,
        }
    elif policy_class == "HIT":
        policy_config = {
            "lr": args["lr"],
            "hidden_dim": args["hidden_dim"],
            "dec_layers": args["dec_layers"],
            "nheads": args["nheads"],
            "num_queries": args["chunk_size"],
            "camera_names": camera_names,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "backbone": backbone,
            "same_backbones": args["same_backbones"],
            "lr_backbone": lr_backbone,
            "context_len": 183 + args["chunk_size"],  # for 224,400
            "num_queries": args["chunk_size"],
            "use_pos_embd_image": args["use_pos_embd_image"],
            "use_pos_embd_action": args["use_pos_embd_action"],
            "feature_loss": args["feature_loss_weight"] > 0,
            "feature_loss_weight": args["feature_loss_weight"],
            "self_attention": args["self_attention"] == 1,
            "state_idx": state_idx,
            "action_idx": action_idx,
            "state_mask": state_mask,
            "action_mask": action_mask,
        }
    else:
        raise NotImplementedError
    print(f"====================FINISH INIT POLICY========================:")
    config = {
        "num_steps": num_steps,
        "eval_every": eval_every,
        "validate_every": validate_every,
        "save_every": save_every,
        "ckpt_dir": ckpt_dir,
        "resume_ckpt_path": resume_ckpt_path,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": False,
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "load_pretrain": args["load_pretrain"],
        "height": args["height"],
        "width": args["width"],
        "normalize_resnet": args["normalize_resnet"],
        "wandb": args["wandb"],
        "pretrained_path": args["pretrained_path"],
        "randomize_data_degree": args["randomize_data_degree"],
        "randomize_data": args["randomize_data"],
    }
    all_configs = {**config, **args, "task_config": task_config}

    if args["width"] < 0:
        args["width"] = None
    if args["height"] < 0:
        args["height"] = None

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, "config.pkl")
    all_config_path = os.path.join(ckpt_dir, "all_configs.json")
    expr_name = ckpt_dir.split("/")[-1]
    if not is_eval and args["wandb"]:
        wandb.init(
            project=PROJECT_NAME, reinit=True, entity=WANDB_USERNAME, name=expr_name
        )
        wandb.config.update(config)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    with open(all_config_path, "w") as fp:
        json.dump(all_configs, fp, indent=4)

    # Load datasets
    if args.data_format == 'lerobot' and args.repo_id:
        print(f"Loading data from LeRobot format, repo_id: {args.repo_id}")
        train_dataloader, val_dataloader, norm_stats, is_sim = load_lerobot_data(
            args.repo_id,
            config['camera_names'],
            config['batch_size_train'],
            config['batch_size_val'],
            config['chunk_size'],
            args.policy_class,
            train_ratio=0.9,
            width=config.get('width'),
            height=config.get('height'),
            normalize_resnet=config.get('normalize_resnet', False),
            data_aug=config.get('data_aug', False),
            observation_name=config.get('observation_name', []),
            feature_loss=config.get('feature_loss', False),
            grayscale=config.get('grayscale', False),
            randomize_color=config.get('randomize_color', False),
            randomize_index=config.get('randomize_index'),
            randomize_data_degree=config.get('randomize_data_degree', 0),
            randomize_data=config.get('randomize_data', False)
        )
    else:
        print("Loading data from HDF5 format")
        train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
            config['dataset_dir'],
            config['name_filter'],
            config['camera_names'],
            config['batch_size_train'],
            config['batch_size_val'],
            config['chunk_size'],
            skip_mirrored_data=config.get('skip_mirrored_data', False),
            load_pretrain=config.get('load_pretrain', False),
            policy_class=args.policy_class,
            stats_dir_l=config.get('stats_dir_l', None),
            sample_weights=config.get('sample_weights', None),
            train_ratio=config.get('train_ratio', 0.99),
            width=config.get('width'),
            height=config.get('height'),
            normalize_resnet=config.get('normalize_resnet', False),
            data_aug=config.get('data_aug', False),
            observation_name=config.get('observation_name', []),
            feature_loss=config.get('feature_loss', False),
            grayscale=config.get('grayscale', False),
            randomize_color=config.get('randomize_color', False),
            randomize_index=config.get('randomize_index'),
            randomize_data_degree=config.get('randomize_data_degree', 0),
            randomize_data=config.get('randomize_data', False)
        )

    # Print sample data information
    for data_batch in train_dataloader:
        image_data, qpos_data, action_data, is_pad = data_batch
        print(f"Sample batch shapes:")
        print(f"- image_data: {image_data.shape}")
        print(f"- qpos_data: {qpos_data.shape}")
        print(f"- action_data: {action_data.shape}")
        print(f"- is_pad: {is_pad.shape}")
        break

    # Train the model
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}")
    if args["wandb"]:
        wandb.finish()


def main():
    """
    Main function for training HIT models. Handles argument parsing and training setup.
    """
    parser = argparse.ArgumentParser(description='HIT Training Script')
    parser.add_argument('--task', type=str, default='rearrange_objects', choices=list(TASK_CONFIGS.keys()), 
                        help='Task to train on')
    parser.add_argument('--policy_class', type=str, default='HIT', choices=['HIT', 'ACT'], 
                        help='Policy class to use')
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Whether to use wandb for logging')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed')
    parser.add_argument('--data_format', type=str, default='hdf5', choices=['hdf5', 'lerobot'],
                        help='Data format to use (hdf5 or lerobot)')
    parser.add_argument('--repo_id', type=str, default=None,
                        help='Repository ID for LeRobot datasets (required when data_format is lerobot)')
    args = parser.parse_args()

    task_config = TASK_CONFIGS[args.task]
    config = deepcopy(task_config['train_config'])
    config['policy_class'] = args.policy_class
    config['policy_config'] = task_config['policy_config']
    config['seed'] = args.seed
    set_seed(config['seed'])

    # Set up experiment directory
    exp_name = f"{args.task}_{args.policy_class}_{config['policy_config'].get('backbone', '')}_qpos_{config['seed']}"
    save_dir = os.path.expanduser(f"./rearrange_objects/{exp_name}/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config['ckpt_dir'] = save_dir
    print(f'Experiment directory: {save_dir}')

    if args.use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'HIT'),
            entity=config.get('wandb_entity', None),
            config=config,
            name=exp_name,
        )

    # Load datasets
    if args.data_format == 'lerobot' and args.repo_id:
        print(f"Loading data from LeRobot format, repo_id: {args.repo_id}")
        train_dataloader, val_dataloader, norm_stats, is_sim = load_lerobot_data(
            args.repo_id,
            config['camera_names'],
            config['batch_size_train'],
            config['batch_size_val'],
            config['chunk_size'],
            args.policy_class,
            train_ratio=0.9,
            width=config.get('width'),
            height=config.get('height'),
            normalize_resnet=config.get('normalize_resnet', False),
            data_aug=config.get('data_aug', False),
            observation_name=config.get('observation_name', []),
            feature_loss=config.get('feature_loss', False),
            grayscale=config.get('grayscale', False),
            randomize_color=config.get('randomize_color', False),
            randomize_index=config.get('randomize_index'),
            randomize_data_degree=config.get('randomize_data_degree', 0),
            randomize_data=config.get('randomize_data', False)
        )
    else:
        print("Loading data from HDF5 format")
        train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
            config['dataset_dir'],
            config['name_filter'],
            config['camera_names'],
            config['batch_size_train'],
            config['batch_size_val'],
            config['chunk_size'],
            skip_mirrored_data=config.get('skip_mirrored_data', False),
            load_pretrain=config.get('load_pretrain', False),
            policy_class=args.policy_class,
            stats_dir_l=config.get('stats_dir_l', None),
            sample_weights=config.get('sample_weights', None),
            train_ratio=config.get('train_ratio', 0.99),
            width=config.get('width'),
            height=config.get('height'),
            normalize_resnet=config.get('normalize_resnet', False),
            data_aug=config.get('data_aug', False),
            observation_name=config.get('observation_name', []),
            feature_loss=config.get('feature_loss', False),
            grayscale=config.get('grayscale', False),
            randomize_color=config.get('randomize_color', False),
            randomize_index=config.get('randomize_index'),
            randomize_data_degree=config.get('randomize_data_degree', 0),
            randomize_data=config.get('randomize_data', False)
        )

    # Print sample data information
    for data_batch in train_dataloader:
        image_data, qpos_data, action_data, is_pad = data_batch
        print(f"Sample batch shapes:")
        print(f"- image_data: {image_data.shape}")
        print(f"- qpos_data: {qpos_data.shape}")
        print(f"- action_data: {action_data.shape}")
        print(f"- is_pad: {is_pad.shape}")
        break

    # Train the model
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}")
    if args["wandb"]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", action="store", type=str, help="config file", required=False
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", default="h1_ckpt/0", action="store", type=str, help="ckpt_dir"
    )
    parser.add_argument(
        "--policy_class",
        default="ACT",
        action="store",
        type=str,
        help="policy_class, capitalize",
    )
    parser.add_argument(
        "--task_name",
        default="h1_fold_shirt_0420-0006",
        action="store",
        type=str,
        help="task_name",
    )

    parser.add_argument(
        "--batch_size", default=32, action="store", type=int, help="batch_size"
    )
    parser.add_argument("--seed", default=0, action="store", type=int, help="seed")
    parser.add_argument(
        "--num_steps", default=100000, action="store", type=int, help="num_steps"
    )
    parser.add_argument("--lr", default=1e-5, action="store", type=float, help="lr")
    parser.add_argument("--load_pretrain", action="store_true", default=False)
    parser.add_argument(
        "--pretrained_path",
        action="store",
        type=str,
        help="pretrained_path",
        required=False,
    )

    parser.add_argument(
        "--eval_every",
        action="store",
        type=int,
        default=100000,
        help="eval_every",
        required=False,
    )
    parser.add_argument(
        "--validate_every",
        action="store",
        type=int,
        default=1000,
        help="validate_every",
        required=False,
    )
    parser.add_argument(
        "--save_every",
        action="store",
        type=int,
        default=10000,
        help="save_every",
        required=False,
    )
    parser.add_argument(
        "--resume_ckpt_path",
        action="store",
        type=str,
        help="resume_ckpt_path",
        required=False,
    )
    parser.add_argument("--skip_mirrored_data", action="store_true")
    parser.add_argument(
        "--actuator_network_dir",
        action="store",
        type=str,
        help="actuator_network_dir",
        required=False,
    )
    parser.add_argument("--history_len", action="store", type=int)
    parser.add_argument("--future_len", action="store", type=int)
    parser.add_argument("--prediction_len", action="store", type=int)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--no_encoder", action="store_true")
    # dec_layers
    parser.add_argument(
        "--dec_layers", action="store", type=int, default=7, required=False
    )
    parser.add_argument("--nheads", action="store", type=int, default=8, required=False)
    parser.add_argument(
        "--use_pos_embd_image", action="store", type=int, default=0, required=False
    )
    parser.add_argument(
        "--use_pos_embd_action", action="store", type=int, default=0, required=False
    )

    # feature_loss_weight
    parser.add_argument(
        "--feature_loss_weight", action="store", type=float, default=0.0
    )
    # self_attention
    parser.add_argument("--self_attention", action="store", type=int, default=1)

    # for backbone
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--same_backbones", action="store_true")
    # use mask
    parser.add_argument("--use_mask", action="store_true")

    # for image
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument(
        "--normalize_resnet", action="store_true"
    )  ### not used - always normalize - in the model.forward
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--randomize_color", action="store_true")
    parser.add_argument("--randomize_data", action="store_true")
    parser.add_argument("--randomize_data_degree", action="store", type=int, default=3)

    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--model_type", type=str, default="HIT")
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    # torch.cuda.set_device(args.gpu_id)
    PROJECT_NAME = "H1"
    WANDB_USERNAME = "ha-thienloc131"
    main_train(vars(args))
