
"""
Utility functions for creating and configuring policy models and optimizers.

This module provides factory methods to instantiate different policy architectures
and their corresponding optimizers based on the specified policy class and configuration.
Supported policy classes include ACT (Action Chunking Transformer) and HIT (Humanoid 
Imitation with Transformers).
"""

from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy, HITPolicy


def make_policy(policy_class, policy_config):
    """
    Factory function to create a policy model instance.
    
    Args:
        policy_class (str): The type of policy to create ('ACT', 'HIT', etc.)
        policy_config (dict): Configuration parameters for the policy
        
    Returns:
        nn.Module: An instance of the specified policy class
        
    Raises:
        NotImplementedError: If the specified policy class is not supported
    """
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'HIT':
        policy = HITPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    """
    Factory function to create an optimizer for a policy model.
    
    Args:
        policy_class (str): The type of policy ('ACT', 'HIT', etc.)
        policy (nn.Module): The policy model instance
        
    Returns:
        torch.optim.Optimizer: An optimizer instance configured for the policy
        
    Raises:
        NotImplementedError: If the specified policy class is not supported
    """
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'HIT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer