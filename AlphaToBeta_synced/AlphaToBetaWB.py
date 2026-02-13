# AlphaToBeta.py
# This script sets up and trains a reinforcement learning agent using Proximal Policy Optimization (PPO) to evolve protein
# sequences towards a desired structure, specifically targeting alpha helices evolving into beta sheets. The reward function
# is modified in such a way that it also counts the frequecny of different amino acids in the environment of the mutated sequence

# Import torch and check CUDA availability
import torch
print("Built from source:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Import custom environment and other libraries
import Helix_in_protein_with_neigh # this import will make sure that we are importing the whole protein utils. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil
import gymnasium as gym
import yaml # for saving the hyperparameters
import wandb
from wandb.integration.sb3 import WandbCallback


## Importing stablebaseline's Proximal Policy Optimisation and other utilities
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import CallbackList

#-------------Defining training hyperparameters-------------#

# Multiplier for total episodes
number_of_epochs = 160
total_timestep_per_rollout=1024
total_timesteps = int(total_timestep_per_rollout)*number_of_epochs
# print(total_episodes)
gamma = 0.95  # Discount factor
# Create log dir
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

env_config={
    'file_containing_sequence_database':'../../csv_files/15_aa_alpha_helix_dataset_with_whole_protein_sequence_with_start_and_end_fixed.csv',
    'protein_length_limit':300,
    'folder_to_save_validation_files':'validation_structures',
    'reward_cutoff':30.0,
    'unique_path_to_give_for_file':'with_protein_run',
    'sequence_encoding_type':'esm',
    'secondary_structure_to_disrupt':'both',
    'maximum_number_of_allowed_mutations_per_episode':15,
    'validation':True,
    'use_proline':True,
    'use_plddt_in_reward':False,
    'distance_cutoff':6.0,
    'chain_id':'A',
    'total_episodes':total_timesteps
}

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": total_timesteps,
    "env_id": "ProteinEvolution",
}

# Create the environment
env = Helix_in_protein_with_neigh.ProteinEvolution( file_containing_sequence_database='../../csv_files/15_aa_alpha_helix_dataset_with_whole_protein_sequence_with_start_and_end_fixed.csv',
                                                    protein_length_limit=300,
                                                    folder_to_save_validation_files='validation_structures',
                                                    reward_cutoff=30.0,
                                                    unique_path_to_give_for_file='with_protein_run',
                                                    sequence_encoding_type='esm',
                                                    secondary_structure_to_disrupt='both',
                                                    maximum_number_of_allowed_mutations_per_episode=15,
                                                    validation=True,
                                                    use_proline=True,
                                                    use_plddt_in_reward=False,
                                                    distance_cutoff=6.0,
                                                    chain_id='A',
                                                    total_episodes=total_timesteps)


run = wandb.init(
    project="AlphaToBeta",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
# Wrap the environment with a Monitor for logging
env = Monitor(env, log_dir)
new_logger = configure(log_dir, [ "stdout", "log", "csv", "tensorboard"])

# Create the PPO model with specified parameters
model = PPO('MlpPolicy', env, verbose=1, gamma=0.95)
model.set_logger(new_logger)

# Callbacks
Wandb_callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="saved_models/AtoB_ppo_training_whole_protein_with_proline",
        verbose=2,
    )
# Create the callback list
callbacks = CallbackList([Wandb_callback]) # Add more if required

# Train the agent
model.learn(total_timesteps=total_timesteps,callback=callbacks, tb_log_name = f"_{total_timesteps}_run1")

with open("run.yaml", "a") as f:
    yaml.dump({
        "hyperparameters": {
            # "learning_rate": learning_rate,
            "gamma": gamma,
            "n_steps": total_timesteps,
            "batch_size": number_of_epochs,
            "reward_cutoff": env_config['reward_cutoff'],
            "maximum_number_of_allowed_mutations_per_episode": env_config['maximum_number_of_allowed_mutations_per_episode'],
            "use_proline": env_config['use_proline'],
            "distance_cutoff": env_config['distance_cutoff'],
            "validation": env_config['validation'],
            "secondary_structure_to_disrupt": env_config['secondary_structure_to_disrupt'],
            "sequence_encoding_type": env_config['sequence_encoding_type'],
        }
    }, f)

model.save("saved_models/AtoB_ppo_training_whole_protein_with_proline")
run.finish()