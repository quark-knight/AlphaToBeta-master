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


## Importing stablebaseline's Proximal Policy Optimisation and other utilities
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

#-------------Defining training hyperparameters-------------#

# Multiplier for total episodes
multiplier_for_episodes = 1
total_episodes = int(14000)*multiplier_for_episodes
# Create log dir
log_dir = "AtoB_ppo_training_log_whole_protein"
os.makedirs(log_dir, exist_ok=True)


# Create the environment
env = Helix_in_protein_with_neigh.ProteinEvolution( file_containing_sequence_database='csv_files/15_aa_alpha_helix_dataset_with_whole_protein_sequence_with_start_and_end_fixed.csv',
                                                    protein_length_limit=300,
                                                    folder_to_save_validation_files='validation_structures',
                                                    reward_cutoff=70.0,
                                                    unique_path_to_give_for_file='with_protein_run',
                                                    sequence_encoding_type='esm',
                                                    secondary_structure_to_disrupt='both',
                                                    maximum_number_of_allowed_mutations_per_episode=15,
                                                    use_proline=True,
                                                    distance_cutoff=6.0,
                                                    chain_id='A',
                                                    total_episodes=total_episodes)


# Wrap the environment with a Monitor for logging
env = Monitor(env, log_dir)
new_logger = configure(log_dir, [ "csv", "tensorboard"])

# Create the PPO model with specified parameters
model = PPO('MlpPolicy', env, verbose=1, gamma=0.95)
model.set_logger(new_logger)

# Train the agent
model.learn(total_timesteps=total_episodes, tb_log_name = f"_{total_episodes}_run1")
model.save("saved_models/AtoB_ppo_training_whole_protein_with_proline_tp")