import torch
print("Built from source:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import Helix_in_protein # this import will make sure that we are importing the whole protein utils. 
import stable_baselines3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil
import gymnasium as gym
import numpy as np

## Proximal Policy Optimisation
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

env = Helix_in_protein.ProteinEvolution(  file_containing_sequence_database='csv_files/15_aa_alpha_helix_dataset_with_whole_protein_sequence_with_start_and_end_fixed.csv',
                                          protein_length_limit=300,
                                          folder_to_save_validation_files='validation_structures',
                                          reward_cutoff=70.0,
                                          unique_path_to_give_for_file='with_protein_run',
                                          sequence_encoding_type='esm',
                                          maximum_number_of_allowed_mutations_per_episode=15)

# Create log dir
log_dir = "AtoB_ppo_training_log_whole_protein"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)
new_logger = configure(log_dir, [ "csv", "tensorboard"])
model = PPO('MlpPolicy', env, verbose=1, gamma=0.95)
model.set_logger(new_logger)
# Train the agent
N=1
model.learn(total_timesteps=int(14000)*N, tb_log_name = "_14000_run1")

model.save("saved_models/AtoB_ppo_training_whole_protein_with_proline_tp")