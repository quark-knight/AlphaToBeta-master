#!/usr/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --partition=iiser
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gpu2_info.out

ls -l /usr/bin/bash
which bash

ls -l /usr/bin/env bash
which bash

echo "Running on host:"
hostname
echo "User: $USER"
echo "Home: $HOME"

# Load modules
module purge
module load cuda-12.4

use-mambaforge() {
  # same as my login node
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
  conda activate "${1:-base}"
}
use-mambaforge AlphaToBeta

# Source mambaforge (FIX PATH IF NEEDED)
# MAMBA=$HOME/mambaforge
# source "$MAMBA/etc/profile.d/conda.sh"

# Activate environment
# conda activate AlphaToBeta

# Verify executables
which python
python --version
which nvidia-smi
nvidia-smi

cd /storage/aditya/AlphaMut-master/AlphaToBeta_synced
export PYTHONPATH=$PWD

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import Helix_in_protein_with_neigh
print("Local import OK")
PY
