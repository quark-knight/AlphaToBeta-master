#!/usr/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --partition=iiser       # force job to run on gpu2
#SBATCH --gres=gpu:1            # request 1 GPU (enough to run nvidia-smi)
#SBATCH --time=00:05:00         # auto-kill after 5 minutes
#SBATCH --output=gpu2_info.out  # save output to this file

# Enable strict error handling, Stops on errors, typos, hidden failures
set -euo pipefail
#-e — Exit immediately on error
#-u — Treat unset variables as an error
#-o pipefail — Return exit status of the last command in the pipe that failed

module purge
module load cuda-12.4

echo "Running on host:" 
hostname
echo "User: $USER"
echo "Home: $HOME"

use-mambaforge() {
  # same as my login node
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
  conda activate "${1:-base}"
}
use-mambaforge AlphaToBeta

# Verify executables
which python
python --version
# Run the GPU info command
which nvidia-smi
nvidia-smi                      # uncomment to see GPU status

cd /storage/aditya/AlphaMut-master/AlphaToBeta_synced
export PYTHONPATH=$PWD
python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Avail:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
import Helix_in_protein_with_neigh
print("Local import OK")
PY

