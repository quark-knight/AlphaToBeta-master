#!/usr/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --partition=GPU-AI      # force job to run on gpu2
#SBATCH --gres=gpu:1            # request 1 GPU (enough to run nvidia-smi)
#SBATCH --time=00:05:00         # auto-kill after 5 minutes
#SBATCH --output=gpu2_info.out  # save output to this file
# Run the GPU info command
nvidia-smi 

module purge
module load cuda-12.4

use-mambaforge() {
  # same as your login node
  source "$HOME/Aditya/mambaforge/etc/profile.d/conda.sh"
  conda activate "${1:-base}"
}
use-mambaforge AlphaToBeta

cd /storage/arnab/Aditya/AlphaToBeta_synced
export PYTHONPATH=$PWD
python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Avail:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
import Helix_in_protein
print("Local import OK")
PY

