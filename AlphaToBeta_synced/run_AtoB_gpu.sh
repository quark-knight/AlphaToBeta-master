#!/usr/bin/bash
#SBATCH --job-name=AtoB      
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # number of GPUs
#SBATCH --exclusive                   # exclusive node access
#SBATCH --time=192:00:00              # time (HH:MM:SS)
#SBATCH --ntasks-per-node=1
#SBATCH --partition=iiser
#SBATCH -D /storage/aditya/AlphaToBeta_synced
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out


set -euo pipefail

# -------- 1) Paths & dirs --------
# Work dir is set by SBATCH -D above. Keep a handle to the initial slurm files.
JOB_STDOUT="$(pwd)/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
JOB_STDERR="$(pwd)/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

cleanup() {
  mv -f "$JOB_STDOUT" "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true
  mv -f "$JOB_STDERR" "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" 2>/dev/null || true
}

trap cleanup EXIT


mkdir -p logs outputs saved_models  # Python also makes its own log_dir; this is separate

# -------- 2) Modules & CUDA --------
module purge
module load cuda-12.4

# Helpful diagnostics
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] CUDA module: $(module list 2>&1 | sed -n 's/^ *\\(cuda[^ ]*\\).*$/\\1/p')"
nvidia-smi || true


# -------- 3) Conda env --------
# We have this function defined in our bashrc to load mambaforge and envs

use-mambaforge() {
  # same as your login node
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
  conda activate "${1:-base}"
}
use-mambaforge AlphaToBeta

# -------- 4) Runtime env for headless + local imports --------
export MPLBACKEND=Agg                    # matplotlib headless
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}" # so 'import Helix_in_protein' works from same folder
# (Optional: quiet down NCCL if cluster fabric is quirky)
# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

# -------- 5) Run --------
# Going to work dir
cd /storage/aditya/AlphaMut-master/AlphaToBeta_synced

echo "[$(date)] Starting AlphaToBeta.py ..."
# Use srun for proper Slurm accounting
srun python AlphaToBeta.py
echo "[$(date)] Finished AlphaToBeta.py."




# SCRATCH_DIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
# mkdir -p "$SCRATCH_DIR"
# rsync -a --delete csv_files/ "$SCRATCH_DIR/data/"

# mkdir -p outputs
# rsync -a "$SCRATCH_DIR/outputs/" outputs/ 

# export PYTHONUNBUFFERED=1 


# python -u AlphaMut.py \
#   --data "$SCRATCH_DIR/data" \
#   --output "$SCRATCH_DIR/outputs"

# mkdir -p outputs
# rsync -a "$SCRATCH_DIR/outputs/" outputs/


