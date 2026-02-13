#!/usr/bin/bash
#SBATCH --job-name=AtoB      
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # number of GPUs
#SBATCH --time=192:00:00              # time (HH:MM:SS)
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU-AI

# PROJECT ROOT = where AlphaToBeta.py lives
# -------------------------------
#SBATCH -D /storage/arnab/Aditya/AlphaMut-master/AlphaToBeta_synced

# Tell Slurm to write logs INSIDE this run dir
# -------------------------------------------------
#SBATCH --output=runs/run_%j/logs/%x_%j.out
#SBATCH --error=runs/run_%j/logs/%x_%j.err

# Explanation: SBATCH directives are read BEFORE runtime;
# therefore these MUST be relative to -D.

# Safety
set -euo pipefail

# starting a timer
start=$(date +%s%3N)

# -------- 1) Paths & dirs --------
# 1) Create a unique run directory for this job
# -------------------------------------------------
RUN_ROOT="runs"
RUN_ID="run_${SLURM_JOB_ID}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
# mkdir -p "$RUN_DIR"

# 2) Create run-specific subdirectories
# -------------------------------------------------
mkdir -p \
  "$RUN_DIR/logs" \
  "$RUN_DIR/saved_models"

cd "$RUN_DIR"

# 3) Modules & CUDA 
# -------------------------------------------------
module purge
module load cuda-12.4

# Helpful diagnostics
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] CUDA module: $(module list 2>&1 | sed -n 's/^ *\\(cuda[^ ]*\\).*$/\\1/p')"
nvidia-smi || true

# 4) Save run metadata
# -------------------------------------------------
cat > run.yaml <<EOF
run_id: $RUN_ID
created_at: $(date -Iseconds)
slurm_job_id: ${SLURM_JOB_ID:-none}
hostname: $(hostname)
conda_env: AlphaToBeta
cuda: $(module -t list 2>&1 | grep -E '^cuda[-/]' || echo none)
EOF


# 5) Conda env --------
# -------------------------------------------------
# We have this function defined in our bashrc to load mambaforge and envs

use-mambaforge() {
  # same as your login node
  source "$HOME/Aditya/mambaforge/etc/profile.d/conda.sh"
  conda activate "${1:-base}"
}
use-mambaforge AlphaToBeta

# 6) Runtime env for headless + local imports --------
# -------------------------------------------------
export MPLBACKEND=Agg                    # matplotlib headless
export PYTHONUNBUFFERED=1
PARENT_DIR="$(dirname "$PWD")"
export PYTHONPATH="$PARENT_DIR:${PYTHONPATH:-}"  # so 'import Helix_in_protein_with_neigh' works from run director
# dirnme(pwd) gets us one level up
# (Optional: quiet down NCCL if cluster fabric is quirky)
# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

# 7) Run --------
# -------------------------------------------------
echo "[$(date)] Starting AlphaToBeta.py ... at Run directory: $(pwd)"
# Use srun for proper Slurm accounting
# srun python ../../AlphaToBeta.py
srun python ../../AlphaToBetaWB.py
echo "[$(date)] Finished AlphaToBeta.py."

#printing the total time taken
end=$(date +%s%3N)
runtime=$((end - start))
sec=$((runtime / 1000))
ms=$((runtime % 1000))
echo "Total time taken: ${sec}.${ms}  seconds"

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


