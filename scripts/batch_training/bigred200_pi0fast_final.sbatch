#!/bin/bash
# Final paper-ready sweep on rollouts_all 2 (pi0-FAST Franka).
# All models, equal epochs (1000), 5 seeds, focused hyperparam grid.
#
# Submit with:
#   sbatch scripts/batch_training/bigred200_pi0fast_final.sbatch

#SBATCH -J safe_pi0fast_final
#SBATCH -A c02114
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 18:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# lmod's bash hooks reference PS1; with `set -u` that aborts the script.
export PS1=${PS1:-""}

export PROJ="/N/scratch/$USER/safe"
export SAFE_OPENPI_DROID_ROLLOUT_ROOT="${PROJ}/"
export ENV_PREFIX="/N/scratch/$USER/envs/vla-safe"

export PIP_CACHE_DIR="/N/scratch/$USER/.cache/pip"
export HF_HOME="/N/scratch/$USER/.cache/huggingface"
export WANDB_DIR="${PROJ}/wandb"
export WANDB_MODE=offline
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR"

module purge
module load conda/25.3.0
module load cudatoolkit/12.6
conda activate "$ENV_PREFIX"

echo "Host: $(hostname)"
nvidia-smi
echo "Data root: $SAFE_OPENPI_DROID_ROLLOUT_ROOT"

cd "$PROJ"
bash scripts/batch_training/submit_pi0fast_final.bash
