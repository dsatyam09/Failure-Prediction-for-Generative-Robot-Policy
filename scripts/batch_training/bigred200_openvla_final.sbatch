#!/bin/bash
# Final paper-ready sweep on the OpenVLA WidowX dataset.
# All models, equal epochs, 5 seeds, focused hyperparam grid.
#
# Submit with:
#   sbatch scripts/batch_training/bigred200_openvla_final.sbatch

#SBATCH -J safe_openvla_final
#SBATCH -A c02114
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# lmod's bash hooks reference PS1; with `set -u` that aborts the script.
export PS1=${PS1:-""}

export PROJ="/N/scratch/$USER/safe"
export SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT="${PROJ}/"
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
echo "Data root: $SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT"
ls "$SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT/openvla_widowx" 2>/dev/null | head

cd "$PROJ"
bash scripts/batch_training/submit_openvla_final.bash
