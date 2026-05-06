#!/bin/bash
# New-architecture sweep on Big Red 200 — runs alongside the original sweep.
# Trains GRU + Causal Transformer (with full ablation passes) + TCN.
#
# Submit with:
#   sbatch scripts/batch_training/bigred200_pi0fast_newarches.sbatch

#SBATCH -J safe_newarches
#SBATCH -A c02114
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 2-00:00:00                   # gpu partition max is 2 days
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# lmod's bash hooks reference PS1; with `set -u` that aborts the script.
export PS1=${PS1:-""}

# ---- Paths (scratch-based; Slate not provisioned for this user yet) ----
export PROJ="/N/scratch/$USER/safe"
export SAFE_OPENPI_DROID_ROLLOUT_ROOT="${PROJ}/"
export ENV_PREFIX="/N/scratch/$USER/envs/vla-safe"

export PIP_CACHE_DIR="/N/scratch/$USER/.cache/pip"
export HF_HOME="/N/scratch/$USER/.cache/huggingface"
export WANDB_DIR="${PROJ}/wandb"
export WANDB_MODE=offline   # compute nodes lack outbound internet; sync later from login node
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR"

# ---- Modules + env ----
# IMPORTANT: cudatoolkit/12.6 is required — without it, torch's
# `_preload_cuda_deps()` does dlopen() on bundled NVIDIA libraries which can
# hang for hours on Lustre I/O.
module purge
module load conda/25.3.0
module load cudatoolkit/12.6
conda activate "$ENV_PREFIX"

# ---- Sanity prints ----
echo "Host: $(hostname)"
nvidia-smi
echo "Data root: $SAFE_OPENPI_DROID_ROLLOUT_ROOT"

# ---- Run the new-architectures multirun script ----
cd "$PROJ"
bash scripts/batch_training/submit_pi0fast_droid_newarches.bash
