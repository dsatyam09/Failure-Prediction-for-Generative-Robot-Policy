#!/bin/bash
# Smoke test on Big Red 200: run ONE training config against the
# "rollouts_all 2" dataset to prove the data loader + pipeline work.
# Submit with:
#   sbatch scripts/batch_training/bigred200_smoke_test.sbatch
#
# Before submitting, edit the three values marked TODO below.

#SBATCH -J safe_smoke
#SBATCH -A c02114
#SBATCH -p gpu-debug
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
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
# hang for hours on Lustre I/O. Loading the system cudatoolkit makes the
# system CUDA libs available so torch finds them quickly.
module purge
module load conda/25.3.0
module load cudatoolkit/12.6
conda activate "$ENV_PREFIX"

# ---- Sanity prints ----
echo "Host: $(hostname)"
nvidia-smi
echo "Data root: $SAFE_OPENPI_DROID_ROLLOUT_ROOT"
ls "$SAFE_OPENPI_DROID_ROLLOUT_ROOT/rollouts_all 2" | head

# ---- Single training run: smallest LSTM config, 1 seed ----
cd "$PROJ"
python -u -m failure_prob.train \
    train.wandb_group_name=pizero_fast_droid_rollouts_all_2_smoke \
    dataset=pizero_fast_droid_rollouts_all_2 \
    dataset.data_path_prefix="${SAFE_OPENPI_DROID_ROLLOUT_ROOT}" \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=lstm \
    model.n_layers=1 \
    model.hidden_dim=128 \
    model.lr=3e-4 \
    model.lambda_reg=1e-2 \
    model.n_epochs=50 \
    train.seed=0 \
    train.exp_suffix=smoke

echo "Smoke test finished."
