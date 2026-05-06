#!/bin/bash
# Generate feature-visualization plots (succ.png, taskid.png) for the
# "rollouts_all 2" pi0-FAST Franka dataset on Big Red 200.
#
# Runs ONE projector per job (set via env var PROJECTOR=pca|tsne|umap).
# Submit three jobs in parallel for full coverage:
#   sbatch --export=ALL,PROJECTOR=pca  scripts/batch_training/bigred200_feat_vis_rollouts_all_2.sbatch
#   sbatch --export=ALL,PROJECTOR=tsne scripts/batch_training/bigred200_feat_vis_rollouts_all_2.sbatch
#   sbatch --export=ALL,PROJECTOR=umap scripts/batch_training/bigred200_feat_vis_rollouts_all_2.sbatch
#
# Outputs land in $PROJ/notebooks/feat_vis/<exp_name>-<projector>/.

#SBATCH -J safe_featvis_ra2
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

PROJECTOR=${PROJECTOR:-pca}
echo "Projector: ${PROJECTOR}"

# ---- Paths (scratch-based; Slate not provisioned for this user yet) ----
export PROJ="/N/scratch/$USER/safe"
export SAFE_OPENPI_DROID_ROLLOUT_ROOT="${PROJ}/"
export ENV_PREFIX="/N/scratch/$USER/envs/vla-safe"

export PIP_CACHE_DIR="/N/scratch/$USER/.cache/pip"
export HF_HOME="/N/scratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# ---- Modules + env ----
module purge
module load cudatoolkit/12.6 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ---- Sanity prints ----
echo "Host: $(hostname)"
nvidia-smi
python -c "import torch; print('cuda=', torch.cuda.is_available(), 'dev=', torch.cuda.get_device_name(0))"
echo "Data root: $SAFE_OPENPI_DROID_ROLLOUT_ROOT"
ls "$SAFE_OPENPI_DROID_ROLLOUT_ROOT/rollouts_all 2" | head

# ---- Run one projector ----
cd "$PROJ"
mkdir -p notebooks/feat_vis

echo "=============================="
echo "Running feat_vis with projector=${PROJECTOR}"
echo "=============================="
python scripts/visualize_features.py \
    hydra.job.chdir=False \
    dataset=pizero_fast_droid_rollouts_all_2 \
    dataset.data_path_prefix="${SAFE_OPENPI_DROID_ROLLOUT_ROOT}" \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    projector=${PROJECTOR} \
    custom_cmap=True \
    save_video=False \
    train.exp_suffix=rolloutsall2

echo "feat_vis (${PROJECTOR}) finished."
ls -la "$PROJ/notebooks/feat_vis/" || true
