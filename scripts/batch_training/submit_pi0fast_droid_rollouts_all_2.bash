#!/bin/bash

# Run experiments for Pi0FAST model on the 'rollouts_all 2' real-world Franka dataset.
# TRIMMED grid (target: ~20-25 hours total). Cap n_epochs=200 for neural models;
# 3 seeds (0-1-2) instead of 5; drop dominated lr/lambda values.
# Expects SAFE_OPENPI_DROID_ROLLOUT_ROOT to point at the directory that CONTAINS
# the "rollouts_all 2" folder (e.g. the repo root, with a trailing slash).

GROUP_NAME=pizero_fast_droid_rollouts_all_2
DATASET=pizero_fast_droid_rollouts_all_2

## LSTM and MLP — main neural baselines
## 2 models × 2 layers × 2 hidden × 2 lr = 16 configs × 3 seeds (inside) ≈ 13 hr
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=indep,lstm \
    model.n_epochs=200 \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2 \
    train.seed=0-1-2 \
    train.exp_suffix=rolloutsall2

## Embed (cosine + euclid) — fast (n_epochs=1)
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=embed \
    model.n_epochs=1 \
    model.distance=cosine,euclid \
    model.use_success_only=False \
    model.topk=10 \
    model.cumsum=True \
    train.seed=0-1-2 \
    train.exp_suffix=rolloutsall2_embed

## Embed (mahala) — fast (n_epochs=1)
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=embed \
    model.n_epochs=1 \
    model.distance=mahala \
    model.use_success_only=False \
    model.cumsum=True \
    train.seed=0-1-2 \
    train.exp_suffix=rolloutsall2_embed

## Embed (pca-kmeans) — small grid
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=embed \
    model.distance=pca_kmeans \
    model.pca_dim=64 \
    model.n_clusters=32 \
    model.use_success_only=False \
    model.cumsum=True \
    train.seed=0-1-2 \
    train.exp_suffix=rolloutsall2_embed

## RND + logpZO — capped at n_epochs=200 (their default already)
for MODEL in rnd logpzo; do
    python -u -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
        dataset.feat_name=pre_logits \
        dataset.token_idx_rel=mean \
        model=${MODEL} \
        model.use_success_only=False \
        train.seed=0-1-2 \
        train.exp_suffix=rolloutsall2_chen
done

## Handcrafted precomputed metrics — fast
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2 \
    train.exp_suffix=rolloutsall2_handcrafted
