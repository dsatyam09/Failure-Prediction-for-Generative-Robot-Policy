#!/bin/bash

# New-architecture sweep for the 'rollouts_all 2' dataset. Runs alongside the
# original sweep. TRIMMED grid (target: ~25-30 hours total).
# Cap n_epochs=200; 3 seeds (0-1-2) instead of 5.
#
# Models: GRU, Causal Transformer (with focused ablation), TCN.

GROUP_NAME=pizero_fast_droid_rollouts_all_2_newarches
DATASET=pizero_fast_droid_rollouts_all_2

#############################################################################
# 1. GRU — same trimmed grid as LSTM in the main sweep
#    2 layers × 2 hidden × 2 lr = 8 configs × 3 seeds ≈ 7 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=gru \
    model.n_epochs=200 \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2 \
    train.seed=0-1-2 \
    train.exp_suffix=newarches_gru

#############################################################################
# 2. Causal Transformer — main grid
#    depth × width × lr × dropout = 2*2*2*2 = 16 configs × 3 seeds ≈ 13 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=transformer \
    model.n_epochs=200 \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.n_heads=4 \
    model.ff_mult=4 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2 \
    model.dropout=0.0,0.1 \
    train.seed=0-1-2 \
    train.exp_suffix=newarches_tx_main

#############################################################################
# 3. Transformer ablation: n_heads
#    Fixed: depth=2, hidden=128, lr=3e-4, dropout=0
#    Tests "does multi-head attention matter at this scale".
#    3 configs × 3 seeds ≈ 2.5 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=transformer \
    model.n_epochs=200 \
    model.n_layers=2 \
    model.hidden_dim=128 \
    model.n_heads=1,4,8 \
    model.ff_mult=4 \
    model.lr=3e-4 \
    model.lambda_reg=1e-2 \
    model.dropout=0.0 \
    train.seed=0-1-2 \
    train.exp_suffix=newarches_tx_abl_heads

#############################################################################
# 4. Transformer ablation: dropout
#    Fixed: depth=2, hidden=128, n_heads=4, lr=3e-4
#    Important for regularization on small data.
#    3 configs × 3 seeds ≈ 2.5 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=transformer \
    model.n_epochs=200 \
    model.n_layers=2 \
    model.hidden_dim=128 \
    model.n_heads=4 \
    model.ff_mult=4 \
    model.lr=3e-4 \
    model.lambda_reg=1e-2 \
    model.dropout=0.0,0.1,0.2 \
    train.seed=0-1-2 \
    train.exp_suffix=newarches_tx_abl_dropout

#############################################################################
# 5. TCN — depth × width × lr
#    2 depths × 2 widths × 2 lr = 8 configs × 3 seeds ≈ 7 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=tcn \
    model.n_epochs=200 \
    model.n_layers=3,4 \
    model.hidden_dim=64,128 \
    model.kernel_size=3 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2 \
    model.dropout=0.0 \
    train.seed=0-1-2 \
    train.exp_suffix=newarches_tcn
