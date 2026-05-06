#!/bin/bash
# Final paper-ready sweep on the rollouts_all 2 dataset.
# - All neural models train for the same n_epochs=1000
# - 5 seeds (0-1-2-3-4) for confidence intervals
# - Focused hyperparam grid (informed by preliminary sweep)
# - Single wandb group for clean comparison
#
# Estimated wall-clock: ~16-20 hours

GROUP_NAME=pizero_fast_droid_rollouts_all_2_final
DATASET=pizero_fast_droid_rollouts_all_2

#############################################################################
# 1. Recurrent models — LSTM, GRU + per-step MLP (indep) baseline
#    16 configs × 5 seeds (inside) ≈ 4-5 hr per arch
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=indep,lstm,gru \
    model.n_epochs=400 \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2,1e-1 \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final \
    train.eval_save_ckpt=True

#############################################################################
# 2. Causal Transformer — improved defaults (AdamW + warmup, dropout=0.1)
#    16 configs × 5 seeds ≈ 4-5 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=transformer \
    model.n_epochs=400 \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.n_heads=4 \
    model.ff_mult=4 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2,1e-1 \
    model.dropout=0.1,0.2 \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final \
    train.eval_save_ckpt=True

#############################################################################
# 3. TCN
#    8 configs × 5 seeds ≈ 2-3 hr
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=tcn \
    model.n_epochs=400 \
    model.n_layers=3,4 \
    model.hidden_dim=64,128 \
    model.kernel_size=3 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2 \
    model.dropout=0.0,0.1 \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final \
    train.eval_save_ckpt=True

#############################################################################
# 4. Embed baselines (cosine, euclid, mahala) — n_epochs=1, very fast
#############################################################################
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
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_embed

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
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_embed

#############################################################################
# 5. RND + logpZO (Chen-style baselines), n_epochs=200 (their default)
#############################################################################
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
        train.seed=0-1-2-3-4 \
        train.exp_suffix=final_chen
done

#############################################################################
# 6. Handcrafted precomputed metrics
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_handcrafted
