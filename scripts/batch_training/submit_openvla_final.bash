#!/bin/bash
# Final paper-ready sweep on the OpenVLA WidowX dataset.
# - All neural models train for n_epochs=1000 (n_epochs=200 for RND/logpZO)
# - 5 seeds (0-1-2-3-4)
# - Focused hyperparam grid
# - Single wandb group for clean comparison

GROUP_NAME=openvla_widowx_final
DATASET=openvla_widowx

#############################################################################
# 1. Recurrent + per-step MLP — LSTM, GRU, indep
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=indep,lstm,gru \
    model.batch_size=64 \
    model.n_epochs=400 \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.lr=3e-4,1e-3 \
    model.lambda_reg=1e-2,1e-1 \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final \
    train.eval_save_ckpt=True

#############################################################################
# 2. Causal Transformer (improved defaults)
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=transformer \
    model.batch_size=64 \
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
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=tcn \
    model.batch_size=64 \
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
# 4. Embed (cosine, euclid, mahala) — fast, n_epochs=1
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
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
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=embed \
    model.n_epochs=1 \
    model.distance=mahala \
    model.use_success_only=False \
    model.cumsum=True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_embed

#############################################################################
# 5. RND + logpZO baselines
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=rnd \
    train.roc_every=50 \
    model.batch_size=32 \
    model.use_success_only=False \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_chen

python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean \
    dataset.load_to_cuda=False \
    model=logpzo \
    train.roc_every=50 \
    model.batch_size=32 \
    model.forward_chunk_size=512 \
    model.use_success_only=False \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_chen

#############################################################################
# 6. Handcrafted precomputed metrics
#############################################################################
python -u -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=final_handcrafted
