# Sample Results

A curated snapshot of figures produced by the SAFE experiments on Big Red 200.
Full per-run outputs (~1500 PNGs across both datasets and all architectures)
are not committed — only a representative subset.

## `sample_feat_vis/`

2D projections of the VLA hidden-state features (output of
`scripts/visualize_features.py`). Each `*_succ.png` colors points by
ground-truth success/failure; each `*_taskid.png` colors by task ID.

| File | Source |
|------|--------|
| `pi0fast_pca_*.png` | rollouts_all 2 (pi0-FAST Franka) — PCA |
| `pi0fast_tsne_*.png` | rollouts_all 2 — t-SNE |
| `pi0fast_umap_*.png` | rollouts_all 2 — UMAP |
| `openvla_pca_*.png` | OpenVLA WidowX — PCA |

## `sample_failure_curves/`

Per-rollout failure-score curves from a trained TCN model on OpenVLA WidowX
(output of `scripts/plot_failure_curves.py --per-rollout`).

| File | What it shows |
|------|---------------|
| `tcn_openvla_val_seen_overlay.png` | All val_seen rollouts on one axis, score vs. timestep, red=failed, blue=succeeded, dashed line = detection threshold |
| `tcn_openvla_val_unseen_overlay.png` | Same for val_unseen (held-out tasks — the test set) |
| `tcn_openvla_val_unseen_detection_time.png` | Histogram: when does the score first cross threshold? Failed rollouts (red) → true detections; successful rollouts (blue) → false alarms |
| `tcn_openvla_val_unseen_per_task.png` | Same overlay split into per-task panels |
| `tcn_final_val_unseen_task04_ep00*_*.png` | Individual rollouts: one PNG per rollout, vertical orange line = first detection |

## How these were generated

```bash
# On BR200, after the final sweeps finish
python scripts/extract_results.py --csv results.csv \
    --out logs/safe_pi0fast_final-<jobid>.out \
    --err logs/safe_pi0fast_final-<jobid>.err

python scripts/plot_failure_curves.py \
    --ckpt /N/scratch/sbdubey/safe/logs/<exp_name>/<date>/<time>/model_final.ckpt \
    --out-dir notebooks/failure_curves \
    --threshold 0.5 --per-rollout
```

See `BIGRED200_FEAT_VIS_SETUP.md` § 6 for the full workflow.
