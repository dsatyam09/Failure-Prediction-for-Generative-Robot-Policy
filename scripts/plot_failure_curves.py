#!/usr/bin/env python
"""
Plot per-rollout failure-detection score curves for a trained model.

Loads a saved checkpoint (requires `train.eval_save_ckpt=True` in training),
re-runs evaluation on val_seen and val_unseen, and produces:

  - per-rollout score curves (one PNG per split, all rollouts overlaid,
    colored red/blue by ground-truth fail/success)
  - aggregate detection-time histogram (when does score first cross threshold)
  - per-task panel grid (rollouts grouped by task)

Usage on BR200:
  python scripts/plot_failure_curves.py \
    --ckpt /N/scratch/sbdubey/safe/logs/<exp_name>/<date>/<time>/model_final.ckpt \
    --threshold 0.5 \
    --out-dir /N/scratch/sbdubey/safe/notebooks/failure_curves

If --ckpt is omitted, scans logs/ for the most recent run dir with a saved ckpt.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

# Make sure we can import failure_prob from the repo regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from failure_prob.conf import process_cfg  # noqa: E402
from failure_prob.data import load_rollouts, split_rollouts  # noqa: E402
from failure_prob.data.utils import RolloutDataset  # noqa: E402
from failure_prob.model import get_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def find_latest_ckpt(logs_root: str) -> str:
    """Find the most recent run dir under logs_root that contains model_final.ckpt."""
    candidates = glob.glob(os.path.join(logs_root, "**", "model_final.ckpt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No model_final.ckpt found under {logs_root}. Train with train.eval_save_ckpt=True first."
        )
    return max(candidates, key=os.path.getmtime)


def model_forward_split(model, rollouts, batch_size: int, device: str):
    """Return list[np.ndarray]: one (T,) score array per rollout."""
    ds = RolloutDataset(model.cfg, rollouts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_scores, all_lens = [], []
    with torch.no_grad():
        for batch in dl:
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch_dev)  # (B, T, 1)
            scores = out.squeeze(-1).cpu().numpy()
            lens = batch_dev["valid_masks"].sum(dim=-1).cpu().numpy()
            for i in range(scores.shape[0]):
                all_scores.append(scores[i, : int(lens[i])])
    return all_scores


def plot_overlay(rollouts, scores, threshold, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    for r, s in zip(rollouts, scores):
        T = len(s)
        x = np.arange(T)
        color = "tab:red" if r.episode_success == 0 else "tab:blue"
        ax.plot(x, s, color=color, alpha=0.35, linewidth=0.8)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0,
               label=f"detection threshold = {threshold}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("failure score")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    # Custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="tab:red", alpha=0.7, label="failed rollout"),
        Line2D([0], [0], color="tab:blue", alpha=0.7, label="successful rollout"),
        Line2D([0], [0], color="black", linestyle="--", label=f"threshold={threshold}"),
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_detection_time_hist(rollouts, scores, threshold, save_path, title):
    """For each rollout, find first timestep where score>threshold ('detection time').
    Plot histograms separately for success and fail rollouts.
    """
    success_times, fail_times, success_misses, fail_misses = [], [], 0, 0
    for r, s in zip(rollouts, scores):
        idx = np.argmax(s > threshold)
        detected = (s > threshold).any()
        rel_t = idx / max(len(s) - 1, 1)
        if r.episode_success:
            if detected:
                success_times.append(rel_t)
            else:
                success_misses += 1
        else:
            if detected:
                fail_times.append(rel_t)
            else:
                fail_misses += 1

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bins = np.linspace(0, 1, 21)
    if fail_times:
        ax.hist(fail_times, bins=bins, alpha=0.55, color="tab:red",
                label=f"failed → detected ({len(fail_times)})")
    if success_times:
        ax.hist(success_times, bins=bins, alpha=0.55, color="tab:blue",
                label=f"success → false-alarmed ({len(success_times)})")
    ax.set_xlabel("normalized detection time (0=start of rollout, 1=end)")
    ax.set_ylabel("count")
    ax.set_title(f"{title}\n"
                 f"missed: {fail_misses} fail / {success_misses} success")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_one_per_rollout(rollouts, scores, threshold, save_dir, prefix):
    """Save one PNG per rollout — useful for cherry-picking paper figures.
    Filename: <prefix>_task<id>_ep<idx>_<succ|fail>.png
    """
    os.makedirs(save_dir, exist_ok=True)
    for r, s in zip(rollouts, scores):
        T = len(s)
        x = np.arange(T)
        succ = "succ" if r.episode_success else "fail"
        color = "tab:red" if not r.episode_success else "tab:blue"
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
        ax.plot(x, s, color=color, linewidth=1.5)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8,
                   label=f"detection threshold = {threshold}")
        # First detection time
        idx = np.argmax(s > threshold) if (s > threshold).any() else None
        if idx is not None:
            ax.axvline(idx, color="darkorange", linestyle=":", linewidth=1.2,
                       label=f"first detection @ t={idx}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("failure score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"task {r.task_id}: {r.task_description[:60]}  |  ep {r.episode_idx}  |  GT: {succ.upper()}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fname = f"{prefix}_task{r.task_id:02d}_ep{r.episode_idx:03d}_{succ}.png"
        fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
        plt.close(fig)


def plot_per_task_grid(rollouts, scores, threshold, save_path, title):
    by_task = {}
    for r, s in zip(rollouts, scores):
        by_task.setdefault((r.task_id, r.task_description), []).append((r, s))
    n = len(by_task)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=150,
                             squeeze=False)
    for ax_idx, ((task_id, task_desc), pairs) in enumerate(by_task.items()):
        ax = axes[ax_idx // cols][ax_idx % cols]
        for r, s in pairs:
            color = "tab:red" if r.episode_success == 0 else "tab:blue"
            ax.plot(s, color=color, alpha=0.4, linewidth=0.8)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"task {task_id}: {task_desc[:40]}", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    # blank unused subplots
    for k in range(len(by_task), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="Path to a saved model_final.ckpt; if omitted, find latest under --logs-root")
    p.add_argument("--logs-root", default="/N/scratch/sbdubey/safe/logs",
                   help="Where to scan for model_final.ckpt if --ckpt is omitted")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Detection threshold for the failure score (default 0.5)")
    p.add_argument("--out-dir", default="/N/scratch/sbdubey/safe/notebooks/failure_curves",
                   help="Where to save plots")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--per-rollout", action="store_true",
                   help="Also save ONE PNG per rollout (for paper figure selection).")
    args = p.parse_args()

    ckpt_path = args.ckpt or find_latest_ckpt(args.logs_root)
    print(f"Loading checkpoint: {ckpt_path}")
    run_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config.yaml beside {ckpt_path}. Did you set eval_save_logs=True?")

    cfg = OmegaConf.load(cfg_path)
    # The saved config already has data_path resolved (process_cfg ran during
    # the original training). Clearing data_path_prefix prevents process_cfg
    # from prepending it a second time.
    cfg.dataset.data_path_prefix = None
    if cfg.dataset.get("data_path_unseen") is not None:
        # Same defensive clear in case unseen path is split out.
        pass
    cfg = process_cfg(cfg)

    print("Loading rollouts (this is slow on first run)…")
    all_rollouts = load_rollouts(cfg)
    rollouts_by_split = split_rollouts(cfg, all_rollouts)

    input_dim = rollouts_by_split["train"][0].hidden_states.shape[-1]
    model = get_model(cfg, input_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)
    arch = cfg.model.name
    suffix = cfg.train.exp_suffix

    for split in ("val_seen", "val_unseen"):
        rollouts = rollouts_by_split.get(split, [])
        if not rollouts:
            print(f"  (skipping empty split: {split})")
            continue
        print(f"  Scoring {len(rollouts)} rollouts in {split}…")
        scores = model_forward_split(model, rollouts, args.batch_size, device)

        base = f"{arch}_{suffix}_{split}"
        plot_overlay(rollouts, scores, args.threshold,
                     os.path.join(args.out_dir, f"{base}_overlay.png"),
                     title=f"{arch} | {split} | failure score over time")
        plot_detection_time_hist(rollouts, scores, args.threshold,
                                 os.path.join(args.out_dir, f"{base}_detection_time.png"),
                                 title=f"{arch} | {split} | first-detection time distribution")
        plot_per_task_grid(rollouts, scores, args.threshold,
                           os.path.join(args.out_dir, f"{base}_per_task.png"),
                           title=f"{arch} | {split} | score curves per task")
        if args.per_rollout:
            indiv_dir = os.path.join(args.out_dir, f"{base}_per_rollout")
            print(f"  Saving {len(rollouts)} per-rollout PNGs to {indiv_dir}")
            plot_one_per_rollout(rollouts, scores, args.threshold, indiv_dir, prefix=base)
    print(f"\nDone. Plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
