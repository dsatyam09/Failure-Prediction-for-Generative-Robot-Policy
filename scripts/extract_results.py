#!/usr/bin/env python
"""
Extract failure-detection results from SLURM stdout/stderr logs.

The training script prints two things we need:
  - .out: full hydra config (yaml) at the start of each multirun config,
    plus per-seed "Running seed N" markers
  - .err: wandb "Run summary:" blocks per seed run, with the final metrics

We pair them in order (each multirun config → N seed runs in .err),
producing a per-run leaderboard.

Usage on BR200:
  python scripts/extract_results.py \
    --out /N/scratch/sbdubey/safe/logs/safe_newarches-6960747.out \
    --err /N/scratch/sbdubey/safe/logs/safe_newarches-6960747.err \
    --csv /N/scratch/sbdubey/safe/results_newarches.csv

If --out / --err omitted, scans /N/scratch/sbdubey/safe/logs/ for both
safe_newarches-* and safe_rollouts_all_2-* and combines them.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict


METRIC_KEYS = (
    "falert_early_prc_auc/model_train",
    "falert_early_prc_auc/model_val_seen",
    "falert_early_prc_auc/model_val_unseen",
)


def parse_out_configs(out_path):
    """Parse .out file. Yield list of dicts, one per (multirun config, seed) in order.

    Also captures "Saving model checkpoint to PATH" lines and pairs them with
    the most recent seed run (so each run dict gets a `ckpt_path`).
    """
    if not os.path.isfile(out_path):
        return []

    with open(out_path) as f:
        lines = f.readlines()

    runs = []  # list of dicts (one per seed run, in order)
    current_config = None  # currently active multirun config

    # Match lines inside the model: block. We're looking for top-level keys
    # like `name: lstm`, `n_layers: 2`, etc. Hydra dumps config in YAML form.
    # We track being inside `model:` block specifically.
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Start of a new multirun config: look for "dataset:" at column 0
        # (hydra dumps top-level keys at column 0 with nested keys indented).
        if line.startswith("dataset:"):
            # Reset; we'll fill by scanning forward.
            current_config = {
                "model_name": None,
                "n_layers": None,
                "hidden_dim": None,
                "n_heads": None,
                "ff_mult": None,
                "kernel_size": None,
                "distance": None,
                "topk": None,
                "pca_dim": None,
                "n_clusters": None,
                "lr": None,
                "lambda_reg": None,
                "dropout": None,
                "n_epochs": None,
                "cumsum": None,
                "exp_suffix": None,
                "wandb_group": None,
            }
            # Scan forward for model: and train: blocks.
            j = i + 1
            in_model = False
            in_train = False
            while j < len(lines):
                lj = lines[j].rstrip("\n")
                if lj.startswith("model:"):
                    in_model = True
                    in_train = False
                elif lj.startswith("train:"):
                    in_train = True
                    in_model = False
                elif lj and not lj.startswith(" ") and not lj.startswith("-"):
                    # New top-level section (e.g. "model:" finished). Stop scanning.
                    if in_model or in_train:
                        in_model = False
                        in_train = False
                    # Don't break — could be another section we don't care about
                    if lj.startswith("Loading rollouts"):
                        # Config dump done.
                        break

                if in_model and lj.startswith("  ") and ":" in lj:
                    k, _, v = lj.strip().partition(":")
                    v = v.strip()
                    if k in current_config:
                        current_config[k] = v if v != "" else None
                    if k == "name":
                        current_config["model_name"] = v
                if in_train and lj.startswith("  ") and ":" in lj:
                    k, _, v = lj.strip().partition(":")
                    v = v.strip()
                    if k == "exp_suffix":
                        current_config["exp_suffix"] = v
                    elif k == "wandb_group_name":
                        current_config["wandb_group"] = v
                j += 1
            i = j
            continue

        # "Running seed N" — emit a run record with current config + this seed.
        m = re.match(r"^Running seed (\d+)\s*$", line)
        if m and current_config is not None:
            r = dict(current_config)
            r["seed"] = m.group(1)
            r["ckpt_path"] = None
            runs.append(r)

        # "Saving model checkpoint to PATH" — attach to the most recent run
        m_ckpt = re.match(r"^Saving model checkpoint to\s+(.+\.ckpt)\s*$", line)
        if m_ckpt and runs:
            runs[-1]["ckpt_path"] = m_ckpt.group(1).strip()

        i += 1

    return runs


def parse_err_summaries(err_path):
    """Parse .err for `wandb: Run summary:` blocks. Return list of metric dicts in order."""
    if not os.path.isfile(err_path):
        return []

    with open(err_path) as f:
        content = f.read()

    # Each run summary block starts with "wandb: Run summary:" and ends roughly at
    # "wandb: You can sync this run" or another "wandb: Run summary:".
    blocks = []
    summary_re = re.compile(r"wandb: Run summary:\s*(.*?)(?=wandb: You can sync this run|\Z)", re.DOTALL)
    for m in summary_re.finditer(content):
        body = m.group(1)
        block = {}
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("wandb:"):
                continue
            line = line[len("wandb:"):].strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)  # split on last whitespace -> (key, value)
            if len(parts) != 2:
                continue
            key, val = parts
            try:
                fval = float(val)
            except ValueError:
                continue
            block[key.strip()] = fval
        # Only keep blocks that contain the headline metric
        if "falert_early_prc_auc/model_val_unseen" in block:
            blocks.append(block)
    return blocks


def fmt_metric(x, w=8):
    if x is None:
        return f"{'-':>{w}}"
    return f"{x:>{w}.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Path to a SLURM .out log")
    p.add_argument("--err", default=None, help="Path to a SLURM .err log (paired with --out)")
    p.add_argument("--logs-dir", default="/N/scratch/sbdubey/safe/logs",
                   help="If --out/--err omitted, scan this dir for safe_*-*.{out,err} pairs")
    p.add_argument("--csv", default=None, help="Optional path for full per-run CSV")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    # Build list of (out, err) pairs to process
    pairs = []
    if args.out and args.err:
        pairs.append((args.out, args.err))
    else:
        for out in sorted(glob.glob(os.path.join(args.logs_dir, "safe_*-*.out"))):
            err = out[:-4] + ".err"
            if os.path.isfile(err):
                pairs.append((out, err))
            else:
                print(f"  (skipping {os.path.basename(out)}: no matching .err)", file=sys.stderr)
    print(f"Processing {len(pairs)} (.out, .err) pair(s).")

    all_runs = []
    for out, err in pairs:
        cfgs = parse_out_configs(out)
        sums = parse_err_summaries(err)
        n = min(len(cfgs), len(sums))
        if len(cfgs) != len(sums):
            print(f"  warning: in {os.path.basename(out)} got {len(cfgs)} seed-runs from .out "
                  f"but {len(sums)} summary blocks from .err; pairing first {n}.", file=sys.stderr)
        for c, s in zip(cfgs[:n], sums[:n]):
            row = dict(c)
            for k in METRIC_KEYS:
                row[k.replace("falert_early_prc_auc/model_", "auc_")] = s.get(k)
            row["epoch"] = int(s.get("epoch", 0)) if s.get("epoch") is not None else None
            row["source_log"] = os.path.basename(out)
            all_runs.append(row)
    print(f"Parsed {len(all_runs)} per-seed runs total.\n")

    if not all_runs:
        print("Nothing parsed.")
        return

    # Per-architecture summary
    by_arch = defaultdict(list)
    for r in all_runs:
        if r["auc_val_unseen"] is None:
            continue
        by_arch[r["model_name"] or "?"].append(r)

    # ============================================================
    # AGGREGATE OVER SEEDS — find best hyperparam config per arch
    # by averaging val_unseen across all seeds, then reporting mean±std
    # ============================================================
    def hp_key(r):
        return tuple((k, r.get(k)) for k in (
            "n_layers", "hidden_dim", "n_heads", "ff_mult", "kernel_size",
            "distance", "topk", "pca_dim", "n_clusters",
            "lr", "lambda_reg", "dropout", "cumsum"))

    import statistics as st

    paper_rows = []
    for arch, runs in by_arch.items():
        # group by hyperparam config (excluding seed)
        by_hp = defaultdict(list)
        for r in runs:
            by_hp[hp_key(r)].append(r)
        # for each config, mean over seeds
        cfg_summaries = []
        for k, group in by_hp.items():
            n_seeds = len(group)
            if n_seeds < 2:
                continue
            mean_test = st.mean(g["auc_val_unseen"] for g in group)
            std_test = st.stdev(g["auc_val_unseen"] for g in group)
            mean_val = st.mean(g["auc_val_seen"] for g in group if g["auc_val_seen"] is not None)
            std_val = st.stdev(g["auc_val_seen"] for g in group if g["auc_val_seen"] is not None) if n_seeds >= 2 else 0
            mean_train = st.mean(g["auc_train"] for g in group if g["auc_train"] is not None)
            std_train = st.stdev(g["auc_train"] for g in group if g["auc_train"] is not None) if n_seeds >= 2 else 0
            cfg_summaries.append({
                "arch": arch, "n_seeds": n_seeds, "rep": group[0],
                "mean_test": mean_test, "std_test": std_test,
                "mean_val": mean_val, "std_val": std_val,
                "mean_train": mean_train, "std_train": std_train,
            })
        if not cfg_summaries:
            # fall back to best single run for arch with too few seeds
            best = max(runs, key=lambda x: x["auc_val_unseen"])
            paper_rows.append({
                "arch": arch, "n_seeds": 1, "rep": best,
                "mean_test": best["auc_val_unseen"], "std_test": 0.0,
                "mean_val": best["auc_val_seen"] or 0.0, "std_val": 0.0,
                "mean_train": best["auc_train"] or 0.0, "std_train": 0.0,
            })
        else:
            best_cfg = max(cfg_summaries, key=lambda x: x["mean_test"])
            paper_rows.append(best_cfg)
    paper_rows.sort(key=lambda x: x["mean_test"], reverse=True)

    print("=" * 110)
    print("PAPER-READY ABLATION TABLE — best hyperparam config per architecture (mean ± std over seeds)")
    print("Test = held-out unseen tasks (zero-shot generalization)")
    print("Val  = held-out rollouts of seen tasks")
    print("=" * 110)
    print(f"{'Architecture':<14} {'# Seeds':>8} "
          f"{'Train AUC (mean ± std)':>23} "
          f"{'Val AUC (mean ± std)':>22} "
          f"{'Test AUC (mean ± std)':>22}")
    print("-" * 110)
    for r in paper_rows:
        print(f"{r['arch']:<14} {r['n_seeds']:>8} "
              f"  {r['mean_train']:.4f} ± {r['std_train']:.4f}    "
              f"  {r['mean_val']:.4f} ± {r['std_val']:.4f}    "
              f"  {r['mean_test']:.4f} ± {r['std_test']:.4f}")

    print("\n" + "=" * 96)
    print("BEST INDIVIDUAL RUN PER ARCHITECTURE (single seed, useful for cherry-picking ckpts)")
    print("=" * 96)
    print(f"{'Architecture':<14} {'#runs':>6} "
          f"{'Best Test AUC':>14} {'Best Val AUC':>14} {'Best Train AUC':>14}")
    print("-" * 96)

    arch_rows = []
    for arch, runs in by_arch.items():
        best = max(runs, key=lambda x: x["auc_val_unseen"])
        n = len(runs)
        arch_rows.append({"arch": arch, "n": n, "best": best})
    arch_rows.sort(key=lambda x: x["best"]["auc_val_unseen"], reverse=True)
    for r in arch_rows:
        b = r["best"]
        print(f"{r['arch']:<14} {r['n']:>6} "
              f"{b['auc_val_unseen']:>14.4f} "
              f"{fmt_metric(b['auc_val_seen'], 14)} "
              f"{fmt_metric(b['auc_train'], 14)}")

    # Best config per architecture
    print("\n" + "=" * 96)
    print("BEST CONFIG PER ARCHITECTURE")
    print("=" * 96)
    for r in arch_rows:
        b = r["best"]
        hps = []
        for k in ("n_layers", "hidden_dim", "n_heads", "ff_mult", "kernel_size",
                  "distance", "topk", "pca_dim", "n_clusters",
                  "lr", "lambda_reg", "dropout", "cumsum"):
            if b.get(k) not in (None, "-"):
                hps.append(f"{k}={b[k]}")
        print(f"\n  [{r['arch']}]  val_unseen={b['auc_val_unseen']:.4f}  "
              f"val_seen={fmt_metric(b['auc_val_seen'])}  train={fmt_metric(b['auc_train'])}")
        print(f"    seed={b['seed']}  n_epochs={b['n_epochs']}  exp_suffix={b['exp_suffix']}")
        print(f"    {'  '.join(hps)}")

    # Top-K runs overall
    sorted_all = sorted([r for r in all_runs if r["auc_val_unseen"] is not None],
                        key=lambda x: x["auc_val_unseen"], reverse=True)[: args.top_k]
    print("\n" + "=" * 96)
    print(f"TOP {args.top_k} RUNS OVERALL (by val_unseen AUC)")
    print("=" * 96)
    print(f"{'rank':>4} {'arch':<12} {'val_unseen':>10} {'val_seen':>9} {'train':>7} {'seed':>5} {'exp_suffix':<32}")
    print("-" * 96)
    for i, r in enumerate(sorted_all, 1):
        print(f"{i:>4} {(r['model_name'] or '?'):<12} {r['auc_val_unseen']:>10.4f} "
              f"{fmt_metric(r['auc_val_seen'], 9)} {fmt_metric(r['auc_train'], 7)} "
              f"{str(r['seed']):>5} {str(r['exp_suffix'] or '')[:32]:<32}")

    if args.csv:
        keys = list(all_runs[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_runs)
        print(f"\nWrote per-run CSV → {args.csv}")


if __name__ == "__main__":
    main()
