# SAFE on Big Red 200 — Complete Project Guide

End-to-end guide for running the SAFE failure-detection codebase on Indiana
University's Big Red 200 (BR200) cluster, including all the lessons learned
across multiple weeks of debugging. Designed so a fresh agent (Claude or human)
can pick up this project cold and continue.

If you're new here, read this top-to-bottom once. After that, jump into
**§ "Day-to-day workflow"** for the recurring task list.

---

## 1. What this project does

We train **failure detectors** for Vision-Language-Action (VLA) robot models.
The detectors take per-timestep features from a VLA's hidden state and predict
whether the rollout is going to fail. Two datasets:

| Dataset | VLA model | Robot | Size on disk |
|---------|-----------|-------|--------------|
| `rollouts_all 2/` | pi0-FAST | Franka | ~47 GB |
| `openvla_widowx/` | OpenVLA | WidowX | ~1.6 GB |

The headline metric is **Test AUC** = AUPRC on rollouts of *unseen* tasks (zero-shot generalization).

Eight failure-detection model families compared:
LSTM, GRU, MLP (called `indep`), causal Transformer, TCN, distance-based Embed
(cosine/euclid/mahala), RND, log-density (logpZO), plus precomputed handcrafted metrics.

---

## 2. Account / paths reference

| Item | Value |
|------|-------|
| BR200 username | `sbdubey` |
| BR200 SSH | `ssh sbdubey@bigred200.uits.iu.edu` (password + Duo MFA) |
| Slurm account | `c02114` |
| Project root on BR200 | `/N/scratch/sbdubey/safe/` |
| Conda env on BR200 | `/N/scratch/sbdubey/envs/vla-safe/` |
| Local repo on Mac | `/Users/satyam/Personal/Projects/safe/` |
| Wandb username | `sbdubey09` (or 09-org) |

**`/N/scratch` purges files after 30 days of no access.** For long-term storage,
move to `/N/slate/` once that allocation is provisioned.

---

## 3. Critical lessons (read these or you'll waste days)

### 3.1 PyTorch CUDA hang on import
On BR200 compute nodes, **`import torch` hangs indefinitely** (Lustre stalls
at `dlopen` of bundled NVIDIA `.so` libraries) unless you load the system
cudatoolkit module first. Always:
```bash
module purge
module load conda/25.3.0
module load cudatoolkit/12.6
conda activate /N/scratch/sbdubey/envs/vla-safe
```
The `cudatoolkit/12.6` line is *essential*; both sbatch files include it.

### 3.2 wandb compute-node network restriction
Compute nodes have **no outbound internet**. `wandb.init()` blocks forever
trying to reach `api.wandb.ai`. Always set:
```bash
export WANDB_MODE=offline
```
Runs save locally to `/N/scratch/sbdubey/safe/wandb/wandb/offline-run-*`.
Then sync from the **login node** (which has internet) afterwards:
```bash
wandb sync /N/scratch/sbdubey/safe/wandb/wandb/
```

### 3.3 lmod + `set -u` collision
The conda module's bash hooks reference `$PS1`, which is unset in non-interactive
sbatch shells, so `set -u` aborts the script. Every sbatch must have:
```bash
export PS1=${PS1:-""}
```
…before any `module load`.

### 3.4 Rsync of paths with spaces
Source path `"rollouts_all 2/"` over SSH gets re-split by the remote shell,
so the `2/` is dropped and data lands at `rollouts_all/` instead.
Either rename to remove the space OR escape twice:
```bash
rsync -avhP --partial \
    "/Users/satyam/Personal/Projects/safe/rollouts_all 2/" \
    sbdubey@bigred200.uits.iu.edu:'"/N/scratch/sbdubey/safe/rollouts_all 2/"'
```
We currently work around it with a post-hoc `mv`:
```bash
mv /N/scratch/sbdubey/safe/rollouts_all "/N/scratch/sbdubey/safe/rollouts_all 2"
```

### 3.5 Tqdm goes to stderr
Training prints config/results to stdout (which is **block-buffered to file**),
but tqdm progress bars go to stderr (line-buffered). If `tail -f *.out` shows
nothing, tail the `.err` file instead.

### 3.6 Slurm walltime affects backfill
Asking for `2-00:00:00` (2 days) locks you out of every smaller backfill window
in the queue, often pushing your start time 3-5 days out. The final sweeps
actually finish in ~9-13 hours, so request **`-t 18:00:00`** to fit more
backfill slots.

### 3.7 First-run training is slow due to GPU warmup
The first epoch shows ~6 sec/iter. Don't extrapolate — once cuDNN benchmarks
its kernels, subsequent epochs run at ~25 it/s (≈40 ms/epoch on this small
dataset). 200-epoch runs finish in **~7 seconds** of training each.

---

## 4. One-time setup (only do this once)

### 4.1 SSH access
1. Confirm BR200 entitlement:
   ```bash
   groups   # should include iu-entlmt-app-rt-bigred200-users
   ```
2. Confirm Slurm allocation:
   ```bash
   sacctmgr show associations user=$USER
   # should show Account=c02114
   ```

### 4.2 Conda env on scratch
```bash
ssh sbdubey@bigred200.uits.iu.edu
module load conda/25.3.0
conda init bash && source ~/.bashrc
conda create --prefix /N/scratch/$USER/envs/vla-safe python=3.10 -y
conda activate /N/scratch/$USER/envs/vla-safe
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas scipy pyyaml tqdm "imageio[ffmpeg]" hydra-core omegaconf scikit-learn opencv_python einops wandb plotly matplotlib natsort flask ml_dtypes umap-learn
```

### 4.3 Repo + package install
```bash
cd /N/scratch/$USER/safe
git clone https://github.com/vla-safe/SAFE.git .
pip install -e .
```

### 4.4 Wandb login (one-time, on login node)
```bash
wandb login --relogin    # paste API key from https://wandb.ai/authorize
```

### 4.5 Push local code that's not in the public repo
The local repo has files not in the public SAFE GitHub (custom dataset YAMLs,
new model implementations, etc.). Sync from Mac:
```bash
bash /Users/satyam/Personal/Projects/safe/sync_to_br200.sh
```

### 4.6 Dataset transfer
```bash
# rollouts_all 2 — pi0-FAST Franka, ~47 GB
rsync -avhP --partial \
    "/Users/satyam/Personal/Projects/safe/rollouts_all 2/" \
    sbdubey@bigred200.uits.iu.edu:"/N/scratch/sbdubey/safe/rollouts_all 2/"
# If it lands at .../rollouts_all/, mv it on BR200 to add the " 2" suffix.

# OpenVLA WidowX — ~1.6 GB
rsync -avhP --partial \
    /Users/satyam/Personal/Projects/safe/openvla_widowx/ \
    sbdubey@bigred200.uits.iu.edu:/N/scratch/sbdubey/safe/openvla_widowx/
```
Verify on BR200:
```bash
du -sh "/N/scratch/sbdubey/safe/rollouts_all 2"
du -sh /N/scratch/sbdubey/safe/openvla_widowx
```

---

## 5. Repository layout

```
safe/
├── failure_prob/                # main package (installed via pip -e .)
│   ├── conf/                    # Hydra configs
│   │   ├── config.yaml          # base
│   │   ├── feat_vis.yaml        # feature-visualization config
│   │   ├── dataset/             # per-dataset YAML (one per VLA)
│   │   ├── model/               # per-model YAML (lstm, gru, transformer, tcn, indep, embed, rnd, logpzo)
│   │   └── __init__.py          # registers structured configs (the source of truth for hyperparams)
│   ├── data/                    # dataset loaders (one .py per VLA)
│   ├── model/                   # model implementations
│   │   ├── lstm.py / gru.py / indep.py / transformer.py / tcn.py / embed.py / rnd.py / logpZO.py
│   │   └── base.py              # BaseModel + optimizer/scheduler setup
│   ├── utils/                   # routines, video, conformal, metrics, etc.
│   └── train.py                 # main training entrypoint
│
├── scripts/                     # standalone scripts
│   ├── visualize_features.py    # PCA/t-SNE/UMAP of pre_logits → succ.png + taskid.png
│   ├── plot_failure_curves.py   # per-rollout failure-score plots from a saved checkpoint
│   ├── extract_results.py       # parses SLURM .err logs into a paper-ready leaderboard
│   ├── get_wandb_metrics.py     # pulls metrics from wandb.ai (the original SAFE one)
│   └── batch_training/          # SLURM submission scripts
│       ├── bigred200_smoke_test.sbatch              # single LSTM config, gpu-debug
│       ├── bigred200_feat_vis_rollouts_all_2.sbatch # PCA/t-SNE/UMAP, gpu-debug
│       ├── bigred200_pi0fast_final.sbatch           # final paper sweep, rollouts_all 2
│       ├── bigred200_openvla_final.sbatch           # final paper sweep, OpenVLA WidowX
│       ├── submit_pi0fast_final.bash                # the multirun grid invoked by the .sbatch above
│       └── submit_openvla_final.bash                # ditto for OpenVLA
│
├── notebooks/                   # output figures (gitignored mostly)
│   ├── feat_vis/                # 2D feature projections per dataset
│   └── failure_curves/          # per-rollout score timelines
│
├── BIGRED200_FEAT_VIS_SETUP.md  # this file
├── BIGRED200_GUIDANCE.md        # generic BR200 onboarding (for new users)
├── sync_to_br200.sh             # rsync helper for Mac → BR200
└── setup_envs.bash              # local env vars (gitignored)
```

The dataset directories (`rollouts_all 2/`, `openvla_widowx/`) are gitignored —
get them via the [SAFE repo download links](https://github.com/vla-safe/SAFE)
or directly from the lab.

---

## 6. Day-to-day workflow

### 6.1 Submit the final paper sweeps (both datasets, in parallel)
```bash
cd /N/scratch/sbdubey/safe
sbatch scripts/batch_training/bigred200_pi0fast_final.sbatch
sbatch scripts/batch_training/bigred200_openvla_final.sbatch
squeue -u $USER
```
Each sweep:
- Trains LSTM/GRU/MLP/Transformer/TCN with 5 seeds, n_epochs=400 (or 1000)
- Plus Embed/RND/logpZO/handcrafted baselines
- Uses focused hyperparam grid informed by preliminary sweep
- Saves `model_final.ckpt` for every config (so `plot_failure_curves.py` can replay)
- Wall-clock ~9-13 hours on a free node

### 6.2 Monitor
```bash
squeue -u $USER                                              # is it running?
squeue --start -u $USER                                      # estimated start time
sacct -X -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode

# Live tqdm progress (loading + per-epoch loss)
tail -f /N/scratch/sbdubey/safe/logs/safe_pi0fast_final-<jobid>.err
```

### 6.3 After both sweeps finish
```bash
# 1. Sync wandb runs from BR200 login node (which has internet) to wandb.ai
module load conda/25.3.0
conda activate /N/scratch/sbdubey/envs/vla-safe
cd /N/scratch/sbdubey/safe
wandb sync wandb/wandb/

# 2. Extract per-architecture leaderboard from SLURM logs
python scripts/extract_results.py \
    --out logs/safe_pi0fast_final-<jobid>.out \
    --err logs/safe_pi0fast_final-<jobid>.err \
    --csv results_pi0fast_final.csv

python scripts/extract_results.py \
    --out logs/safe_openvla_final-<jobid>.out \
    --err logs/safe_openvla_final-<jobid>.err \
    --csv results_openvla_final.csv
```
The CSV has a `ckpt_path` column for picking specific runs to plot.

### 6.4 Generate per-rollout failure-detection plots
```bash
# Find best LSTM checkpoint from CSV
LSTM_CKPT=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next}
    $h["model_name"]=="lstm" && $h["ckpt_path"]!="" {
        print $h["auc_val_unseen"], $h["ckpt_path"]
    }' /N/scratch/sbdubey/safe/results_pi0fast_final.csv \
    | sort -k 1 -nr | head -1 | awk '{print $2}')
echo "Best LSTM ckpt: $LSTM_CKPT"

# Generate plots for that ckpt
python scripts/plot_failure_curves.py \
    --ckpt "$LSTM_CKPT" \
    --out-dir /N/scratch/sbdubey/safe/notebooks/failure_curves_lstm_pi0fast \
    --threshold 0.5 --per-rollout
```
For other architectures, swap `=="lstm"` to `=="gru"`, `=="transformer"`, etc.

### 6.5 Pull plots back to Mac
```bash
# from Mac
rsync -avz \
    sbdubey@bigred200.uits.iu.edu:/N/scratch/sbdubey/safe/notebooks/ \
    /Users/satyam/Personal/Projects/safe/notebooks/
```

---

## 7. SBatch file inventory

| File | Purpose | Partition | Walltime |
|------|---------|-----------|----------|
| `bigred200_smoke_test.sbatch` | One LSTM config, 50 epochs — verify the pipeline | `gpu-debug` | 1h |
| `bigred200_feat_vis_rollouts_all_2.sbatch` | PCA/t-SNE/UMAP feature plots (one projector per `--export PROJECTOR=`) | `gpu-debug` | 1h |
| `bigred200_pi0fast_final.sbatch` | Full paper sweep on rollouts_all 2 | `gpu` | 18h |
| `bigred200_openvla_final.sbatch` | Full paper sweep on OpenVLA WidowX | `gpu` | 18h |

Internally they all:
1. `module purge && module load conda/25.3.0 && module load cudatoolkit/12.6`
2. `conda activate /N/scratch/$USER/envs/vla-safe`
3. `export WANDB_MODE=offline`
4. `cd /N/scratch/$USER/safe`
5. Invoke the appropriate `submit_*.bash` (which has the multirun grid)

---

## 8. Failure-mode cheat sheet

| Symptom | Cause | Fix |
|---------|-------|-----|
| Job runs forever, `.err` shows only `Loading rollouts: 0/N` for >5 min | Torch CUDA dlopen hang on Lustre | `module load cudatoolkit/12.6` BEFORE python (already in sbatch) |
| Job fails with `wandb.init()` blocking | Compute node has no internet | Set `export WANDB_MODE=offline` (already in sbatch) |
| Sbatch dies immediately with `PS1: unbound variable` | `set -u` + lmod hooks | `export PS1=${PS1:-""}` before `module load` (already in sbatch) |
| `tail -f *.out` shows nothing for minutes | stdout block-buffered | Tail `*.err` instead (tqdm goes there) |
| Job pending for 3+ days with `(Priority)` reason | 2-day walltime locks out backfill | Cut walltime to 18h (already in latest sbatch) |
| Hydra error: `Could not find 'pizero_fast_droid_rollouts_all_2'` | Local-only dataset YAML not synced to BR200 | `bash sync_to_br200.sh` |
| Hydra training hangs indefinitely after config-print | Same as PS1 hang or CUDA hang | See above two |
| `dataset.data_path` doubled like `/N/scratch/.../N/scratch/...` | Saved config + `process_cfg` re-prepending | `cfg.dataset.data_path_prefix = None` before `process_cfg` (fixed in `plot_failure_curves.py`) |
| `IndexError: list index out of range` in `plot_failure_curves.py` | Dataset path was wrong → 0 rollouts loaded | Same fix as above |
| Rsync says it sent 47 GB but `du -sh` is missing | Path with `" 2"` got eaten by remote shell | `mv /N/scratch/sbdubey/safe/rollouts_all "/N/scratch/sbdubey/safe/rollouts_all 2"` |

---

## 9. Recovering after laptop sleep / SSH drop

SLURM jobs run on the cluster; they survive your laptop sleeping/closing.
Just SSH back in and check:
```bash
squeue -u $USER                          # still pending or running?
sacct -X -j <jobid> --format=JobID,State,Elapsed,ExitCode  # historical state
tail -100 /N/scratch/sbdubey/safe/logs/<jobname>-<jobid>.err
```

If your **laptop's rsync** dropped (e.g., your Mac slept mid-transfer), just
re-run the same `rsync -avhP --partial …` command — it picks up where it
left off.

---

## 10. Reference

- IU HPC quickstart: https://kb.iu.edu/d/aoku
- BR200 partitions: `gpu-debug` (1h, 2 nodes, fast queue), `gpu` (2-day, ~60 nodes), `general` (4-day, CPU only)
- Original SAFE paper: https://arxiv.org/abs/2506.09937
- SAFE repo: https://github.com/vla-safe/SAFE
- Companion BR200 onboarding for new users (less project-specific): `BIGRED200_GUIDANCE.md`
