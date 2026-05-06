# Architectural Ablation for VLA Failure Detection — Building on SAFE

A course-project extension of the **SAFE** failure-detection framework
(Gu et al., NeurIPS 2025) that asks: *which sequence-modeling architecture
is best for detecting failures of Vision-Language-Action (VLA) robot policies
on a small dataset?*

We extend the original SAFE codebase with three additional architectures
(GRU, causal Transformer, Temporal Convolutional Network), an end-to-end
training/evaluation pipeline tuned for Indiana University's Big Red 200
HPC cluster, paper-ready ablation tables, and per-rollout failure-detection
visualizations.

> ⚠️ **All credit for the underlying methodology, data collection, and the
> codebase this work builds on goes to the SAFE authors.** We did not
> design SAFE; we adapted their public release and added baseline architectures
> + reproducibility tooling for a class project. Their paper, repository, and
> dataset releases are cited in detail in the **Acknowledgements** section
> below — please cite them, not us.

![Splash Figure (from the original SAFE paper)](assets/safe-teaser-static.png)

---

## Table of contents

1. [What's new in this fork](#whats-new-in-this-fork)
2. [Hardware & environment requirements](#hardware--environment-requirements)
3. [Quick-start (high level)](#quick-start-high-level)
4. [Step-by-step reproduction](#step-by-step-reproduction)
5. [Repository layout](#repository-layout)
6. [Sample results](#sample-results)
7. [Acknowledgements & full credit](#acknowledgements--full-credit)
8. [Citation](#citation)
9. [License](#license)

---

## What's new in this fork

Everything original to SAFE is unchanged in spirit; we **added** the following on top:

| Addition | File(s) | Purpose |
|----------|---------|---------|
| **GRU detector** | `failure_prob/model/gru.py`, `failure_prob/conf/model/gru.yaml` | Lighter cousin of LSTM, drop-in for the existing recurrent slot |
| **Causal Transformer** | `failure_prob/model/transformer.py`, `failure_prob/conf/model/transformer.yaml` | Self-attention with causal mask, AdamW + LR warmup |
| **TCN (Temporal Conv Net)** | `failure_prob/model/tcn.py`, `failure_prob/conf/model/tcn.yaml` | Dilated causal-convolution alternative to RNNs |
| **Final paper sweep scripts** | `scripts/batch_training/{bigred200_pi0fast_final.sbatch, bigred200_openvla_final.sbatch, submit_*_final.bash}` | Parallel-on-cluster sweeps with focused hyperparam grids and 5 seeds |
| **Result extractor** | `scripts/extract_results.py` | Parses SLURM `.out`/`.err` logs into a paper-ready leaderboard with mean ± std across seeds; emits a per-run CSV |
| **Per-rollout failure-curve plots** | `scripts/plot_failure_curves.py` | Loads any saved checkpoint and renders score-over-time plots (overlay, per-task panels, detection-time histogram, one PNG per rollout) |
| **Cluster docs** | `BIGRED200_FEAT_VIS_SETUP.md`, `BIGRED200_GUIDANCE.md` | End-to-end "how to run this on Big Red 200" + general cluster onboarding |
| **Dataset structure docs** | `docs/DATASET_LAYOUT.md` | What the on-disk dataset directories should look like |
| **Cluster sync helper** | `sync_to_br200.sh` | rsync wrapper that excludes large dirs |
| **Updated training & data utilities** | `failure_prob/conf/__init__.py`, `failure_prob/data/utils.py`, `failure_prob/train.py` | Register new model configs (GRU/Transformer/TCN), small data-loading fixes |

The original SAFE failure-detection method, dataset format, evaluation
metrics, and training loop are **unchanged** — we only swapped in additional
detector architectures alongside the existing LSTM / MLP / KNN-style /
RND / log-density / handcrafted baselines.

---

## Hardware & environment requirements

This project uses non-trivial compute. The course-project results in `results/`
were generated on:

| Resource | Spec |
|----------|------|
| GPU | 1× NVIDIA A100-SXM4-40GB |
| Node | Big Red 200 GPU compute node (SLURM-managed) |
| RAM | 96–200 GB |
| Storage | `/N/scratch/$USER` Lustre filesystem (~50 GB free) |
| Walltime per sweep | ~9–13 hours (one full architecture grid × 5 seeds) |
| OS / scheduler | SUSE Linux + SLURM 23.x |
| Python | 3.10 |
| PyTorch | 2.x with CUDA 12.8 wheels |
| CUDA toolkit module | `cudatoolkit/12.6` (system module — required to avoid Lustre-induced `dlopen` hang) |

**It's fine if you can't run it yourself** — the data, model checkpoints, and
result CSVs are produced offline on a research cluster that requires an IU
account. Below we describe exactly what would be needed if you wanted to.

A single-GPU desktop (A100/RTX 3090/4090, 24+ GB VRAM, 64 GB RAM, ~50 GB disk)
should also work for the OpenVLA WidowX dataset (~1.6 GB). The pi0-FAST Franka
dataset (~47 GB) needs more storage and is slower to load.

---

## Quick-start (high level)

```bash
# 1. Clone
git clone <this-repo-url>
cd safe

# 2. Create environment
conda create --prefix ./envs/vla-safe python=3.10 -y
conda activate ./envs/vla-safe
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas scipy pyyaml tqdm "imageio[ffmpeg]" hydra-core omegaconf scikit-learn opencv_python einops wandb plotly matplotlib natsort flask ml_dtypes umap-learn
pip install -e .

# 3. Get the datasets (see docs/DATASET_LAYOUT.md for expected on-disk layout)
#    rollouts_all 2 (~47 GB pi0-FAST Franka):
#      https://drive.google.com/file/d/13z_cdwnaJota2iHkZbhYgVALujZwtM3b/view?usp=sharing
#    openvla_widowx (~1.6 GB OpenVLA WidowX):
#      https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing
#    Unzip both at the repo root.

# 4. Configure data paths
cp setup_envs.bash.template setup_envs.bash
# Edit setup_envs.bash to point at the datasets you just unzipped.

# 5. (On a workstation, no SLURM) Run a single training config to verify:
source setup_envs.bash
export WANDB_MODE=offline   # or `wandb login` if you want online runs
python -u -m failure_prob.train \
    dataset=pizero_fast_droid_rollouts_all_2 \
    dataset.data_path_prefix="${SAFE_OPENPI_DROID_ROLLOUT_ROOT}" \
    dataset.feat_name=pre_logits dataset.token_idx_rel=mean dataset.load_to_cuda=False \
    model=lstm model.n_epochs=200 train.seed=0 train.exp_suffix=quicktest

# 6. (Optional) Run the full paper sweeps and produce ablation tables.
#    See "Step-by-step reproduction" below for SLURM-cluster instructions.
```

---

## Step-by-step reproduction

For full reproduction of the final ablation table + per-rollout figures,
we used Indiana University's Big Red 200. The complete walkthrough (with
all the cluster-specific gotchas we hit and fixed) is in
**[`BIGRED200_FEAT_VIS_SETUP.md`](BIGRED200_FEAT_VIS_SETUP.md)**.

The condensed version:

```bash
# --- ON A SLURM-MANAGED HPC NODE ---

# 1. Set up env (one-time, see BIGRED200_FEAT_VIS_SETUP.md §4)
module purge
module load conda/25.3.0
module load cudatoolkit/12.6      # IMPORTANT — without this, torch hangs on Lustre
conda activate /path/to/envs/vla-safe

# 2. Submit both final sweeps in parallel
cd /path/to/safe
sbatch scripts/batch_training/bigred200_pi0fast_final.sbatch
sbatch scripts/batch_training/bigred200_openvla_final.sbatch

# 3. Monitor (each runs ~9-13 hours)
squeue -u $USER
sacct -X -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode

# --- AFTER BOTH JOBS COMPLETE ---

# 4. Sync wandb runs from a node that has internet (wandb is offline on compute nodes)
wandb sync wandb/wandb/

# 5. Build per-architecture leaderboards (mean ± std over 5 seeds, paper-style)
python scripts/extract_results.py \
    --out logs/safe_pi0fast_final-<jobid>.out \
    --err logs/safe_pi0fast_final-<jobid>.err \
    --csv results_pi0fast_final.csv

python scripts/extract_results.py \
    --out logs/safe_openvla_final-<jobid>.out \
    --err logs/safe_openvla_final-<jobid>.err \
    --csv results_openvla_final.csv

# 6. Generate per-rollout failure-detection plots for the best LSTM
LSTM_CKPT=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next}
    $h["model_name"]=="lstm" && $h["ckpt_path"]!="" {
        print $h["auc_val_unseen"], $h["ckpt_path"]
    }' results_pi0fast_final.csv | sort -k 1 -nr | head -1 | awk '{print $2}')

python scripts/plot_failure_curves.py \
    --ckpt "$LSTM_CKPT" \
    --out-dir notebooks/failure_curves_lstm_pi0fast \
    --threshold 0.5 --per-rollout
```

Replace `lstm` with `gru`/`transformer`/`tcn`/etc. to plot any other architecture.

---

## Repository layout

See [`BIGRED200_FEAT_VIS_SETUP.md` § 5](BIGRED200_FEAT_VIS_SETUP.md) for the
authoritative tree. Highlights:

```
safe/
├── failure_prob/                # Python package (the SAFE codebase)
│   ├── conf/                    # Hydra configs
│   ├── data/                    # Dataset loaders
│   ├── model/                   # All detector models (incl. our additions)
│   ├── utils/                   # Eval routines, video, conformal, metrics
│   └── train.py                 # Main training entrypoint
├── scripts/
│   ├── batch_training/          # SLURM submission scripts
│   ├── extract_results.py       # OUR tool — parses SLURM logs into CSV/leaderboard
│   ├── plot_failure_curves.py   # OUR tool — per-rollout failure-score plots
│   └── visualize_features.py    # FROM SAFE — 2D feature projections
├── results/                     # Curated sample outputs (committed; small PNGs)
├── docs/DATASET_LAYOUT.md       # On-disk dataset structure
├── BIGRED200_FEAT_VIS_SETUP.md  # Full BR200 reproduction guide
└── BIGRED200_GUIDANCE.md        # Generic IU HPC onboarding
```

The training datasets (`rollouts_all 2/`, `openvla_widowx/`) are gitignored —
download them from the SAFE authors' release links above.

---

## Sample results

A small curated set of figures lives in [`results/`](results/) for quick
inspection without running the pipeline:

- `results/sample_feat_vis/` — PCA / t-SNE / UMAP projections of pi0-FAST and OpenVLA hidden states
- `results/sample_failure_curves/` — TCN per-rollout failure-score curves on OpenVLA WidowX (overlay, per-task panels, detection-time histogram, individual rollout PNGs)

See [`results/README.md`](results/README.md) for what each figure means.

---

## Acknowledgements & full credit

This is **almost entirely the SAFE authors' work**. They built the framework,
collected the datasets, designed the methodology, and released a clean,
reproducible codebase. We are extremely grateful for the public release —
this project would not exist without it.

**Original SAFE authors (NeurIPS 2025):**
[Qiao Gu](https://georgegu1997.github.io/),
[Yuanliang Ju](https://scholar.google.com/citations?user=rG90YVAAAAAJ&hl=zh-CN),
[Shengxiang Sun](https://owensun2004.github.io/),
[Igor Gilitschenski](https://www.gilitschenski.org/igor/),
[Haruki Nishimura](https://harukins.github.io/),
[Masha Itkina](https://mashaitkina.weebly.com/),
[Florian Shkurti](https://www.cs.toronto.edu/~florian/).

- **SAFE paper**: *SAFE: Multitask Failure Detection for Vision-Language-Action Models* (NeurIPS 2025) — [arXiv](https://arxiv.org/abs/2506.09937), [project page](https://vla-safe.github.io/), [PDF](https://arxiv.org/pdf/2506.09937)
- **SAFE GitHub repository (the basis of this fork)**: https://github.com/vla-safe/SAFE
- **Pi0-FAST Franka rollouts dataset**: https://drive.google.com/file/d/13z_cdwnaJota2iHkZbhYgVALujZwtM3b/view?usp=sharing — collected and released by the SAFE authors
- **OpenVLA WidowX rollouts dataset**: https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing — collected and released by the SAFE authors

The SAFE codebase itself stands on the shoulders of:

- [openvla](https://github.com/openvla/openvla) — OpenVLA
- [openpi](https://github.com/Physical-Intelligence/openpi) — pi0 / pi0-FAST
- [open-pi-zero](https://github.com/allenzren/open-pi-zero) — open reimplementation of pi0-style models
- [FAIL-Detect](https://github.com/CXU-TRI/FAIL-Detect) — Chen et al.'s failure-detection baselines (RND + log-density)

We additionally thank the IU UITS Research Technologies team for Big Red 200
access and support.

---

## Citation

If you use this work, **please cite the SAFE paper** (which is the actual
research contribution):

```bibtex
@article{gu2025safe,
  title   = {SAFE: Multitask Failure Detection for Vision-Language-Action Models},
  author  = {Gu, Qiao and Ju, Yuanliang and Sun, Shengxiang and Gilitschenski, Igor and Nishimura, Haruki and Itkina, Masha and Shkurti, Florian},
  journal = {arXiv preprint arXiv:2506.09937},
  year    = {2025}
}
```

If you specifically use the additional architectures or scripts from this
fork (GRU/Transformer/TCN detectors, paper-ready leaderboard extractor, or
per-rollout plot script), feel free to also reference this repository.

---

## License

This fork inherits the license of the upstream SAFE repository. Please refer
to the original SAFE repo for the canonical license file.
