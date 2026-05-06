# Failure Prediction for Generative Robot Policy: An Architecture Ablation on SAFE

This is a course-project extension of [SAFE](https://github.com/vla-safe/SAFE) (Gu et al., NeurIPS 2025). The original work introduced the multitask failure-detection problem for VLA models and showed it can be solved zero-shot on unseen tasks. We took their public release, kept the core method intact, and asked one practical question on top of it: which sequence-modeling backbone actually works best for this job when the dataset is small?

To answer it we plugged in three more architectures (GRU, a causal Transformer, and a TCN), ran the whole sweep on Indiana University's Big Red 200 cluster, and built tooling to turn the SLURM logs into a paper-style ablation table plus per-rollout failure-score plots.

> Almost everything that matters here is the SAFE authors' work. Their method, their data, their codebase. We added some baseline architectures and reproducibility scripts. If you cite anything, please cite their NeurIPS paper, not this fork. Full attribution is in the [Acknowledgements](#acknowledgements) below.

![Splash figure (from the original SAFE paper)](assets/safe-teaser-static.png)

---

## Table of contents

1. [What's actually new in this fork](#whats-actually-new-in-this-fork)
2. [What you'd need to run it](#what-youd-need-to-run-it)
3. [Quick start (single GPU)](#quick-start-single-gpu)
4. [Full reproduction (HPC cluster)](#full-reproduction-hpc-cluster)
5. [Repository layout](#repository-layout)
6. [Sample results](#sample-results)
7. [Acknowledgements](#acknowledgements)
8. [Citation](#citation)
9. [License](#license)

---

## What's actually new in this fork

Everything that was already in SAFE works the same way. We just added a few files alongside theirs.

| What | Where it lives | Why |
|------|----------------|-----|
| GRU detector | `failure_prob/model/gru.py` plus a one-line config in `failure_prob/conf/model/gru.yaml` | Lighter than LSTM, easy to drop in for an apples-to-apples comparison |
| Causal Transformer | `failure_prob/model/transformer.py`, `failure_prob/conf/model/transformer.yaml` | Standard self-attention with a causal mask. Uses AdamW and learning-rate warmup since regular Adam was unstable on this small dataset |
| TCN (dilated causal conv) | `failure_prob/model/tcn.py`, `failure_prob/conf/model/tcn.yaml` | A non-recurrent, parallelizable alternative to LSTMs |
| Final sweep scripts | `scripts/batch_training/bigred200_*_final.sbatch` and `submit_*_final.bash` | Run the whole architecture grid for both datasets in one shot, on SLURM, with 5 seeds |
| Result extractor | `scripts/extract_results.py` | Pulls `wandb` summary blocks out of the SLURM stderr logs and produces a paper-ready leaderboard with mean and std across seeds |
| Failure-curve plotter | `scripts/plot_failure_curves.py` | Loads any saved checkpoint and plots per-rollout failure scores over time. Useful for picking figures |
| Cluster docs | `BIGRED200_FEAT_VIS_SETUP.md`, `BIGRED200_GUIDANCE.md` | Step-by-step guide for running this on Big Red 200, including all the gotchas we hit |
| Dataset layout doc | `docs/DATASET_LAYOUT.md` | What the on-disk dataset folders should look like once you download them |

The SAFE training loop, evaluation routine, and dataset format are untouched. Our extra detectors live next to the existing LSTM, MLP, KNN-style, RND, log-density, and handcrafted baselines and use the exact same loss and evaluation code.

---

## What you'd need to run it

We did the final sweeps on Indiana University's Big Red 200. If you don't have access to that, the smaller OpenVLA WidowX experiments will fit on a single workstation GPU.

| Resource | What we used | Minimum to reproduce |
|----------|--------------|----------------------|
| GPU | 1× NVIDIA A100 (40 GB) | 1× A100 / RTX 3090 / RTX 4090 with 24+ GB |
| RAM | 96 to 200 GB on the cluster node | 64 GB for the full Franka dataset, 32 GB for OpenVLA |
| Disk | ~50 GB free for both datasets | Same |
| OS | SUSE Linux + SLURM 23.x | Any Linux with conda |
| Python | 3.10 | 3.10 |
| PyTorch | 2.x with CUDA 12.8 wheels | Same |
| Cluster module | `cudatoolkit/12.6` (this is essential, not optional, see note below) | Equivalent system CUDA on your machine |
| Wall-clock per dataset sweep | About 9 to 13 hours | Similar |

One thing that bit us hard: on Big Red 200's Lustre filesystem, `import torch` would hang for hours because the bundled NVIDIA `.so` libraries take forever to `dlopen`. The fix is to load the system cudatoolkit module first, so torch finds the system libs. If you're running on your own machine and have CUDA installed normally, you won't hit this.

The full set of cluster-specific landmines (and how we got around them) is documented in [`BIGRED200_FEAT_VIS_SETUP.md`](BIGRED200_FEAT_VIS_SETUP.md).

---

## Quick start (single GPU)

This path works on any Linux box with a compatible GPU. It runs one config to confirm everything wires up, no SLURM needed.

```bash
# 1. Clone
git clone https://github.com/dsatyam09/Failure-Prediction-for-Generative-Robot-Policy.git
cd Failure-Prediction-for-Generative-Robot-Policy

# 2. Set up the environment
conda create -n vla-safe python=3.10 -y
conda activate vla-safe
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas scipy pyyaml tqdm "imageio[ffmpeg]" hydra-core omegaconf scikit-learn opencv_python einops wandb plotly matplotlib natsort flask ml_dtypes umap-learn
pip install -e .

# 3. Get the datasets (released by the SAFE authors). See docs/DATASET_LAYOUT.md
#    for the expected folder structure once you've unzipped them.
#      Pi0-FAST Franka rollouts (~47 GB):
#        https://drive.google.com/file/d/13z_cdwnaJota2iHkZbhYgVALujZwtM3b/view?usp=sharing
#      OpenVLA WidowX rollouts (~1.6 GB):
#        https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing

# 4. Tell the code where the datasets live
cp setup_envs.bash.template setup_envs.bash
# Edit setup_envs.bash so the paths point to the unzipped datasets

# 5. Run a single config. ~10 minutes with a warm cache.
source setup_envs.bash
export WANDB_MODE=offline   # or run `wandb login` if you'd rather log online
python -u -m failure_prob.train \
    dataset=pizero_fast_droid_rollouts_all_2 \
    dataset.data_path_prefix="${SAFE_OPENPI_DROID_ROLLOUT_ROOT}" \
    dataset.feat_name=pre_logits dataset.token_idx_rel=mean dataset.load_to_cuda=False \
    model=lstm model.n_epochs=200 train.seed=0 train.exp_suffix=quicktest
```

If that finishes with `Smoke test finished` and a wandb summary, you're set.

---

## Full reproduction (HPC cluster)

This is what we actually did to produce the numbers in the report. It assumes a SLURM-managed cluster with a `gpu` partition. The complete walkthrough, including every error we hit and what fixed it, lives in [`BIGRED200_FEAT_VIS_SETUP.md`](BIGRED200_FEAT_VIS_SETUP.md). The condensed version:

```bash
# 1. Set up the env once. See BIGRED200_FEAT_VIS_SETUP.md §4 for the full version.
module purge
module load conda/25.3.0
module load cudatoolkit/12.6     # required, see note above
conda activate /path/to/envs/vla-safe

# 2. Submit both final sweeps in parallel (one per dataset)
cd /path/to/safe
sbatch scripts/batch_training/bigred200_pi0fast_final.sbatch
sbatch scripts/batch_training/bigred200_openvla_final.sbatch

# 3. Watch them. Each takes about 9 to 13 hours.
squeue -u $USER
sacct -X -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode

# 4. After both jobs finish, sync wandb runs from a node that has internet.
#    Compute nodes don't, so we keep wandb in offline mode and sync later.
wandb sync wandb/wandb/

# 5. Build the per-architecture leaderboard (mean and std over the 5 seeds)
python scripts/extract_results.py \
    --out logs/safe_pi0fast_final-<jobid>.out \
    --err logs/safe_pi0fast_final-<jobid>.err \
    --csv results_pi0fast_final.csv

python scripts/extract_results.py \
    --out logs/safe_openvla_final-<jobid>.out \
    --err logs/safe_openvla_final-<jobid>.err \
    --csv results_openvla_final.csv

# 6. Plot per-rollout failure curves for the best LSTM run
LSTM_CKPT=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next}
    $h["model_name"]=="lstm" && $h["ckpt_path"]!="" {
        print $h["auc_val_unseen"], $h["ckpt_path"]
    }' results_pi0fast_final.csv | sort -k 1 -nr | head -1 | awk '{print $2}')

python scripts/plot_failure_curves.py \
    --ckpt "$LSTM_CKPT" \
    --out-dir notebooks/failure_curves_lstm_pi0fast \
    --threshold 0.5 --per-rollout
```

Swap `lstm` for `gru`, `transformer`, or `tcn` to plot whichever architecture you want.

---

## Repository layout

The full tree is in [`BIGRED200_FEAT_VIS_SETUP.md` §5](BIGRED200_FEAT_VIS_SETUP.md). The short version:

```
safe/
├── failure_prob/                  # the SAFE Python package
│   ├── conf/                      # Hydra configs
│   ├── data/                      # dataset loaders
│   ├── model/                     # all detector models, including ours
│   ├── utils/                     # eval routines, conformal stuff, video
│   └── train.py                   # main training entrypoint
├── scripts/
│   ├── batch_training/            # SLURM and bash submission scripts
│   ├── extract_results.py         # ours: SLURM logs to leaderboard CSV
│   ├── plot_failure_curves.py     # ours: per-rollout failure-score plots
│   └── visualize_features.py      # from SAFE: 2D feature projections
├── results/                       # curated sample plots, committed
├── docs/DATASET_LAYOUT.md         # what the dataset folders should look like
├── BIGRED200_FEAT_VIS_SETUP.md    # cluster reproduction guide
└── BIGRED200_GUIDANCE.md          # generic IU HPC onboarding
```

The training datasets aren't committed (too big, and they belong to the SAFE authors). Download links are in the [Acknowledgements](#acknowledgements) section.

---

## Sample results

A small set of figures lives in [`results/`](results/) so you can see what the pipeline produces without running anything.

* `results/sample_feat_vis/` has PCA, t-SNE, and UMAP projections of pi0-FAST and OpenVLA hidden states.
* `results/sample_failure_curves/` has TCN per-rollout failure-score curves on the OpenVLA WidowX dataset. There's an overlay plot, per-task panels, a detection-time histogram, and a few individual rollout PNGs you can use as paper figures.

The `results/README.md` explains what each file shows.

---

## Acknowledgements

The credit for everything that's actually being done here belongs to the SAFE authors. We did not invent the method, collect the data, or write the codebase. We adapted their public release for a class project.

**SAFE: Multitask Failure Detection for Vision-Language-Action Models** (NeurIPS 2025) by [Qiao Gu](https://georgegu1997.github.io/), [Yuanliang Ju](https://scholar.google.com/citations?user=rG90YVAAAAAJ&hl=zh-CN), [Shengxiang Sun](https://owensun2004.github.io/), [Igor Gilitschenski](https://www.gilitschenski.org/igor/), [Haruki Nishimura](https://harukins.github.io/), [Masha Itkina](https://mashaitkina.weebly.com/), and [Florian Shkurti](https://www.cs.toronto.edu/~florian/).

* Paper: [arXiv:2506.09937](https://arxiv.org/abs/2506.09937)
* Project page: https://vla-safe.github.io/
* Original codebase (this fork builds on it): https://github.com/vla-safe/SAFE
* Pi0-FAST Franka rollout dataset: https://drive.google.com/file/d/13z_cdwnaJota2iHkZbhYgVALujZwtM3b/view?usp=sharing
* OpenVLA WidowX rollout dataset: https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing

The SAFE codebase itself was built on top of these projects, all of which deserve credit too:

* [openvla](https://github.com/openvla/openvla)
* [openpi](https://github.com/Physical-Intelligence/openpi)
* [open-pi-zero](https://github.com/allenzren/open-pi-zero)
* [FAIL-Detect](https://github.com/CXU-TRI/FAIL-Detect) (the source of the RND and log-density baselines)

We also thank IU UITS Research Technologies for Big Red 200 access.

---

## Citation

If you use this work, please cite the SAFE paper. That's the actual research contribution.

```bibtex
@article{gu2025safe,
  title   = {SAFE: Multitask Failure Detection for Vision-Language-Action Models},
  author  = {Gu, Qiao and Ju, Yuanliang and Sun, Shengxiang and Gilitschenski, Igor and Nishimura, Haruki and Itkina, Masha and Shkurti, Florian},
  journal = {arXiv preprint arXiv:2506.09937},
  year    = {2025}
}
```

If you specifically use the GRU, Transformer, or TCN detectors from this fork, or one of our extraction or plotting scripts, feel free to also link back to this repo.

---

## License

This fork inherits the license of the upstream SAFE repository. See the original repo for the canonical license file.
