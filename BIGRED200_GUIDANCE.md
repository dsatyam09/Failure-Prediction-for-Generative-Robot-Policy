# Big Red 200 — Practical Guidance

A walkthrough for getting a new researcher productive on Indiana University's
Big Red 200 (BR200) GPU cluster. Written from real experience setting up the
SAFE project (PyTorch + 47 GB dataset + SLURM jobs). Skim the table of
contents and jump to whatever you need.

---

## 0. What is BR200?

BR200 is IU's flagship supercomputer. For deep learning, you mostly care about:

- **66 GPU nodes**, each with 4× NVIDIA A100-40GB (you typically request 1 of these per job).
- **SLURM scheduler** — you don't run code directly; you submit jobs and SLURM finds you a node.
- **Login nodes** (`login1`, `login2`) — what you SSH into. Use these only for editing files, submitting jobs, small file ops. **Never run training on a login node.**
- Different **partitions** = different queues with different time limits and priority. Pick the right one (see §6).

---

## 1. Access prerequisites

Before you can do anything, you need three things:

1. **An IU account** with two-factor (Duo) set up.
2. **Membership in the BR200 entitlement group** — confirm with:
   ```bash
   groups
   ```
   You should see `iu-entlmt-app-rt-bigred200-users`.
3. **A SLURM allocation** (project account) — without this you can log in but can't run jobs.
   - Get one via [projects.rt.iu.edu](https://projects.rt.iu.edu/).
   - Students can self-enroll in PI "lamhuber" → "HPC for Students" (no advisor needed).
   - Researchers create a project under their PI's username.
   - Confirm with:
     ```bash
     sacctmgr show associations user=$USER
     ```
     If you see an `Account` (e.g. `c02114`) → you're set. Otherwise, request one and wait a few days.

---

## 2. First connection

```bash
ssh YOUR_IU_USERNAME@bigred200.uits.iu.edu
# password + Duo (push or passcode)
```

You'll land on `login1` or `login2`.

**Pro tip — set up SSH multiplexing** so you don't Duo-prompt every command. Add to your local `~/.ssh/config`:

```
Host br200
    HostName bigred200.uits.iu.edu
    User YOUR_IU_USERNAME
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60
```

Then:
```bash
mkdir -p ~/.ssh/sockets
ssh br200    # one Duo prompt; subsequent connections in 4h are free
```

---

## 3. Storage tiers (critical to get right)

BR200 has three storage areas. Putting things in the wrong one is the #1 source of pain.

| Mount | Size | Lifetime | Use for |
|-------|------|----------|---------|
| `~` (home, `/N/u/$USER/BigRed200`) | 100 GB | Permanent | Dotfiles, small scripts, code you edit |
| `/N/scratch/$USER/` | Large (TB) | **Auto-deleted after 30 days of no access** | Datasets, conda envs, model checkpoints, job outputs |
| `/N/slate/$USER/` | 800 GB default | Permanent | Long-term data (only if your project has a Slate allocation) |

**Rules of thumb:**
- Conda envs, datasets, output dirs, anything multi-GB → **scratch**, not home. Home fills up fast.
- Set cache vars to scratch in your `~/.bashrc`:
  ```bash
  export PIP_CACHE_DIR="/N/scratch/$USER/.cache/pip"
  export HF_HOME="/N/scratch/$USER/.cache/huggingface"
  mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"
  ```
- The 30-day scratch purge is real. If you go on vacation, `touch` your data periodically or move it to Slate.

---

## 4. The module system

BR200 uses Lmod. Modules add things to your PATH/LD_LIBRARY_PATH on demand.

```bash
module avail              # list all modules
module avail conda        # find conda modules specifically
module load conda/25.3.0  # load conda
module load cudatoolkit/12.6
module list               # what's currently loaded
module purge              # unload everything
```

Always `module purge` at the start of an sbatch script and load only what you need — avoids weird interaction effects between modules.

---

## 5. Setting up your Python environment (one-time)

This is what worked end-to-end for the SAFE project. Replace `myproj` and the package list with your own.

```bash
# Get conda
module load conda/25.3.0
conda init bash
source ~/.bashrc

# Create env on scratch (NEVER in home)
conda create --prefix /N/scratch/$USER/envs/myproj python=3.10 -y
conda activate /N/scratch/$USER/envs/myproj

# Install your stack
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install <your other deps>
```

**Gotcha:** `import torch` on the **login node** can throw `FileNotFoundError: nvidia/cublas/lib`. That's because the login node has no GPU/CUDA. Don't worry — it'll work fine on a compute node where CUDA is loaded via `module load cudatoolkit/12.6`. To verify imports without touching torch:
```bash
python -c "import hydra, umap; print('ok')"
```

---

## 6. SLURM partitions cheat sheet

The choice of partition decides your queue wait + max walltime:

| Partition | Max walltime | Best for | Notes |
|-----------|--------------|----------|-------|
| `gpu-debug` | 1 hour | Quick tests, CI-like jobs | Fast queue, only 2 GPU nodes total |
| `gpu` | 2 days | Long training jobs | Can have 4-8 h+ queue waits |
| `general` | 4 days | CPU-only jobs (preprocessing, eval) | Many nodes, fast queue |

**Heuristic:** profile a small run on `gpu-debug` first. If it works, scale up to `gpu` for the full run.

Check what's available right now:
```bash
sinfo -p gpu-debug
sinfo -p gpu
squeue -p gpu --start | head     # see scheduled start times for pending jobs
squeue -p gpu-debug | wc -l      # queue depth
```

---

## 7. Anatomy of an sbatch script

Minimal template — save as `myjob.sbatch`:

```bash
#!/bin/bash
#SBATCH -J myjob_name              # appears in squeue
#SBATCH -A YOUR_ACCOUNT_ID         # from sacctmgr; e.g. c02114
#SBATCH -p gpu-debug               # partition (see §6)
#SBATCH -N 1                       # number of nodes
#SBATCH --gres=gpu:1               # 1 GPU; omit for CPU-only
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00                # walltime HH:MM:SS
#SBATCH -o logs/%x-%j.out          # stdout: <jobname>-<jobid>.out
#SBATCH -e logs/%x-%j.err          # stderr

set -euo pipefail
mkdir -p logs

# Always purge first, then load only what you need
module purge
module load cudatoolkit/12.6 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /N/scratch/$USER/envs/myproj

# Sanity prints — useful for debugging
echo "Host: $(hostname)"
nvidia-smi
python -c "import torch; print('cuda=', torch.cuda.is_available())"

# Your actual work
cd /N/scratch/$USER/myproj
python train.py
```

Submit:
```bash
sbatch myjob.sbatch        # → "Submitted batch job 1234567"
```

---

## 8. Daily workflow commands

```bash
# Submit
sbatch myjob.sbatch

# What's running / queued for me?
squeue -u $USER

# What's the state? (R = running, PD = pending, CG = completing, F = failed)
squeue -u $USER -o "%.10i %.9P %.8j %.8T %.10M %R"

# Estimated start time for a pending job
squeue -j JOBID --start

# Live tail the logs
tail -f logs/myjob_name-JOBID.out
tail -f logs/myjob_name-JOBID.err

# Cancel a job
scancel JOBID

# Cancel ALL my pending jobs
scancel -u $USER -t PD

# After a job finishes, see how it went
sacct -X -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

---

## 9. File transfer (Mac/laptop ↔ BR200)

Use `rsync` over SSH. From your **local machine**:

```bash
# Push (local → BR200)
rsync -avhP /local/path/ YOUR_USERNAME@bigred200.uits.iu.edu:/N/scratch/$USER/path/

# Pull (BR200 → local)
rsync -avhP YOUR_USERNAME@bigred200.uits.iu.edu:/N/scratch/$USER/path/ /local/path/
```

Flags explained:
- `-a` archive (preserves permissions, dates, recursion)
- `-v` verbose
- `-h` human-readable sizes
- `-P` show progress + `--partial` (resume on disconnect)
- Add `--exclude='.git' --exclude='*.egg-info'` to skip noise

**For large transfers (>10 GB):** wrap in `tmux` so it survives disconnects:
```bash
tmux new -s xfer
# paste rsync command
# Ctrl-b d to detach; tmux attach -t xfer to come back
```

### Gotcha: paths with spaces over SSH

If your path has a space, the remote shell may eat it during expansion. Example: source `"data/foo bar/"` gets uploaded into `data/foo` (the `bar/` got dropped). Two fixes:

1. **Rename to remove the space first** (simplest).
2. **Escape twice** for the remote side:
   ```bash
   rsync -avhP "/local/foo bar/" user@host:'"/remote/foo bar/"'
   ```
3. **Or use `--protect-args` / `-s`**.

If a transfer "succeeded" but the data isn't where you expect, run on BR200:
```bash
find /N/scratch/$USER -maxdepth 4 -name "<part-of-name>*"
```
to figure out where it actually went.

---

## 10. Debugging failed jobs

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Job exits immediately, `.err` shows config-not-found | Missing config file (didn't get rsynced) | Push the missing file from local |
| `OUT_OF_MEMORY` in `sacct` | Asked for too little RAM | Bump `--mem` |
| `TIMEOUT` | Walltime too short | Increase `-t` or move to a longer-walltime partition |
| `NODE_FAIL` | Hardware glitch (rare) | Just resubmit |
| `tail -f` shows no progress for many minutes | stdout is block-buffered when redirected to a file | Tail the `.err` file instead (tqdm goes to stderr) |
| `import torch` on login node fails with CUDA error | Login nodes have no GPU | Ignore — it works on the compute node |
| `conda: command not found` after re-login | Module not auto-loaded | `module load conda/25.3.0` (consider adding to `~/.bashrc`) |
| Pasted multi-line command breaks at line wrap | Terminal pastes line breaks as Enter | Save as a `.sh` file and run `bash file.sh`, or chain with `&&` on one line |
| Job shows `PD` with reason `Priority` for hours | Real queue contention or per-user concurrency cap | Check `squeue -u $USER` count, consider switching partitions |

---

## 11. Common mistakes (especially for first-timers)

- ❌ Running training on the login node. Always submit via `sbatch` or `salloc`.
- ❌ Storing data in home. 100 GB fills up shockingly fast — use scratch.
- ❌ Using `general` partition for GPU jobs. It has no GPUs; the job will run on CPU and crawl.
- ❌ Forgetting `-A YOUR_ACCOUNT`. The job will be rejected.
- ❌ Asking for max walltime "to be safe". Shorter walltimes get scheduled faster.
- ❌ Not purging modules. Stale module state causes weird CUDA-version mismatches.
- ❌ `git clone`-ing into a non-empty dir. Clone elsewhere first, then move.
- ❌ Pasting a multi-line command into a terminal that interprets newlines literally.
- ❌ Trying to run `apt install` / `sudo` anything. You're not root.

---

## 12. A real worked example: SAFE project

Concrete numbers for sanity-checking your own setup. The SAFE feat-vis pipeline:

| Step | Time | Notes |
|------|------|-------|
| Conda env creation + pip installs | ~10 min | One-time |
| `git clone` of SAFE repo | seconds | |
| `rsync` of 47 GB dataset to scratch | ~40 min | Depends on home upload speed |
| Single feat-vis job (load + 1 projector) | ~33 min | 30 min loading + 3 min PCA |
| 3 jobs in parallel on `gpu-debug` | ~33 min total | Each runs on its own node |
| Pull 800 MB of plots back to laptop | ~2 min | |

**Total elapsed:** about an hour and a half once env is set up.

**Key trick learned:** when one big job won't fit in the `gpu-debug` 1h walltime but the script is embarrassingly parallel (e.g. 3 different projectors), submit 3 separate jobs with an env-var dispatch:

```bash
sbatch --export=ALL,PROJECTOR=pca  myjob.sbatch
sbatch --export=ALL,PROJECTOR=tsne myjob.sbatch
sbatch --export=ALL,PROJECTOR=umap myjob.sbatch
```

…and inside the sbatch:
```bash
PROJECTOR=${PROJECTOR:-pca}
python script.py projector=${PROJECTOR}
```

This beats waiting 4-8 h on the `gpu` partition queue.

---

## 13. Where to get more help

- Official: `hps-admin@iu.edu`
- IU KB: https://kb.iu.edu/d/aoku
- Slack: ask in your lab's HPC channel; UITS has staff who answer fast
- BR200 maintenance reservations: announced in the SSH banner at login — plan around them

---

## 14. Quick start TL;DR

```bash
# One-time
ssh user@bigred200.uits.iu.edu
module load conda/25.3.0
conda init bash && source ~/.bashrc
conda create --prefix /N/scratch/$USER/envs/myproj python=3.10 -y
conda activate /N/scratch/$USER/envs/myproj
pip install <your deps>

# Each project
mkdir -p /N/scratch/$USER/myproj
git clone <repo> /N/scratch/$USER/myproj
# rsync data from laptop into /N/scratch/$USER/myproj/

# Each run
cd /N/scratch/$USER/myproj
sbatch scripts/myjob.sbatch
squeue -u $USER
tail -f logs/<jobname>-<jobid>.out
```

If you remember just one thing: **everything heavy goes on `/N/scratch`, every job is submitted via `sbatch`, and `gpu-debug` is your friend for testing.**
