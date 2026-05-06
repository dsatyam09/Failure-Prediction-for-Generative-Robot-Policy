#!/usr/bin/env bash
# Push local repo changes to BR200, excluding the dataset (already there)
# and other ephemeral dirs. Run from your Mac:
#   bash sync_to_br200.sh

set -euo pipefail

rsync -avz \
  --exclude='.git' \
  --exclude='rollouts_all 2' \
  --exclude='wandb' \
  --exclude='outputs' \
  --exclude='multirun' \
  --exclude='logs' \
  --exclude='*.egg-info' \
  --exclude='__pycache__' \
  /Users/satyam/Personal/Projects/safe/ \
  sbdubey@bigred200.uits.iu.edu:/N/scratch/sbdubey/safe/
