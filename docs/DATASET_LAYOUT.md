# Dataset Layout Reference

The training datasets are **not** committed to this repo (multi-GB). This file
documents their expected on-disk layout so anyone reproducing the experiments
knows what the data should look like after they download / generate it.

Get the data from:
- **rollouts_all 2** (pi0-FAST Franka): https://drive.google.com/file/d/13z_cdwnaJota2iHkZbhYgVALujZwtM3b/view?usp=sharing
- **openvla_widowx**: https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing

Both should be unzipped at the repo root.

---

## `rollouts_all 2/` — pi0-FAST Franka rollouts

~47 GB. Organized by date, then by task within each date.

```
rollouts_all 2/
├── 20250420_rollouts/
│   ├── task0_pick_up_the_lid_and_place_it_on_the_pot/
│   │   ├── env_records/                          # one record per episode
│   │   │   ├── task0--ep0--succ0--meta.pkl       # episode metadata (success label, task id, etc.)
│   │   │   ├── task0--ep0--succ0--external_left.mp4   # external view
│   │   │   ├── task0--ep0--succ0--wrist.mp4           # wrist-mounted camera
│   │   │   ├── task0--ep1--succ0--meta.pkl
│   │   │   └── ...
│   │   └── policy_records/                        # one record per inference call
│   │       ├── step_1000--droid_debug--task_0--ep_9--t_72--meta.pkl
│   │       ├── step_1001--droid_debug--task_0--ep_9--t_80--meta.pkl
│   │       └── ...
│   ├── task1_pick_up_the_lid_from_the_pot_and_place_it_on_the_table/
│   ├── task2_.../
│   └── ...
├── 20250422_rollouts/
├── 20250429_rollouts/
├── 20250502_rollouts/
├── 20250504_rollouts/
├── 20250505_rollouts/
├── 20250509_rollouts/
└── 20250510_rollouts/
```

### Per-episode `env_records/*--meta.pkl` contents
Pickled dict with at least:
- `task_id` (int)
- `episode_idx` (int)
- `episode_success` (0 or 1)
- `task_description` (str)
- `task_suite_name` (str)
- `replan_steps` (int — execution horizon)

### Per-step `policy_records/*--meta.pkl` contents
Pickled dict with the inference output. The relevant feature for failure
detection is `pre_logits` of shape `(17, 2048)` (17 action tokens × 2048 hidden
dim, bfloat16). The codebase mean-pools the 17 tokens to `(2048,)` per
timestep.

Other keys: `observation/state`, `prompt`, `actions (10,7)`, `decode_step`,
`encoded`, `logits`, `raw_actions`, `state`, `action_start_index_in_vocab`,
`action_end_index_in_vocab`.

### Discovery in code
The `failure_prob/data/pizero_fast_droid.py` loader iterates `data_path` from
`failure_prob/conf/dataset/pizero_fast_droid_rollouts_all_2.yaml` (which
hard-lists every task folder).

---

## `openvla_widowx/` — OpenVLA WidowX rollouts

~1.6 GB. Flat structure: one directory per task, files named by episode.

```
openvla_widowx/
├── put_the_carrot_on_plate_1/
│   ├── task_put_the_carrot_on_plate--ep0--succ0.csv     # per-step features (csv)
│   ├── task_put_the_carrot_on_plate--ep0--succ0.mp4     # rollout video
│   ├── task_put_the_carrot_on_plate--ep0--succ0.pkl     # rollout metadata
│   ├── task_put_the_carrot_on_plate--ep1--succ0.csv
│   └── ...
├── task_lift_aaa_battery_1/
├── task_lift_aaa_battery_2/
├── task_lift_blue_cup/
├── task_lift_eggplant_1/
├── task_lift_eggplant_2/
├── task_lift_red_bottle_1/
├── task_lift_red_bottle_2/
├── task_lift_red_bottle_3/
├── task_lift_red_bottle_4/
├── task_put_blue_cup_on_plate_1/
├── task_put_blue_cup_on_plate_2/
├── task_put_the_carrot_on_plate_1/
├── task_put_the_carrot_on_plate_2/
├── task_put_the_red_block_into_the_pot_1/
├── task_put_the_red_block_into_the_pot_2/
├── task_put_the_red_bottle_into_pot_1/
├── task_put_the_red_bottle_into_pot_2/
└── task_put_the_red_bottle_into_pot_3/
```

### Filename schema
`task_<task_name>--ep<idx>--succ<0|1>.{csv,mp4,pkl}`

`succ` digit is the **ground-truth success label** (1 = succeeded, 0 = failed).

### Discovery in code
`failure_prob/data/openvla.py` globs `openvla_widowx/**/` from
`failure_prob/conf/dataset/openvla_widowx.yaml`.

---

## Where the loaders compute the splits

Both datasets are split into:
- `train` — held-out rollouts of *seen* tasks (what we train on)
- `val_seen` — held-out rollouts of *seen* tasks (held-out within-distribution)
- `val_unseen` — rollouts of *held-out tasks* (zero-shot generalization, the test set)

Splits are seeded by `train.seed` for reproducibility. The split ratios live in
each dataset's YAML (`unseen_task_ratio`, `seen_train_ratio`).

---

## Sanity-check commands

```bash
# Dataset sizes
du -sh "rollouts_all 2"            # expected: ~47 GB
du -sh openvla_widowx              # expected: ~1.6 GB

# Count rollouts (env_records files)
find "rollouts_all 2" -name "*--meta.pkl" | wc -l   # expected: ~5000+
find openvla_widowx -name "*.pkl" | wc -l           # expected: ~500+

# Pick one rollout, inspect contents
python -c "
import pickle, sys
p = pickle.load(open('rollouts_all 2/20250420_rollouts/task0_pick_up_the_lid_and_place_it_on_the_pot/env_records/task0--ep0--succ0--meta.pkl', 'rb'))
print(p.keys())
print('task_id', p.get('task_id'))
print('episode_success', p.get('episode_success'))
"
```
