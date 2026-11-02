# Khoa's MetaWorld π0.5 + VLA-Corrector setup

This guide reproduces the local MetaWorld workflow used in this repository:

```text
MetaWorld MT50 demonstrations
  + fine-tuned PI0.5 policy
  → frozen visual-latent extraction
  → one of: original Corrector / action-prefix / uncertainty Corrector
```

The dataset, PI0.5 checkpoint, extracted caches, and training outputs are
large local artifacts. They are ignored by Git and are not included in this
repository.

## Prior results: original 256-token setup

These prior MetaWorld MT50 results use the original code and the full
**256-token (16×16) visual representation, without spatial mean pooling**.
Training was performed on an **RTX 8000 GPU**.

| Method | Easy | Medium | Hard | Very hard | Overall |
| --- | --- | --- | --- | --- | --- |
| Plain π0.5 baseline | 44.46% | 38.18% | 39.17% | 27.00% | **40.70%** |
| Original VLA-Corrector | 81.25% | 50.91% | 50.00% | 28.00% | **65.50%** |
| Action-prefix only | 79.29% | 51.36% | 52.50% | 35.00% | **65.50%** |
| PACE π0.5, no Corrector | 82.14% | 46.82% | 52.50% | 38.00% | **66.40%** |
| Uncertainty Corrector | 80.54% | 49.55% | **57.50%** | **38.00%** | **66.70%** |

## 1. Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml -n lerobot-dynamicVLA
conda activate lerobot-dynamicVLA

python -m pip install -e '.[metaworld]' --no-build-isolation
```

`environment.yml` pins the compatible versions used in this workspace,
including `gymnasium==1.2.3` and `metaworld==3.0.0`. Do not downgrade
Gymnasium to 1.1.0: it conflicts with this repository's declared dependency.

Set local Hugging Face cache locations so downloads stay inside this project:

```bash
cd ~/Public/Augustine/vla-corrector

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

## 2. Download the dataset and fine-tuned PI0.5

The MetaWorld MT50 dataset is `lerobot/metaworld_mt50`. The PI0.5 checkpoint
used here is the public MT50-fine-tuned `tiantianx/pi05_metaworld` snapshot.
It is full PI0.5 fine-tuning from `lerobot/pi05_base`, rather than a Corrector
checkpoint.

```bash
hf download lerobot/metaworld_mt50 \
  --repo-type dataset \
  --local-dir datasets/metaworld_mt50

hf download tiantianx/pi05_metaworld \
  --local-dir pretrained_weights/pi05_metaworld
```

Approximate storage requirements:

```text
MetaWorld MT50 dataset:        45.9 GB
Fine-tuned PI0.5 checkpoint:   16.6 GB
Full-token latent cache:      ~107.5 GB
64-token mean-pooled cache:   ~26.87 GB (optional workflow in section 6)
```

### Make the local PI0.5 compatibility view

The downloaded policy has three older OpenPI configuration fields that this
repository's PI0.5 parser does not accept. The following command creates a
small compatibility directory. It changes only a copied `config.json` and
symlinks the large weights, so no second 16.6 GB model copy is made.

```bash
python scripts/make_pi05_compat.py \
  --source pretrained_weights/pi05_metaworld \
  --output pretrained_weights/pi05_metaworld_compat
```

## 3. Extract frozen PI0.5 visual latents

This is the original full-token workflow. All three Corrector variants in
section 4 use the same output cache below. If you want **only the 64-token
mean-pooling experiments**, skip sections 3–4 and follow section 6 after the
installation and downloads in sections 1–2; the full-token cache is not needed
for those three pooled Correctors.

```bash
python -m siglip_dynamics.extract \
  --dataset-path ./datasets/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --image-key observation.image \
  --no-use-wrist-image \
  --action-key action \
  --output-path ./outputs/metaworld_mt50_pi05_latents_tf54_fixed \
  --encoder-backend pi05 \
  --encoder-policy-path ./pretrained_weights/pi05_metaworld_compat \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --device cuda
```

This is the single-camera MetaWorld setup:

```text
image:       observation.image
action:      normalized 4D MetaWorld action
latent:      PI0.5 visual tokens, D=2048
Corrector:   h=1, K=10
```

## 4. Choose one Corrector training variant

All three variants keep the fine-tuned PI0.5 backbone frozen. They train only
the lightweight latent dynamics Corrector.

### A. Original paper-style VLA-Corrector

This is the legacy, single-current-action Corrector: `z_t, a_t → Δz(t→t+10)`.
It uses cosine training loss, `h=1`, `K=10`, batch size 512, 30 epochs, and a
42M-scale residual MLP.

```bash
export WANDB_MODE=disabled
export WANDB_SILENT=true

torchrun --nproc_per_node=1 -m siglip_dynamics.train \
  --dataset-path ./outputs/metaworld_mt50_pi05_latents_tf54_fixed \
  --checkpoint-dir ./outputs/metaworld_mt50_pi05_corrector \
  --device cuda \
  --model-type mlp \
  --scale 20m \
  --token-dim 2048 \
  --action-dim 4 \
  --action-embed-dim 256 \
  --dropout 0.0 \
  --h-window 1 \
  --k-step-list 10 \
  --batch-size 512 \
  --epochs 30 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --val-ratio 0.1 \
  --seed 42 \
  --grad-clip-norm 1.0 \
  --patience 5 \
  --train-loss-type cosine
```

Best checkpoint:

```text
outputs/metaworld_mt50_pi05_corrector/h1_k10/best_cosine_model.pt
```

### B. Action-prefix Corrector

This Corrector receives the ordered ten planned actions as condition:
`z_t, (a_t ... a_t+9) → Δz(t→t+10)`.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/prefix.yaml \
  --mode train
```

Live MetaWorld evaluation after training:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/prefix.yaml \
  --mode evaluate
```

### C. Action-prefix + uncertainty Corrector

This has the same ordered ten-action prefix, plus an error head that predicts
the expected future RMSE. Its loss is:

```text
cosine loss + MSE loss + 0.1 × expected-RMSE loss
```

Train it:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml \
  --mode train
```

Then use separate successful MetaWorld episodes to calibrate the online alarm.
Calibration is required only for this uncertainty variant:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml \
  --mode collect-calibration \
  --set evaluation.episodes=10 \
  --set evaluation.save_video=false

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml \
  --mode calibrate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml \
  --mode evaluate
```

## 5. MetaWorld evaluation environment

Set these before any live MetaWorld evaluation command:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
```

Use a **new** output directory each time an evaluation command is rerun. The
evaluation programs deliberately refuse to overwrite a non-empty result
directory.

### A. Plain fine-tuned PI0.5 baseline

This runs the policy only; it does not load a Corrector, PACE, or OGG. The
standard evaluator saves rollout videos under its output directory (up to ten
episodes per task group).

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=./pretrained_weights/pi05_metaworld_compat \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.chunk_size=50 \
  --policy.compile_model=false \
  --env.type=metaworld \
  --env.task='easy,medium,hard,very_hard' \
  --env.episode_length=300 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --rename_map='{"observation.image":"observation.images.base_0_rgb"}' \
  --output_dir=./outputs/eval/metaworld_pi05_baseline
```

### B. Original paper-style VLA-Corrector

The original Corrector does not need a calibration phase. Its legacy MAD/EWMA
residual alarm clears the queued actions after a persistent alarm, then makes
an early PI0.5 call; OGG is enabled only for that recovery replan.

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection \
  --policy.path=./pretrained_weights/pi05_metaworld_compat \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.chunk_size=50 \
  --policy.compile_model=false \
  --env.type=metaworld \
  --env.task='easy,medium,hard,very_hard' \
  --env.episode_length=300 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --rename_map='{"observation.image":"observation.images.base_0_rgb"}' \
  --safety_model_path=./outputs/metaworld_mt50_pi05_corrector/h1_k10 \
  --safety_k=10 \
  --safety_history_frames=1 \
  --threshold_window_size=15 \
  --threshold_k_on=3 \
  --threshold_k_off=2 \
  --threshold_trigger_consecutive_steps=5 \
  --threshold_reset_consecutive_steps=5 \
  --meltdown_cooldown_steps=10 \
  --guidance_eta=1 \
  --guidance_apply_every=1 \
  --guidance_loss_objective=attract_delta_z_correction \
  --guidance_compare_baseline=false \
  --meltdown_use_guidance_replan=true \
  --output_dir=./outputs/eval/metaworld_original_corrector \
  --video_fps=10 \
  --save_analysis=true \
  --compute_baseline_metrics=false \
  --save_raw_video=true \
  --save_episode_npz=true \
  --save_summary_csv=true \
  --save_summary_json=true
```

If your legacy training output used a different checkpoint root, replace only
the value of `--safety_model_path`; it must point to the `h1_k10` directory or
to its `best_cosine_model.pt` file.

### C. PACE (training-free adaptive chunk execution)

PACE is separate from the Corrector. It needs no learned Corrector checkpoint,
but it does need its action-profile thresholds calibrated once from the MT50
demonstrations. The command below writes the artifact named in
`configs/pace/metaworld_pi05.yaml`.

```bash
python -m siglip_dynamics.pace calibrate \
  --config configs/pace/metaworld_pi05.yaml
```

PACE's maximum horizon is 50, so the policy must expose and initially queue 50
actions. The controller then selects a shorter safe boundary from that chunk.

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection \
  --policy.path=./pretrained_weights/pi05_metaworld_compat \
  --policy.device=cuda \
  --policy.n_action_steps=50 \
  --policy.chunk_size=50 \
  --policy.compile_model=false \
  --env.type=metaworld \
  --env.task='easy,medium,hard,very_hard' \
  --env.episode_length=300 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --rename_map='{"observation.image":"observation.images.base_0_rgb"}' \
  --pace_config=configs/pace/metaworld_pi05.yaml \
  --pace_disable_corrector=true \
  --guidance_eta=0 \
  --guidance_compare_baseline=false \
  --meltdown_use_guidance_replan=false \
  --compute_baseline_metrics=false \
  --output_dir=./outputs/eval/metaworld_pi05_pace \
  --video_fps=10 \
  --save_analysis=true \
  --save_raw_video=true \
  --save_episode_npz=true \
  --save_summary_csv=true \
  --save_summary_json=true
```

### D. Action-prefix and uncertainty Correctors

For these two experimental variants, use the commands in section 4. Their
`--mode evaluate` commands launch the same MetaWorld evaluator with the exact
checkpoint and monitor settings recorded in their YAML files. The uncertainty
variant additionally requires the successful `collect-calibration` and
`calibrate` steps before evaluation.

## 6. Spatial mean-pooling: extract, train and evaluate all Corrector modes

The original, prefix and uncertainty Correctors each have an additional pooled
variant. The `pool64` configs use a per-camera **`8×8` grid** to reduce the cached
latent from `256×2048` to **`64×2048`** (about **26.87 GB** for MT50). The earlier
`pool32` (32-token), `pool4` (16-token) and full-token configurations remain
available. The online monitor and OGG use the same pooled representation as
training; PI0.5 itself keeps its full visual tokens.

The commands below run **Original Corrector, Prefix Corrector and Uncertainty
Corrector** with the same pooled representation. Complete installation and
downloads in sections 1–2 first. Plain PI0.5 and PACE use the commands in
section 5; this pooling option applies to the Corrector representation.

### A. Activate the environment

Run from the repository root:

```bash
conda activate lerobot-dynamicVLA

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export WANDB_MODE=disabled
export WANDB_SILENT=true
```

### B. Extract the shared 64-token cache once

This averages each `2×2` patch block of the original `16×16` grid, producing
an `8×8` grid: **64×2048 features per frame** for single-camera MetaWorld.
The estimated quantized cache for 204,806 frames is **26.87 GB (25.03 GiB)**,
including scales, actions and episode indices. Dataset downloads, weights and
evaluation videos take additional space.

Use the new output path below; extraction writes its cache arrays. Set
`--image-token-grid-size 8` for this square grid; omit the rectangular
`--image-token-grid-shape` option.

```bash
python -m siglip_dynamics.extract \
  --dataset-path ./datasets/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --image-key observation.image \
  --no-use-wrist-image \
  --action-key action \
  --output-path ./outputs/metaworld_mt50_pi05_latents_pool64 \
  --encoder-backend pi05 \
  --encoder-policy-path ./pretrained_weights/pi05_metaworld_compat \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --image-token-grid-size 8 \
  --max-output-gb 32 \
  --device cuda
```

All three YAML files below use this cache and set
`model.image_token_grid_size: 8`. Online observations and OGG conditioning
are pooled to the same shape as the training data. Pooling reduces token
storage and token-wise computation; it does not reduce the MLP parameter count
or the policy backbone's visual-token count. Accuracy must be measured with
the evaluations below.

### C. Train the pooled Original Corrector

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml \
  --mode train
```

This retains the legacy current-action model, cosine loss and training split.
Checkpoint:

```text
outputs/experiments/metaworld_original_pool64/h1_k10/best_cosine_model.pt
```

### D. Train the pooled Prefix Corrector

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool64.yaml \
  --mode train
```

Checkpoint:

```text
outputs/experiments/metaworld_prefix_pool64/best_model.pt
```

### E. Train the pooled Uncertainty Corrector

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode train
```

Checkpoint:

```text
outputs/experiments/metaworld_uncertainty_pool64/best_model.pt
```

All three modes keep `h=1`, `K=10`, execution cap `E=10` and predicted chunk
size 50, together with each mode's existing training settings. Run each command
after its preceding step completes successfully. Each mode has its own output
directory and does not overwrite full-token checkpoints.

### F. Collect and calibrate the pooled Uncertainty Corrector

Only the uncertainty mode needs these two steps. Pooling changes the residual
distribution, so collect new calibration traces for this checkpoint instead of
reusing full-token, 32-token or 16-token thresholds.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode collect-calibration \
  --set evaluation.episodes=10 \
  --set evaluation.save_video=false

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode calibrate
```

This collects ten episodes **per task** across MT50, using collection seed
10000. Calibration requires at least 50 successful episodes. The trace and
calibration artifact use their own `pool64` paths in the uncertainty YAML.

### G. Evaluate all three pooled modes on all tasks

Each command evaluates `easy,medium,hard,very_hard`, with **20 episodes per
task**, seed **1000**, and at most **300 environment steps per episode**.
Raw rollout videos and CSV/JSON summaries are enabled. Original and Prefix
evaluation can run after their training; Uncertainty evaluation must follow
successful collection and calibration.

```bash
# Original Corrector: legacy MAD/EWMA alarm and OGG recovery.
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml \
  --mode evaluate

# Prefix Corrector: ordered action-prefix conditioning.
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool64.yaml \
  --mode evaluate

# Uncertainty Corrector: calibrated expected-RMSE alarm and gated OGG.
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode evaluate
```

Results and videos are stored under separate directories:

```text
outputs/experiments/eval_metaworld_original_pool64/
outputs/experiments/eval_metaworld_prefix_pool64/
outputs/experiments/eval_metaworld_uncertainty_pool64/
```

For a repeat evaluation, choose a fresh output directory, for example:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool64.yaml \
  --mode evaluate \
  --set evaluation.output=outputs/experiments/eval_metaworld_prefix_pool64_repeat
```

The existing full-token commands remain in sections 3–5. The earlier 16-/32-token
commands and additional representation details are in
[the spatial-pooling ablation guide](docs/metaworld_spatial_pool_ablation.md).

### Training cache location: disk, RAM or GPU VRAM

Extraction still writes the reusable quantized cache to disk. During training,
`training.cache_location` selects where to read it from:

| Value | Training behavior |
| --- | --- |
| `disk` (default) | Memory-map the cache files and read batches as needed. |
| `ram` | Load the packed cache into host RAM once before training. |
| `gpu` | Load the packed cache into CUDA VRAM once; gather and dequantize batches on the GPU. |

This works for Original, Prefix and Uncertainty, including all pooling sizes.
For example, to train Prefix with the 4×4 spatial mean-pooled cache in VRAM:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool4.yaml \
  --mode train \
  --set training.cache_location=gpu \
  --set training.output=outputs/experiments/metaworld_prefix_pool4_vram
```

Use `training.cache_location=ram` for host RAM. The direct Original trainer also
accepts `--cache-location disk|ram|gpu`. The Brev training launcher forwards the
same YAML override through `--set training.cache_location=gpu` (its default
configs still select the 64-token cache).

Features remain int8 and scales remain fp16; only each selected batch is
dequantized to float32. The 204,806-frame **4×4 grid (16 tokens)** requires about
**6.72 GB / 6.26 GiB** for packed arrays, versus **26.87 GB / 25.03 GiB** for the
8×8 grid (64 tokens). Allow additional memory for the model, optimizer and
dequantized batches. Train and validation share one cache allocation per process;
distributed training loads a separate copy per rank. GPU mode requires CUDA and
zero loader workers, as used by both training implementations. An insufficient
VRAM check reports the required cache size rather than changing modes silently.

## 7. Brev: automated setup and pooled-cache extraction

From your **local checkout**, with the Brev CLI already authenticated:

```bash
bash scripts/brev/setup.sh
```

This defaults to instance `primitive-pink-bovid` and remote directory
`/data/vla-corrector-experiments`. It waits for SSH, copies the source including
local edits, creates an isolated Python 3.12 environment, downloads MT50 and
the fine-tuned PI0.5 checkpoint, creates the compatibility view, and extracts
the **64-token (8×8) spatial mean-pooled cache** from section 6. Ignored local
datasets, weights, environments, outputs and credentials are excluded from
the source transfer. Git history is not copied. Downloads occur on the instance.

Setup runs in the background and survives disconnection. The launcher prints
the log path; inspect progress with:

```bash
brev exec primitive-pink-bovid 'cat /data/vla-corrector-experiments/outputs/brev/setup.status; tail -n 30 /data/vla-corrector-experiments/outputs/brev/setup-*.log'
```

Wait for `setup.status` to say `complete` before starting training. The separate
training launcher defaults to Original Corrector; select another mode or train
all three sequentially:

```bash
bash scripts/brev/train.sh original
bash scripts/brev/train.sh prefix
bash scripts/brev/train.sh uncertainty
# Alternatively, train all three in sequence:
bash scripts/brev/train.sh all
```

Each invocation starts a background job on the instance, with timestamped logs
and a `train-<mode>.status` file under `outputs/brev/`. Run one invocation at a
time. Setup and training share a GPU lock. Existing training outputs follow the
trainers' overwrite protection; choose a new output for repeated experiments:

```bash
bash scripts/brev/train.sh prefix --set training.output=outputs/experiments/metaworld_prefix_pool64_repeat
```

Both local launchers accept `--instance NAME` and `--remote-dir /absolute/path`.
Setup also accepts `--wait-seconds 900` and `--foreground`. Inside the instance,
`bash scripts/brev/train.sh --local prefix` runs training in the foreground.
Calibration and evaluation remain separate, as described in section 6.

After preparing the working evaluation policy/tokenizer compatibility view on
the instance, the uncertainty workflow can run all stages in sequence:

```bash
bash scripts/brev/uncertainty.sh --after-prefix
```

To evaluate Prefix before starting uncertainty training, use this instead:

```bash
bash scripts/brev/uncertainty.sh --eval-prefix-first
```

This waits for Prefix training and evaluates its best checkpoint on all 50
tasks (20 episodes each), then starts the uncertainty stages below. A Prefix
evaluation failure stops the workflow before uncertainty training begins.

This queues behind a successful Prefix run, then trains the pooled uncertainty
model, collects 10 episodes per MT50 task with seed 10000 and videos disabled,
calibrates from successful episodes, and evaluates 20 episodes per task with
seed 1000 and videos enabled. Calibration comes **after training** and needs
at least 50 successful episodes. Collection observes the frozen policy without
triggering recovery; evaluation uses the calibrated alarm. All rollout stages
use `pretrained_weights/pi05_metaworld_eval_compat`, including its local tokenizer,
so their policy and protocol identities match. Calibration also checks the error
head: if its correlation is too weak, evaluation uses ordinary replanning with
OGG disabled. Existing outputs are preserved and rejected on a fresh launch.

The queued workflow survives disconnection, releases the GPU to the Prefix job
while waiting, and stops if any stage fails. Check
`outputs/brev/uncertainty-workflow.status` and its timestamped log for progress.
Omit `--after-prefix` to start immediately, or use `--through train` /
`--through calibrate` to stop after that stage. The workflow shares the GPU lock
with the other Brev jobs; `--instance` and `--remote-dir` are supported too.

The cloud environment uses the repository's `metaworld` and `pi` extras with
the core pins in `scripts/brev/constraints.txt`. It uses a checksum-verified
[Miniforge installer](https://github.com/conda-forge/miniforge) and FFmpeg 7.
TorchCodec 0.5 is paired with PyTorch 2.7.1 according to the
[official compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).
This avoids installing every unrelated simulator and optional CUDA extension
from the full workstation `environment.yml`. The resolved package list is saved
to `outputs/brev/installed-requirements.txt`.

Allow **at least 160 GB free** for setup, including the datasets library's
intermediate Arrow copy; use the instance's data disk. Re-running setup resumes
Hugging Face downloads and validates/reuses a completed cache. Extraction itself
cannot resume: an incomplete cache is preserved with an `.incomplete.*` suffix
and extraction starts again. A completed cache with mismatched metadata is
rejected. The scripts do not start training automatically.

## 8. What must not be committed to Git

Do not commit these local artifacts:

```text
.cache/
datasets/
pretrained_weights/
outputs/
third_party/
wandb/
```

They are already excluded by `.gitignore`.
