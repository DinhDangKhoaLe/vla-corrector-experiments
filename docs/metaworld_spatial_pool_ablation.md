# MetaWorld spatial mean-pooling ablation

This is an optional representation ablation for the original Corrector,
action-prefix Corrector, and uncertainty Corrector. Existing YAML files,
full-token caches, checkpoints and output folders remain separate.

The current **8×8 = 64×2048** workflow uses the `pool64` configs and is documented
directly in [README section 6](../README.md#6-spatial-mean-pooling-extract-train-and-evaluate-all-corrector-modes).
The 32-token (`pool32`) workflow remains in section 6 below. Sections 1–5
document the earlier 16-token (`pool4`) variant.

The new configs use `model.image_token_grid_size: 4`. PI0.5's `16×16` visual
patch grid is averaged in non-overlapping `4×4` blocks to produce a `4×4`
grid. Pooling is applied to each camera before concatenation and before
per-token int8 quantization. The feature width stays 2048.

| Representation (one camera) | Latent shape | Estimated MT50 frame cache |
| --- | --- | --- |
| Existing full grid | `256 × 2048` | 107.5 GB |
| Current `8×8` grid (`pool64`) | `64 × 2048` | **26.87 GB** |
| Earlier `4×8` grid (`pool32`) | `32 × 2048` | 13.44 GB |
| New `4×4` ablation | `16 × 2048` | 6.72 GB |
| Global mean (more aggressive) | `1 × 2048` | 0.42 GB |

Estimates use 204,806 frames, int8 features, float16 per-token scales,
float16 four-dimensional actions and int32 episode indices. They exclude
the original dataset, weights, temporary dataset caches and evaluation videos.

The `4×4` grid is a starting experiment, not a proven accuracy optimum. It
retains coarse spatial layout while giving a 16-fold reduction in token data
and in token-wise MLP arithmetic. Parameter counts stay approximately unchanged;
the MLP shares weights across tokens. Total runtime does not scale by 16: the
frozen PI0.5 vision encoder and policy inference still process full images.
Small-object detail can be lost, so compare success rate and wall time against
the corresponding full-token experiment.

## 1. Environment

Complete the installation and dataset/checkpoint downloads in [README.md](../README.md)
first, then run from the repository root:

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

## 2. Extract the separate pooled cache once

All three new configs share this cache. Use a fresh output path: extraction
writes its array files. Do not point this command at your full-token cache.

```bash
python -m siglip_dynamics.extract \
  --dataset-path ./datasets/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --image-key observation.image \
  --no-use-wrist-image \
  --action-key action \
  --output-path ./outputs/metaworld_mt50_pi05_latents_pool4 \
  --encoder-backend pi05 \
  --encoder-policy-path ./pretrained_weights/pi05_metaworld_compat \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --image-token-grid-size 4 \
  --max-output-gb 8 \
  --device cuda
```

Expected cache: `L=16`, `D=2048`, action dimension 4. Extraction still encodes
all dataset frames with the frozen backbone; pooling primarily reduces storage
and subsequent Corrector work.

## 3. Train the three variants

Run each command after the preceding command completes successfully.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool4.yaml \
  --mode train

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool4.yaml \
  --mode train

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool4.yaml \
  --mode train
```

The original variant keeps the legacy trainer, current-action conditioning,
sample-level train/validation split, cosine loss and MAD/OGG evaluation. Prefix
and uncertainty keep their existing experimental trainers and episode splits.
Each variant retains its own existing optimizer, batch size, losses, `h=1`,
`K=10`, execution cap `E=10` and predicted chunk size 50. This is a within-variant
pooling ablation; the legacy and experimental split protocols are still different.

Checkpoints are saved separately:

```text
outputs/experiments/metaworld_original_pool4/h1_k10/best_cosine_model.pt
outputs/experiments/metaworld_prefix_pool4/best_model.pt
outputs/experiments/metaworld_uncertainty_pool4/best_model.pt
```

## 4. Calibrate only the new uncertainty model

Mean pooling changes the residual distribution, so use a new calibration
collection and artifact. Do not reuse full-token uncertainty thresholds.
Ten episodes per task means 500 collection episodes across MT50, not ten total.
The existing calibration rule requires at least 50 successful episodes.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool4.yaml \
  --mode collect-calibration \
  --set evaluation.episodes=10 \
  --set evaluation.save_video=false

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool4.yaml \
  --mode calibrate
```

## 5. Evaluate all tasks

All three commands use the existing task groups, 20 episodes per task,
300 maximum environment steps, seed 1000, and video saving.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool4.yaml \
  --mode evaluate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool4.yaml \
  --mode evaluate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool4.yaml \
  --mode evaluate
```

Separate result directories:

```text
outputs/experiments/eval_metaworld_original_pool4/
outputs/experiments/eval_metaworld_prefix_pool4/
outputs/experiments/eval_metaworld_uncertainty_pool4/
```

Choose a new `evaluation.output` using `--set` when rerunning an evaluation.
For calibration recollection, also update `uncertainty.traces` to the new
collection path before running `calibrate`.

Online monitoring reads the pooling setting from the saved Corrector. It pools
the newly observed visual grid exactly as extraction does. The matured forecast,
observed residual and OGG conditioning all have the same pooled token shape.
The PI0.5 backbone receives its usual full-resolution tokens. Existing unpooled
checkpoints default to grid size zero and retain their original representation.

For `8×8`, use the prepared `metaworld_*_pool64.yaml` configs and extraction
command in README section 6. They use `--image-token-grid-size 8` and separate
cache/output paths. Changing the YAML alone cannot convert a cache or
checkpoint. New training checks reject cache/config representation mismatches.

## 6. Earlier 32-token variant: 4 rows × 8 columns

This averages each non-overlapping `4×2` block of the source `16×16` grid,
preserving row-major spatial order. It produces exactly **32×2048** features
per camera and uses a separate cache and checkpoint family. The full MT50
quantized cache is approximately **13.44 GB (12.51 GiB)**, versus 107.5 GB for
the full grid. These sizes exclude dataset downloads, policy weights and videos.

Use the environment exports in section 1, then extract once:

```bash
python -m siglip_dynamics.extract \
  --dataset-path ./datasets/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --image-key observation.image \
  --no-use-wrist-image \
  --action-key action \
  --output-path ./outputs/metaworld_mt50_pi05_latents_pool32 \
  --encoder-backend pi05 \
  --encoder-policy-path ./pretrained_weights/pi05_metaworld_compat \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --image-token-grid-shape 4 8 \
  --max-output-gb 16 \
  --device cuda
```

Leave `--image-token-grid-size` at its default of 0 when using the rectangular
shape. Do not set it to 32: that flag specifies the side of a square grid.

Train all three variants after extraction completes successfully:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool32.yaml \
  --mode train

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool32.yaml \
  --mode train

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool32.yaml \
  --mode train
```

Collect separate calibration episodes and calibrate the new uncertainty model:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool32.yaml \
  --mode collect-calibration \
  --set evaluation.episodes=10 \
  --set evaluation.save_video=false

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool32.yaml \
  --mode calibrate
```

Evaluate all tasks with video saving, 20 episodes per task and seed 1000:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool32.yaml \
  --mode evaluate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool32.yaml \
  --mode evaluate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool32.yaml \
  --mode evaluate
```

Each config has `model.image_token_grid_shape: [4, 8]`. Training outputs are
under `outputs/experiments/metaworld_{original,prefix,uncertainty}_pool32`;
evaluation outputs are under the corresponding `eval_metaworld_*_pool32`
directories. Calibration also has its own `pool32` trace and artifact paths.
All other method settings match their 16-token/full-token counterparts.
