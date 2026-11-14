# Training and calibration

Run from the repository root after [setup](docs/setup.md):

```bash
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh
```

Both MetaWorld and LIBERO extraction/training configurations are included. On
this machine, the MetaWorld pool64 cache and
`outputs/experiments/metaworld_original_pool64/h1_k10/best_cosine_model.pt`
already exist. Reuse them; skip the two MetaWorld Corrector commands below unless
you deliberately select fresh cache and training output directories. The LIBERO
video-backed dataset is decoded through the LeRobot reader automatically.
These local artifacts are ignored by Git and will not exist in a fresh H100
clone. Use [the H100 guide](docs/h100_setup.md) to build them on the new machine.

The pretrained π0.5 baseline stays frozen. Neither LACO nor VLA-Corrector
updates its policy weights. Baseline and PACE need no policy training.

## LACO (MetaWorld and LIBERO)

Both modes share one trained model and one held-out calibration artifact:

```bash
python -m laco.extract --config configs/laco/metaworld_pi05.yaml
python -m laco.train --config configs/laco/metaworld_pi05.yaml
python -m laco.calibrate --config configs/laco/metaworld_pi05.yaml

python -m laco.extract --config configs/laco/libero_pi05.yaml
python -m laco.train --config configs/laco/libero_pi05.yaml
python -m laco.calibrate --config configs/laco/libero_pi05.yaml
```

PCA is fit on training episodes only. Validation and calibration episodes are
separate. Calibration estimates the one-step EWMA residual threshold. The two
MetaWorld mode configs are `configs/laco/LACO_replan.yaml` and `LACO_adjust.yaml`;
LIBERO uses `configs/laco/libero_LACO_replan.yaml` and `libero_LACO_adjust.yaml`.
No uncertainty model or ensemble is required. Action correction affects XYZ,
preserving the gripper (and LIBERO rotation), and tries the next two queued actions. If the solve is
unreliable or violates bounds, adjustment mode falls back to replanning.

## VLA-Corrector: keep 64 mean-pooled tokens per camera

```bash
bash scripts/extract_corrector_pool64.sh metaworld
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml --mode train
```

This retains the legacy cosine training recipe, K=10 and one-action conditioning.
The 16×16 SigLIP token grid is mean-pooled to 8×8 **per camera**. It is 64 tokens
per active camera, not a single 64-dimensional vector. No offline threshold
calibration is required for the original online median/MAD monitor. OGG can be
disabled at evaluation without retraining.

Existing cache/checkpoint paths are preserved; training refuses to overwrite an
existing run. To retrain, choose a fresh output:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml --mode train \
  --set training.output=outputs/experiments/metaworld_pool64_new
```

For LIBERO, after downloading:

```bash
bash scripts/extract_corrector_pool64.sh libero
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/libero_original_pool64.yaml --mode train

```

For RoboTwin, first complete the [official checkpoint conversion requirements](docs/robotwin_official.md).
Downloading/unpacking the OpenPI archive is insufficient. These commands require
the validated LeRobot export at `pretrained_weights/pi05_robotwin_official_compat`:

```bash
bash scripts/extract_corrector_pool64.sh robotwin --robotwin-tasks adjust_bottle
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_original_pool64.yaml --mode train
```

The caches use the selected checkpoint's action representation. The downloaded
`pi05_libero_base` has empty normalizer feature/stat maps, so its shipped
processors apply identity normalization. Both LIBERO Corrector and LACO extraction
therefore retain raw demo actions. MetaWorld retains its existing quantile
normalization. If you change checkpoints, verify their actual processor artifacts
before reusing caches or trained correctors. LIBERO image keys remain
`observation.images.image` and `observation.images.image2`, matching that policy.
RoboTwin uses all
three cameras (192 pooled tokens) and 14-D actions. LIBERO uses its two cameras
(128 tokens) and 7-D actions. Keep the extraction policy and evaluation policy
identical for each corrector.

## PACE calibration

```bash
python -m siglip_dynamics.pace calibrate --config configs/pace/metaworld_pi05.yaml
python -m siglip_dynamics.pace calibrate --config configs/pace/libero_pi05.yaml
python -m siglip_dynamics.pace calibrate --config configs/pace/robotwin_pi05.yaml
```

Run only commands for datasets already downloaded. No VLA/latent cache is
loaded: calibration reads decoded actions directly. For each task, use up to 50
eligible training demonstrations, Savitzky–Golay 11/order 3 smoothing, valley
spacing 10, and the fifth percentile of positive valley-segment contrasts.
Following the official code, multi-arm calibration uses the more active arm;
online selection takes the earliest accepted boundary across both arms.
Native RoboTwin calibration reads the HDF5 `state` joint fields, matching the
released observed-joint-state calibration; online selection uses decoded targets.
Grippers are excluded. RoboTwin uses absolute-joint differences; MetaWorld and
LIBERO use XYZ delta magnitude. This is a Cartesian adaptation of PACE, not a
claim to reproduce the paper's MetaWorld results (it did not evaluate MetaWorld).

No evaluation successes or fixed-horizon sweeps are used to tune thresholds.
Tasks with no positive training contrasts raise an error for review. Missing
task thresholds also raise, rather than silently using another task's threshold.
Calibration writes a new `pace_v2` artifact and refuses to overwrite one.
