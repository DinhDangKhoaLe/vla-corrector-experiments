# RoboTwin uncertainty-Corrector evaluation

`siglip_dynamics.experiments.run` now detects a RoboTwin cache and routes
`collect-calibration` and `evaluate` to the native SAPIEN runner instead of
the MetaWorld/MuJoCo evaluator.  It preserves the extracted-cache contract:

- RGB camera order: head, left wrist, right wrist;
- native RoboTwin 14-D joint-position actions (`qpos`);
- PI0.5 checkpoint action postprocessing before `task_env.take_action`;
- policy-space action normalization for Corrector conditioning;
- fixed-K latent monitoring, persistence alarm and queue-clearing normal
  replans; and
- per-episode traces, summary CSV/JSON and optional head-camera MP4 files.

## Required runtime

The current `lerobot-dynamicVLA` environment is deliberately not treated as a
RoboTwin simulator environment: it lacks SAPIEN, MPLib and TOPPRA.  Before a
native rollout, activate an environment containing the official RoboTwin
simulator requirements and make this workspace source importable.  Confirm
the critical imports before launching a long job:

```bash
python - <<'PY'
import sapien, mplib, toppra
from siglip_dynamics.experiments.robotwin_eval import evaluate_robotwin
print("RoboTwin simulator and Corrector runner are importable")
PY
```

RoboTwin's published requirement file pins NumPy 1.26 and its simulator stack,
whereas the current LeRobot workspace declares NumPy 2.x.  Use a dedicated
RoboTwin+PI0.5 environment rather than downgrading the active MetaWorld
environment in place.

## Smoke collection

This validates one episode per configured task without writing videos.  It is
not enough data to calibrate a threshold.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode collect-calibration \
  --set evaluation.episodes=1 \
  --set evaluation.save_video=false \
  --set evaluation.output=outputs/experiments/smoke_robotwin_uncertainty
```

## Calibration and evaluation

The configured 15-task suite has 50 successful calibration episodes as the
minimum.  Ten attempted episodes per task gives up to 150 traces; increase it
if fewer than 50 complete successfully.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode collect-calibration \
  --set evaluation.episodes=10 \
  --set evaluation.save_video=false

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode calibrate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode evaluate
```

`ogg_enabled` remains `false` for RoboTwin.  The runner performs calibrated
normal replans (clear queued actions and request a fresh PI0.5 chunk); it does
not claim to provide differentiable OGG for the dual-arm policy.

The legacy/original Corrector evaluator remains intentionally unsupported in
the native runner: its separate MAD/OGG state machine has not been ported.
