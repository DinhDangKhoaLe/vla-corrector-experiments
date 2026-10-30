# Fixed-K uncertainty Corrector

`configs/corrector/uncertainty.yaml` is an opt-in experiment. Original baseline
and prefix-only YAMLs, model paths and outputs retain their behavior. Existing
supported checkpoints remain loadable and saved results are retained. The `train` file
contains commands for the active experiments and PACE.

## Training

Use the existing normalized-action latent cache. No re-extraction, manual labels,
multi-horizon expansion or VLA fine-tuning is required. For fixed K=10, input is
the current latent plus the exact ten ordered normalized actions, and the target
is `z[t+10] - z[t]`. The prefix encoder and residual MLP are the existing ones.
The only additional network is the small expected-RMSE head.

The training objective is:

```
cosine_loss_weight * (1 - cosine(predicted_delta, target_delta))
+ mse_loss_weight * mean((predicted_delta - target_delta)**2)
+ uncertainty_loss_weight * SmoothL1(predicted_log_rmse, log(actual_rmse))
```

The RMSE target and the head inputs are detached. Only the error head learns from
the third term; the first two train the latent predictor. This expected-error
estimate is not a Gaussian standard deviation or a task-failure probability.
Best checkpoint selection uses validation cosine+MSE (with the configured
weights); `last_model.pt` is saved each epoch. All component losses are logged in
`metrics.jsonl`. Defaults are at most 30 epochs with patience 5.

## Monitoring and recovery

E and K are both fixed at 10, matching the existing prefix-only setup. There is
no adaptive-E selector. The runtime schedules forecasts only while at least K
actions remain in the queue, so one forecast matures per chunk, at its normal
replan boundary. With persistence=2, the earliest confirmed alarm is step 20.
An alarm can enable OGG for that boundary's new plan; it does not interrupt the
already completed ten-action prefix.

At each matured forecast the score is:

```
actual_rmse / max(predicted_rmse, rmse_floor)
```

Do not divide cosine error by predicted RMSE: they describe different errors.
Raw RMSE is also checked against its own calibrated threshold, so an inflated
error estimate cannot conceal every mismatch. Replanning needs consecutive high
scores; direction coherence is required only for OGG, never to permit an alarm.

After an alarm the queue and invalid pending forecasts are cleared. Ordinary
replanning is the default. One OGG replan is allowed only when the normalized
score is high, expected RMSE is below its calibrated cap, consecutive residual
directions agree, the error head passed its correlation diagnostic, and guidance
is enabled. The existing OGG objective and gradient implementation are reused.
Cooldown prevents repeated events; low scores support release and the monitor
rearms after cooldown. Logs include raw/normalized scores, predicted error,
direction coherence, alarms and OGG eligibility in `experiment_episodes.jsonl`.

This gate is an experimental policy. Correlation and low expected error do not
prove that an OGG gradient is useful after a disturbance. Compare against the
normal-replan-only ablation before drawing that conclusion.

## Commands

Run these sections in order, stopping if a command fails. Do not run the entire
`train` reference file as a shell script: it also includes other experiments.

```bash
cd /home/student/Public/Augustine/vla-corrector
conda activate lerobot-dynamicVLA
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml --mode train

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml --mode collect-calibration \
  --set evaluation.episodes=5 --set evaluation.save_video=false

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml --mode calibrate

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml --mode evaluate
```

Collection is 5 episodes per task across 50 tasks with seed 10000. The frozen
policy executes normally; monitoring logs are collected without alarm-triggered
interventions. Calibration requires at least 50 successful episodes. If fewer
are available, collect a fresh larger set using a new output folder and update
`uncertainty.traces` to that folder's `experiment_episodes.jsonl`.

Calibration computes empirical 95th percentiles of successful-episode maximum
normalized and raw errors, plus a predicted-error cap for OGG. It also checks
error-head correlation; if that diagnostic fails, OGG is disabled but normal
replanning remains available. No separate validation run is needed, and these
empirical thresholds have no formal coverage guarantee. Demonstration-only calibration would have a different
distribution from policy rollouts, so this recipe uses independent policy runs.

Evaluation is 20 episodes per task (1000 total), seed 1000. Raw videos are enabled
for all episodes. Outputs are `outputs/experiments/eval_uncertainty`; training is
`outputs/experiments/uncertainty`. Checkpoint/protocol fingerprints and task/seed
overlap checks prevent stale calibration or reuse of calibration episodes for
evaluation. Terminal observations are included in traces. Changing K, E, policy,
score parameters, environment or monitor implementation requires recalibration;
changing the video flag, episode count or output location does not.

## Options and comparisons

Edit the YAML before training/calibration. Loss weights affect only this mode.
`model.horizons: [10]` sets K. `evaluation.n_action_steps: 10` sets E, which must
be at least K. `evaluation.guidance_eta` controls existing OGG strength.
`uncertainty.persistence_checks`, `direction_coherence_min`, `episode_quantile`
and `ogg_uncertainty_quantile` set monitoring/gating behavior; they need a fresh
calibration after changes. More permissive values are not proven to improve
success and should be evaluated at matched policy-call budgets.

Normal replanning only, reusing the same artifact and independent output:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/uncertainty.yaml --mode evaluate \
  --set uncertainty.ogg_enabled=false \
  --set evaluation.output=outputs/experiments/eval_uncertainty_no_ogg
```

Repeat evaluation of the existing prefix checkpoint at the same E=10 (no retraining):

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/prefix.yaml --mode evaluate \
  --set evaluation.output=outputs/experiments/eval_prefix_e10_repeat
```

Both YAMLs use E=10. Report success, VLA calls, OGG calls and wall time together.
Ordinary replanning still incurs a VLA call.
