# Evaluation commands

```bash
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh
```

## Complete ordered run with videos and task metrics

```bash
HORIZON=50 EPISODES=20 SEED=1000 \
  bash scripts/run_metaworld_then_libero.sh
```

The script reuses existing models/caches, builds missing MetaWorld artifacts,
evaluates every MetaWorld mode, then extracts/trains/calibrates LIBERO and
evaluates every LIBERO mode. The modes are baseline, LACO_replan, LACO_adjust,
Corrector with OGG, and PACE. Corrector without OGG is excluded from the pipeline.
It stops on any failed command.
Both LACO modes share their training/calibration. It does not download data or
policy weights.

On this Linux workstation the script starts the batch in its own user systemd
scope, with `MemoryHigh=36G`, `MemoryMax=44G`, and `MemorySwapMax=512M`.
This isolates it from the terminal's process group. MetaWorld Parquet loading
reads one row group at a time; it does not build a second Arrow image cache.
LACO's metadata/action pass reads numeric columns without decoding images.
π0.5 restores checkpoint tensors into an empty model before moving it to the
GPU, avoiding a second full random model allocation in host RAM. Missing or
incompatible pretrained weights now stop the run instead of returning an
uninitialized policy.

Each mode runs 20 episodes **per task** and saves one full rollout video per
episode. That is 1,000 videos per MetaWorld mode and 800 per LIBERO mode.
Environment seeds are 1000–1019 for each task across all methods. Change
`HORIZON` to 40, 30, 20, or 10 to change the execution cap; prediction stays 50.
Pass a fresh output root as the script's first argument if desired:

```bash
bash scripts/run_metaworld_then_libero.sh outputs/eval/my_full_run
```

To continue an interrupted run, pass its existing output root with the same
horizon, episode count, and seed. Completed evaluations with all task metrics
and videos are skipped; incomplete evaluation folders are preserved with an
`.incomplete_<timestamp>` suffix before restarting that method. Existing console
logs are appended. Extraction/training/calibration artifacts are reused.

Results are under `outputs/eval/metaworld_then_libero_<timestamp>/` by default:

- `logs/`: console logs for extraction, training, calibration, and evaluation.
- `<benchmark>/<method>/videos/`: one MP4 per episode.
- `<benchmark>/<method>/per_task_metrics.csv`: task group/id, episode count,
  success rate in percent, total policy calls, policy calls per episode, total
  executed actions, and average horizon.
- `<benchmark>/<method>/per_episode_metrics.csv`: success, steps, and calls for each episode.
- `<benchmark>/<method>/metrics_summary.json`: overall metrics and saved-video count.

Rollouts display one compact progress bar per episode, refreshed in place.
The log collector preserves ordinary logs/errors and only the final progress
line of each episode, rather than recording every carriage-return refresh.

Average horizon means **total executed actions / total policy calls**, including
chunks shortened by termination. It is not the average planned chunk length.
Every launcher command below accepts `--save-videos` to save all its episodes,
and automatically produces these common metric files after evaluation.
To regenerate a report from existing native results:

```bash
python -m dynamic_horizon.report outputs/eval/my_full_run/metaworld/baseline
```

MetaWorld MT50: all 50 tasks, 20 episodes each, seed 1000; prediction horizon 50.
Every method below uses maximum execution horizon 50:

```bash
python -m dynamic_horizon.evaluate --method baseline
python -m dynamic_horizon.evaluate --method LACO_replan
python -m dynamic_horizon.evaluate --method LACO_adjust
python -m dynamic_horizon.evaluate --method corrector
python -m dynamic_horizon.evaluate --method pace
```

Train/calibrate first using [training.md](training.md). PACE requires a freshly
calibrated `pace_v2` artifact in decoded action units. Add `--dry-run` to any
command to inspect it without loading a simulator/model. All methods accept
`--horizon 50`, `40`, `30`, `20`, or `10`; the default is **50**. The prediction
chunk remains **50**. Baseline executes the selected fixed horizon; dynamic
methods may execute fewer actions before replanning. PACE uses its full 50-step
profile with the requested execution cap. Its existing calibration can be reused.

```bash
# Example: test all methods with maximum execution horizon 30.
for method in baseline LACO_replan LACO_adjust corrector pace; do
  python -m dynamic_horizon.evaluate --method "$method" --horizon 30
done

# Optional sweep: 5 horizons × 5 methods, 20 episodes/task per run.
for horizon in 50 40 30 20 10; do
  for method in baseline LACO_replan LACO_adjust corrector pace; do
    python -m dynamic_horizon.evaluate --method "$method" --horizon "$horizon"
  done
done
```

Output directories include the method and horizon automatically.

For a quick rollout, use `--episodes 1 --output outputs/smoke/<method>`; this is
one episode per task. Choose a fresh output directory for every run.
If you retrained Corrector in a different directory, use the config wrapper:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml --mode evaluate \
  --set evaluation.checkpoint=outputs/experiments/metaworld_pool64_new/h1_k10 \
  --set evaluation.output=outputs/eval/corrector_pool64_new_h50
```

LIBERO (40 tasks across Spatial/Object/Goal/Long, 20 episodes/task, seed 1000):

```bash
python -m dynamic_horizon.evaluate --benchmark libero --method baseline
python -m dynamic_horizon.evaluate --benchmark libero --method LACO_replan
python -m dynamic_horizon.evaluate --benchmark libero --method LACO_adjust
python -m dynamic_horizon.evaluate --benchmark libero --method corrector
python -m dynamic_horizon.evaluate --benchmark libero --method pace
```

Native RoboTwin (default `adjust_bottle` task, 20 episodes, seed 1000):

These commands require unpacked demonstrations/assets and a validated LeRobot
conversion of the official clean-only π0.5 archive at
`pretrained_weights/pi05_robotwin_official_compat`. Conversion is not implemented
here; see [official checkpoint prerequisites](docs/robotwin_official.md).
The original OpenPI archive cannot be used as `--policy` directly.

```bash
python -m dynamic_horizon.evaluate --benchmark robotwin --method baseline
python -m dynamic_horizon.evaluate --benchmark robotwin --method corrector
python -m dynamic_horizon.evaluate --benchmark robotwin --method pace
```

These additional benchmark recipes require fresh simulator validation after
the environment and assets are installed, including on H100; see
[the H100 migration guide](docs/h100_setup.md). LACO supports MetaWorld and LIBERO;
both LIBERO modes preserve rotation/gripper actions and share one trained XYZ model.
The RoboTwin commands dispatch to its native SAPIEN runner.
For another task, set `evaluation.robotwin_tasks` in its baseline/Corrector
configs and calibrate PACE on demonstrations for that task.

Metrics: policy calls/episode counts each generated chunk; average executed
actions is total environment actions divided by total policy calls. Final
episode termination and detector truncation both shorten a chunk. Policy-call
time excludes the separate LVM/LACO monitoring work. Corrector OGG calls have
their own event CSVs under `analysis/episode_guidance_events/`; final summaries
include guidance request and applied-step counts.
