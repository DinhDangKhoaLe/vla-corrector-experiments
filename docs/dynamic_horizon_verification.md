# Dynamic-horizon verification — 11 November 2026

Continued the saved implementations from `dev_others` on `dev_other` and
checked the pinned author sources linked in
[the source notes](dynamic_horizon_sources.md).

The continuation fixes AutoHorizon's third-denoising-step indexing, loads legacy
π0.5 compatibility configs without rewriting the checkpoint, and skips BCP
groups where every rollout fails. BCP checkpoints now record `optimized_groups`
and cannot be evaluated before receiving an update. PACE calibrations can be
moved while their algorithm settings remain checked.

## Environment and commands

The real-model checks used the local MetaWorld π0.5 checkpoint, Quadro RTX 8000
(48 GB), PyTorch 2.10.0+cu128, Transformers 5.4.0, MuJoCo 3.6.0, MetaWorld 3.0.0,
and SentencePiece 0.2.3. The five configs retain bfloat16, emulated on this GPU.

```bash
conda activate lerobot-dynamicVLA
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl
python -m unittest discover -s tests/dynamic_horizon -v
OMP_NUM_THREADS=4 python -m dynamic_horizon.smoke --output outputs/dynamic_horizon/verified_20261111
```

The recorded directory already exists; use a new output directory to repeat
the smoke command. Training/calibration and full evaluation commands are in
[training.txt](../training.txt) and [evaluation.txt](../evaluation.txt).

## Results

- All 12 regression tests passed; compilation and `git diff --check` passed.
- All five configs resolve 50 tasks, seed 1000, prediction horizon 50, and
  maximum execution horizon 50.
- PACE calibrated all 2,500 demonstrations across 50 tasks. The reusable local
  artifact is `outputs/dynamic_horizon/pace/calibration.json`.
- The real sampler matched native π0.5 normalized predictions exactly for the
  same input/noise. Attention and BCP feature shapes passed; conditional action
  anchors were exact; π0.5 received no gradients.
- All five methods completed one 55-step `reach-v3` smoke episode at seed 1000.
  None succeeded within this deliberately short limit. These runs verify
  execution and logging, not task performance.
- BCP completed two training groups and checkpoint resume with 100-step limits.
  Group one was entirely unsuccessful and correctly skipped. Group two had a
  successful reference, six successful adaptive rollouts and one failed
  adaptive rollout; it produced one optimized group (two update epochs).
  Head weights changed, and reloading the saved head for evaluation passed.
- A separate check reset all 50 tasks at seeds `1000 + task_index`, took three
  intervening actions and repeated each reset. All physical fingerprints and
  agent states matched exactly. Rendered RGB values differed by at most one
  uint8 intensity level, so bitwise image/rollout reproducibility is not claimed.

Raw local reports (ignored by Git):

- `outputs/dynamic_horizon/verified_20261111/verification.json`
- `outputs/dynamic_horizon/verified_20261111/bcp_training/training.jsonl`
- `outputs/dynamic_horizon/physical_reset_verification_20261111.json`

The BCP smoke head is isolated from the production artifact. Production BCP
training and the 50-task × 20-episode evaluations have not been run.
