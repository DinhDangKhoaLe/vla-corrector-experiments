# Dynamic horizon experiments with π0.5

Supported MetaWorld methods:

| Method | Execution control | Training / calibration |
|---|---|---|
| Original π0.5 baseline | Fixed 10 or 50 actions | Frozen pretrained policy |
| LACO_replan | Detect visual residual, discard queue, replan | Local Jacobian model + held-out threshold |
| LACO_adjust | Detect, adjust next two queued actions; replan when guards fail | Same model and threshold |
| VLA-Corrector | LVM truncation + optional OGG on next policy call | 8×8 spatial mean pool: 64 tokens per camera, K=10 |
| PACE | Select earliest accepted motion valley before execution | Per-task demonstration calibration; no policy training |

Default evaluation: seed **1000**, **20 episodes/task**, prediction chunk **50**,
maximum execution horizon **50**. All methods accept
`--horizon 50|40|30|20|10` for execution; prediction chunks remain 50.
LACO has no uncertainty mode in the supported interface.

## Start

```bash
bash scripts/setup_dynamic_horizon.sh
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh
python scripts/check_dynamic_horizon.py
```

The setup command installs the environment only when you run it. Model/data
downloads are separate, explicit commands and have not been run as part of this
update. The supported platform is Linux x86_64 with an NVIDIA GPU.

- [Environment, RoboTwin, LIBERO and download setup](docs/setup.md)
- [Fresh H100 machine setup and run sequence](docs/h100_setup.md)
- [Official clean-only RoboTwin checkpoint and conversion status](docs/robotwin_official.md)
- [Training and calibration commands](training.md)
- [Evaluation commands](evaluation.md)

Each method has its own config directory in `configs/` and separate code modules.
RoboTwin/LIBERO have environment, download, extraction and PACE calibration
recipes. LIBERO and native RoboTwin baseline/Corrector/PACE runners are wired up;
they require their assets and checkpoints before rollout.
LACO's live evaluator supports MetaWorld and LIBERO. RoboTwin downloads now select
the RoboTwin team's official clean-only π0.5 OpenPI archive. Its conversion to
this runner's LeRobot format remains required and is not implemented here;
downloading the archive alone does not enable RoboTwin rollout. No community
checkpoint is substituted automatically. See the official checkpoint guide.

The upstream LeRobot source and internal legacy checkpoint loaders remain because
the supported methods depend on them. Old prefix/uncertainty/BBBB experiment
configs and orchestration scripts were removed. Existing `datasets/`,
`pretrained_weights/`, `outputs/` and result reports are retained.

PACE's joint algorithm and calibration follow the [released implementation](https://github.com/niejunnan25/PACE).
For MetaWorld/LIBERO Cartesian-delta policies we use translation magnitude,
excluding grippers. Calibration reads decoded demo actions directly, without a
latent cache, policy normalization, or evaluation labels. Recalibrate old
`pace_style_v1` artifacts; the updated selector requires `pace_v2`.
