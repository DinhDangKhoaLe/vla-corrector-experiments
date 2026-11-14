# Official clean-only RoboTwin π0.5 checkpoint

The default download is now the **RoboTwin team's official co-trained π0.5**,
replacing `Myungkyu/pi05_robotwin_vanilla_b128_60k` in the download plan.
Existing local files are preserved. No community checkpoint is substituted.

| Item | Official release |
|---|---|
| Hugging Face repository | `TianxingChen/RoboTwin2.0` (**dataset** repository) |
| Pinned checkpoint revision | `a967b852afa21a9cbf19a198f7e653109042e87c` |
| Archive | `cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz` |
| Archive size | 12,443,100,901 bytes, about 12.44 GB |
| Fine-tuning protocol | 50 `demo_clean` trajectories × 50 tasks = 2,500 demonstrations |
| Embodiment | ALOHA-AgileX, joint control |
| Training dataset name | `RoboTwin_lerobot_v30` |
| Initial weights | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Training | 30,000 steps, final step 29999, global batch 256, seed 0 |
| Model action horizon | 50; padded model action dimension 32 |

The clean-only statement concerns RoboTwin fine-tuning; the π0.5 backbone has
broader pretraining. Clean→random results refer to testing on randomized scenes,
not training on randomized demonstrations.

Sources: [official checkpoint](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/cotrain_ckpt/Pi_05),
[official benchmark protocol](https://robotwin-platform.github.io/leaderboard),
[maintainer's training configuration](https://github.com/RoboTwin-Platform/RoboTwin/issues/539).

## Download and unpack explicitly

Run from the repository root in `dynamic_horizon`:

```bash
# Preview only: no download.
python scripts/download_dynamic_horizon.py --benchmark robotwin --robotwin-tasks all

# Download the official checkpoint plus only clean demonstration archives.
python scripts/download_dynamic_horizon.py \
  --benchmark robotwin --robotwin-tasks all --execute --unpack
```

The official checkpoint is pinned separately from the demonstration repository.
Demonstration downloads follow `main`; an exact historical data revision is not
claimed. The maintainers identify `RoboTwin_lerobot_v30` as the training corpus;
the source repository also distributes it under
`lerobot_dataset/RoboTwin_lerobot_v30.zip`. That additional ZIP is not downloaded
by default. Current clean HDF5 archives may have dataset fixes after training.

Paths:

```text
pretrained_weights/robotwin_official/
  cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz
  extracted/                    # original OpenPI contents; source_archive.json records provenance
datasets/robotwin_archives/     # selected aloha-agilex_clean_50.zip archives
datasets/robotwin/              # unpacked HDF5 demonstrations
```

`--robotwin-tasks all` selects all task archives; its default selects only
`adjust_bottle`. The checkpoint remains one co-trained 50-task model either way.
Original archives are retained. Checkpoint extraction rejects unsafe tar members
and reuses an extraction only when its provenance marker matches.

## Required conversion before our training/evaluation commands

**The official OpenPI→LeRobot converter is not implemented in this repository.**
This is an explicit remaining prerequisite for RoboTwin. Downloading or unpacking
the official archive does not produce a runnable LeRobot policy. The downloader
does not call `make_pi05_compat.py` on that archive.

Our Corrector extractor and native evaluator require:

```text
pretrained_weights/pi05_robotwin_official/
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
  ... normalization state referenced by those processors ...
```

A validated conversion must preserve the official weights, vision projector,
three camera mappings, action/state transforms, and normalization statistics.
Check the official ALOHA transforms, not just weight tensor names: the model has
padded 32-D actions while the robot has 14 joint/gripper coordinates.
Do not copy the community model's config or statistics into this export.

After a converter has produced and validated that export, create the zero-copy
compatibility view:

```bash
python scripts/make_pi05_compat.py \
  --source pretrained_weights/pi05_robotwin_official \
  --output pretrained_weights/pi05_robotwin_official_compat
```

This command patches config compatibility only; **it does not convert OpenPI
weights or implement action transforms**. The launcher/extraction wrapper stop
with this prerequisite when the converted weights are absent.

After conversion, follow the RoboTwin sections of [training](../training.md) and
[evaluation](../evaluation.md). Re-extract and retrain the Corrector whenever
changing the policy checkpoint. The previous community model's latent caches,
Corrector, and action normalization must not be reused as if they came from the
official checkpoint. RoboTwin LACO rollout integration is also not implemented;
the native recipes cover baseline, Corrector + OGG, and PACE only.
