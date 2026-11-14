# Setup and downloads

Run from the repository root. Installers run only when invoked; model/dataset
downloads require `--execute`. For a new GPU server, start with the
[H100 migration guide](h100_setup.md). The
[official RoboTwin checkpoint guide](robotwin_official.md) describes its required
conversion and the current method coverage.

## One conda environment

Linux x86_64, NVIDIA GPU, a CUDA 12.6-compatible NVIDIA driver, working EGL and
Vulkan are required. These are host dependencies; conda cannot install a kernel
driver. On Ubuntu, install the host libraries if missing:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libgl1 unzip
vulkaninfo --summary

bash scripts/setup_dynamic_horizon.sh
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh
python scripts/check_dynamic_horizon.py
```

The environment is named `dynamic_horizon`. The YAML provides Python 3.12,
FFmpeg, CUDA toolkit/NVCC and build tools. The installer resolves the local
LeRobot extras and `environments/requirements-dynamic-horizon.txt` together.
It resumes an existing environment after a failed installation. CMake 3 and
Setuptools are installed first, then `egl_probe` is built with
`--no-build-isolation`: it cannot build with CMake 4 and does not declare the
CMake Python module needed inside pip's isolated build environment.
The later "No module named torch" error means that the core pip
install did not complete; finish it before running `setup_robotwin.sh`.

The shared profile uses NumPy 1.26.4 because MPLib 0.2.1 requires NumPy <2.
This required lowering this fork's NumPy minimum and allowing Rerun 0.22.1
(newer Rerun requires NumPy 2). OpenCV 4.11, SciPy 1.15, Torch 2.7.1,
TorchVision 0.22.1 and TorchCodec 0.5 are pinned together. It does not install
RoboTwin's old Torch 2.4/Gymnasium 0.29 requirements over LeRobot. This combined
profile has passed import/dependency checks in `dynamic_horizon` on the Quadro
RTX 8000. PyTorch3D 0.7.9 compiled against Torch
2.7.1/CUDA 12.6 and passed a CUDA KNN smoke test; CuRobo's compiled CUDA
extensions also import. Full MetaWorld evaluations have run on that workstation;
H100 rendering and the official RoboTwin archive conversion have not been
validated. Simulator rollouts need their benchmark assets and checkpoint exports.

To resume or repair an existing `dynamic_horizon` environment:

```bash
bash scripts/setup_dynamic_horizon.sh
conda activate dynamic_horizon
python -m pip check
```

## Models and demonstrations

Preview the plan (no network calls or downloads):

```bash
python scripts/download_dynamic_horizon.py --benchmark all --robotwin-tasks adjust_bottle
```

Run these commands yourself when ready. Accept any gated model terms on Hugging
Face first, then log in if required:

```bash
hf auth login
python scripts/download_dynamic_horizon.py --benchmark metaworld --execute
python scripts/download_dynamic_horizon.py --benchmark libero --execute
python scripts/download_dynamic_horizon.py --benchmark robotwin \
  --robotwin-tasks adjust_bottle --execute --unpack

# Optional: all 50 RoboTwin task demonstration archives (large).
python scripts/download_dynamic_horizon.py --benchmark robotwin \
  --robotwin-tasks all --execute --unpack
```

| Benchmark | Demonstrations | π0.5 checkpoint |
|---|---|---|
| MetaWorld | `lerobot/metaworld_mt50` → `datasets/metaworld_mt50` | `tiantianx/pi05_metaworld` |
| LIBERO | `lerobot/libero` → `datasets/libero` | `lerobot/pi05_libero_base` |
| RoboTwin 2.0 | `TianxingChen/RoboTwin2.0`, selected `aloha-agilex_clean_50.zip` | Official `cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz` in the same HF dataset repository |

MetaWorld/LIBERO exports go into `pretrained_weights/pi05_<benchmark>` with a
`_compat` symlink view; original weights remain unchanged. RoboTwin's official
archive is pinned to revision `a967b852afa21a9cbf19a198f7e653109042e87c` and goes
under `pretrained_weights/robotwin_official/`. `--unpack` safely extracts it to
`extracted/` alongside the retained archive. Its fine-tuning uses 50 clean
demonstrations per task across 50 tasks, with horizon 50 and 30,000 training steps.
See the [maintainer's configuration](https://github.com/RoboTwin-Platform/RoboTwin/issues/539).

**Official RoboTwin OpenPI→LeRobot conversion is not implemented here.**
The downloader does not relabel the original archive as a LeRobot export.
RoboTwin extraction/evaluation expects a separately validated conversion at
`pretrained_weights/pi05_robotwin_official_compat`. No community weights are
downloaded as a fallback. See [conversion requirements](robotwin_official.md).

MetaWorld's published training config uses `lerobot/metaworld_mt50`. LIBERO's
checkpoint is already LIBERO-trained; this is the base release, not the later
`pi05_libero_finetuned_v044` benchmark export. Exact training episode revisions
are not asserted for that base checkpoint. Sources: [MetaWorld training config](https://huggingface.co/tiantianx/pi05_metaworld/blob/main/train_config.json),
[LIBERO policy documentation](https://huggingface.co/docs/lerobot/main/en/pi05).

If the MetaWorld dataset/weights already exist locally, repair the old view's
missing `type: pi05` discriminator without downloading again:

```bash
python scripts/make_pi05_compat.py \
  --source pretrained_weights/pi05_metaworld \
  --output pretrained_weights/pi05_metaworld_compat --refresh-config
```

Raw RoboTwin archives are preserved under `datasets/robotwin_archives`; extracted
per-task HDF5 episodes go below `datasets/robotwin/<task>`. The extractor handles
nested directories. Use matching camera/action normalization statistics from
the policy and verify the downloaded embodiment is ALOHA-AgileX.

The existing MetaWorld dataset was used to validate the new calibration: all
50 task thresholds were produced locally at `outputs/pace/metaworld_decoded_v2.json`.
That generated artifact is ignored by Git; a fresh checkout must run calibration.

## RoboTwin simulator

```bash
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh
bash scripts/setup_robotwin.sh

# Explicit large asset download: robot geometry, objects and textures.
(cd third_party/RoboTwin && bash scripts/_download_assets.sh)
python scripts/check_dynamic_horizon.py --robotwin
```

The source installer pins RoboTwin revision
`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`, initializes XPolicyLab, builds CuRobo
v0.7.8 and PyTorch3D v0.7.9 (commit
`33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`), and applies the official MPLib planning fix and
SAPIEN SRDF filename correction. Patched environment files receive `.bak`
copies. It does not run the asset downloader. Source installation compiles CUDA
extensions and requires sufficient RAM, disk and a compatible compiler.

Set native Corrector task selection with
`--set evaluation.robotwin_tasks=adjust_bottle`; replace it with the exact tasks
you downloaded. Its existing task lifecycle uses the native SAPIEN evaluator,
not the Gym-based MetaWorld runner.

## LIBERO simulator

`hf-libero` supplies the benchmark code, BDDL files, initial states and simulation
assets. The shared profile includes robosuite, robomimic and MuJoCo via its
dependencies. Training demonstrations are a separate `lerobot/libero` download.
Do not install the original Python 3.8 LIBERO requirements into this environment.

```bash
python -c 'from libero.libero import benchmark; print(sorted(benchmark.get_benchmark_dict()))'
python -m dynamic_horizon.evaluate --benchmark libero --method baseline \
  --episodes 1 --output outputs/smoke/libero_baseline
```

That final command runs a real simulator/policy smoke evaluation and requires
the downloaded weights and EGL; add `--dry-run` to inspect the command only.

References: [RoboTwin installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html),
[LIBERO checkpoint and data usage](https://huggingface.co/lerobot/pi05_libero_base),
[released PACE implementation](https://github.com/niejunnan25/PACE).
