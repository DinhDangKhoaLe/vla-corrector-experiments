# Fresh H100 setup

This guide is for Linux x86_64 on a full NVIDIA H100. Run these commands on the
new machine after committing/pushing the code from the development machine.
Checkpoints, datasets, simulator checkouts, caches, and evaluation videos are
ignored by Git and must be downloaded there explicitly.

## Hardware and rendering

Use an H100 with enough host RAM and local storage for the downloaded data,
latent caches, and videos. The existing full-run default uses 20 episodes/task,
seed 1000 and prediction horizon 50. It saves 1,000 videos per MetaWorld method
and 800 per LIBERO method. The five pipeline methods are baseline, LACO_replan,
LACO_adjust, Corrector with OGG, and PACE.

CUDA support alone does not verify EGL/Vulkan rendering. SAPIEN maintainers
report rendering on a **full H100**, recommend **NVIDIA driver 570 or newer**,
and report problems with virtualized GPU partitions. Use a full GPU allocation
for RoboTwin rendering; verify the actual node before starting all evaluations.
See [SAPIEN H100 rendering discussion](https://github.com/haosulab/SAPIEN/issues/250).
H100 compute capability is 9.0 ([NVIDIA GPU list](https://developer.nvidia.com/cuda/gpus)).

Ask the cluster administrator for EGL/OpenGL/Vulkan driver libraries matching
the installed NVIDIA driver. On Ubuntu with administrator access, install the
basic build/runtime tools if missing:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential gcc-12 g++-12 \
  libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libgl1 unzip
nvidia-smi
vulkaninfo --summary
```

`vulkaninfo` should list the NVIDIA device, not only a CPU software renderer.
In an NVIDIA container, enable graphics capabilities when launching it, e.g.
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`; the host still needs the
matching driver-side graphics libraries. Do not install kernel drivers from
inside the project environment.

## Clone and install the shared environment

Miniconda/Miniforge and a working `conda` command are prerequisites.

```bash
git clone --depth 1 --branch main \
  https://github.com/DinhDangKhoaLe/vla-corrector-experiments.git vla-corrector
cd vla-corrector
git checkout main

bash scripts/setup_dynamic_horizon.sh
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh

# Rebuild native CUDA extensions on this H100; do not copy the old conda env.
export TORCH_CUDA_ARCH_LIST="9.0"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
MAX_JOBS=2 bash scripts/setup_robotwin.sh

python -m pip check
python scripts/check_dynamic_horizon.py --robotwin
```

The profile uses Python 3.12, PyTorch 2.7.1/CUDA 12.6, NumPy 1.26.4,
MuJoCo/MetaWorld/LIBERO, SAPIEN/MPLib, CuRobo and PyTorch3D. The compiler exports
above select a CUDA-12.6-compatible host compiler for extension builds; adapt
their paths to the cluster's compiler modules if needed. Import checks do not
launch a simulator or validate checkpoint conversion.

Keep the checkpoint's configured inference precision for matched comparisons;
moving to H100 alone does not require switching these experiments to bf16.
Only one GPU is used by the current sequential rollout pipeline. Running on an
eight-H100 machine does not automatically distribute evaluation across GPUs.

## Download datasets, models and simulator assets

Accept any gated PaliGemma tokenizer terms on Hugging Face, then authenticate.
These commands download data; the installer above does not.

```bash
hf auth login

# Preview all downloads before executing them.
python scripts/download_dynamic_horizon.py --benchmark all --robotwin-tasks all

# MetaWorld/LIBERO policy exports, official RoboTwin archive, clean RoboTwin demos.
python scripts/download_dynamic_horizon.py \
  --benchmark all --robotwin-tasks all --execute --unpack

# RoboTwin simulator meshes/textures/embodiment assets, separate from demos.
(cd third_party/RoboTwin && bash scripts/_download_assets.sh)

# Verify Vulkan offscreen rendering before loading a VLA into the simulator.
(cd /tmp && python -m sapien.example.offscreen)
```

| Benchmark | Policy source | Offline demonstrations |
|---|---|---|
| MetaWorld | `tiantianx/pi05_metaworld` | `lerobot/metaworld_mt50`, 50 tasks / 2,500 episodes |
| LIBERO | `lerobot/pi05_libero_base` | `lerobot/libero`, 40 tasks / 1,693 episodes |
| RoboTwin | Official `TianxingChen/RoboTwin2.0` → `cotrain_ckpt/Pi_05/...tar.gz`, pinned revision | Only selected `aloha-agilex_clean_50.zip` task archives |

The RoboTwin fine-tuning checkpoint is trained on 2,500 clean demonstrations;
its official OpenPI→LeRobot conversion is a **remaining prerequisite**, not an
automatic part of this setup. See [official checkpoint details](robotwin_official.md).

## MetaWorld then LIBERO: extraction, training, calibration and evaluation

On a fresh clone, none of the development machine's trained Correctors or
latent caches are present. This script builds missing artifacts automatically,
runs every MetaWorld method, then prepares and runs LIBERO. Existing completed
evaluations are skipped when the same output root is supplied again.

```bash
conda activate dynamic_horizon
source scripts/activate_dynamic_horizon.sh
export CUDA_VISIBLE_DEVICES=0

HORIZON=50 EPISODES=20 SEED=1000 \
  bash scripts/run_metaworld_then_libero.sh outputs/eval/h100_h50_seed1000
```

Run the same command to resume. A partial method restarts from its first
episode; its previous folder is archived. Change `HORIZON` to 40, 30, 20, or 10
with a new output root for another execution cap; prediction remains 50.
The script uses its own memory-limited systemd scope when user systemd is
available; in containers/cluster jobs without user systemd, scheduler/container
resource limits apply. Logs retain one final progress line per episode.
Metrics and video locations are documented in [evaluation](../evaluation.md).

## RoboTwin after official checkpoint conversion

Current status:

| Method | MetaWorld | LIBERO | RoboTwin |
|---|---|---|---|
| π0.5 baseline | Full local evaluation verified | Configured | Configured, official export required |
| Corrector + OGG | Full local evaluation verified | Configured | Configured, official export required |
| PACE | Full local evaluation verified | Configured | Calibration/native runner configured; official export required |
| LACO_replan / LACO_adjust | Full local evaluation verified | Configured | Not implemented |

The commands below are conditional on a **validated** official LeRobot export
at `pretrained_weights/pi05_robotwin_official_compat` and working SAPIEN
rendering. They do not convert the OpenPI archive. They use the default
`adjust_bottle` task; update `robotwin_tasks` in the two RoboTwin evaluation
configs to run a larger task list.

```bash
bash scripts/extract_corrector_pool64.sh robotwin --robotwin-tasks adjust_bottle
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_original_pool64.yaml --mode train
python -m siglip_dynamics.pace calibrate --config configs/pace/robotwin_pi05.yaml

python -m dynamic_horizon.evaluate --benchmark robotwin --method baseline --horizon 50 --episodes 20 --seed 1000
python -m dynamic_horizon.evaluate --benchmark robotwin --method corrector --horizon 50 --episodes 20 --seed 1000
python -m dynamic_horizon.evaluate --benchmark robotwin --method pace --horizon 50 --episodes 20 --seed 1000
```

RoboTwin currently uses its separate native output format; the combined
MetaWorld/LIBERO resume/video/report wrapper does not cover it. Do not interpret
this setup document as an end-to-end H100 or RoboTwin validation result: no H100
was available for testing this update, and the official archive was not downloaded.
