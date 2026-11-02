# Brev: MetaWorld setup and Corrector workflows

From your **local checkout**, with the Brev CLI already authenticated:

```bash
bash scripts/brev/setup.sh
```

This defaults to instance `primitive-pink-bovid` and remote directory
`/data/vla-corrector-experiments`. It waits for SSH, copies the source including
local edits, creates an isolated Python 3.12 environment, downloads MT50 and
the fine-tuned PI0.5 checkpoint, creates the compatibility view, and extracts
the **64-token (8×8) spatial mean-pooled cache** from
[README section 6](../README.md#6-spatial-mean-pooling-extract-train-and-evaluate-all-corrector-modes).
Ignored local datasets, weights, environments, outputs and credentials are excluded from
the source transfer. Git history is not copied. Downloads occur on the instance.

Setup runs in the background and survives disconnection. The launcher prints
the log path; inspect progress with:

```bash
brev exec primitive-pink-bovid 'cat /data/vla-corrector-experiments/outputs/brev/setup.status; tail -n 30 /data/vla-corrector-experiments/outputs/brev/setup-*.log'
```

Wait for `setup.status` to say `complete` before starting training. The separate
training launcher defaults to Original Corrector; select another mode or train
all three sequentially:

```bash
bash scripts/brev/train.sh original
bash scripts/brev/train.sh prefix
bash scripts/brev/train.sh uncertainty
# Alternatively, train all three in sequence:
bash scripts/brev/train.sh all
```

Each invocation starts a background job on the instance, with timestamped logs
and a `train-<mode>.status` file under `outputs/brev/`. Run one invocation at a
time. Setup and training share a GPU lock. Existing training outputs follow the
trainers' overwrite protection; choose a new output for repeated experiments:

```bash
bash scripts/brev/train.sh prefix --set training.output=outputs/experiments/metaworld_prefix_pool64_repeat
```

Both local launchers accept `--instance NAME` and `--remote-dir /absolute/path`.
Setup also accepts `--wait-seconds 900` and `--foreground`. Inside the instance,
`bash scripts/brev/train.sh --local prefix` runs training in the foreground.
Calibration and evaluation remain separate, as described in
[README section 6](../README.md#6-spatial-mean-pooling-extract-train-and-evaluate-all-corrector-modes).

## Evaluation policy and tokenizer

The cloud runs use `pretrained_weights/pi05_metaworld_eval_compat` for evaluation
and uncertainty calibration. This view reuses the policy weights and points its
preprocessor at a local PaliGemma tokenizer. The tested instance already has it;
`setup.sh` creates the extraction compatibility view, but does **not** create this
second evaluation view.

For the tested setup, the public OpenPI PaliGemma SentencePiece model was
converted to a local tokenizer and its token IDs were checked against the
SentencePiece model. Its provenance is recorded on the instance in
`outputs/brev/eval-tokenizer-provenance.json`. Use the same policy/tokenizer
view for calibration collection and evaluation so their protocol identities match.
The seed-43/44 configs already select this evaluation view; for the base configs,
pass `--set evaluation.policy=pretrained_weights/pi05_metaworld_eval_compat`.

## Uncertainty workflow

After preparing that evaluation view on a new instance, run all stages in sequence:

```bash
bash scripts/brev/uncertainty.sh --after-prefix
```

To evaluate Prefix before starting uncertainty training, use this instead:

```bash
bash scripts/brev/uncertainty.sh --eval-prefix-first
```

This waits for Prefix training and evaluates its best checkpoint on all 50
tasks (20 episodes each), then starts the uncertainty stages below. A Prefix
evaluation failure stops the workflow before uncertainty training begins.

This queues behind a successful Prefix run, then trains the pooled uncertainty
model, collects 10 episodes per MT50 task with seed 10000 and videos disabled,
calibrates from successful episodes, and evaluates 20 episodes per task with
seed 1000 and videos enabled. Calibration comes **after training** and needs
at least 50 successful episodes. Collection observes the frozen policy without
triggering recovery; evaluation uses the calibrated alarm. All rollout stages
use `pretrained_weights/pi05_metaworld_eval_compat`, including its local tokenizer,
so their policy and protocol identities match. Calibration also checks the error
head: if its correlation is too weak, evaluation uses ordinary replanning with
OGG disabled. Existing outputs are preserved and rejected on a fresh launch.

The queued workflow survives disconnection, releases the GPU to the Prefix job
while waiting, and stops if any stage fails. Check
`outputs/brev/uncertainty-workflow.status` and its timestamped log for progress.
Omit `--after-prefix` to start immediately, or use `--through train` /
`--through calibrate` to stop after that stage. The workflow shares the GPU lock
with the other Brev jobs; `--instance` and `--remote-dir` are supported too.

The cloud environment uses the repository's `metaworld` and `pi` extras with
the core pins in `scripts/brev/constraints.txt`. It uses a checksum-verified
[Miniforge installer](https://github.com/conda-forge/miniforge) and FFmpeg 7.
TorchCodec 0.5 is paired with PyTorch 2.7.1 according to the
[official compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).
This avoids installing every unrelated simulator and optional CUDA extension
from the full workstation `environment.yml`. The resolved package list is saved
to `outputs/brev/installed-requirements.txt`.

Allow **at least 160 GB free** for setup, including the datasets library's
intermediate Arrow copy; use the instance's data disk. Re-running setup resumes
Hugging Face downloads and validates/reuses a completed cache. Extraction itself
cannot resume: an incomplete cache is preserved with an `.incomplete.*` suffix
and extraction starts again. A completed cache with mismatched metadata is
rejected. The scripts do not start training automatically.

## Cache residency and seeded repeats

`train.sh` selects the base 64-token configs. Forward RAM/VRAM training options
with `--set`, for example:

```bash
bash scripts/brev/train.sh prefix \
  --set training.cache_location=gpu \
  --set training.output=outputs/experiments/metaworld_prefix_pool64_vram
```

The dedicated seed-43/44 configs also set separate training, calibration and
evaluation paths. Use them through the Python entry point in the remote
`.venv`, following the
[seeded-run instructions](../README.md#i-4x4-pooling-and-seeded-repeats).
The `train.sh` and `uncertainty.sh` wrappers do not automatically select these
configs or the 4×4 cache. The four-run Original/Prefix queue used in the
experiments was launched separately; it is not a feature of these wrappers.

The tested environment used an L40S. Packed training caches require about
25.03 GiB for 64 tokens or 6.26 GiB for 16 tokens, plus model/optimizer/batch
memory. GPU residency affects training; evaluation still runs the frozen PI0.5
policy and online Corrector. Extraction writes the reusable cache to disk first.
