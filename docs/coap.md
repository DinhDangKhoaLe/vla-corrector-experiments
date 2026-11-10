# COAP — Contrastive Observation Action Pretraining

COAP is a separate observation–action compatibility detector. It trains two
small sequence encoders on frozen visual features and executed demonstration
commands, then interrupts incompatible execution with **plain replanning**.
It does not modify VLA-Corrector dynamics, implement policy-history conditioning,
or send a compatibility score into Online Gradient Guidance (OGG).

## Score-controlled generation temperature

The optional [temperature config](../configs/coap/metaworld_pool64_temperature.yaml)
reuses the trained COAP checkpoint and changes the initial Gaussian action-noise
standard deviation in PI0.5 flow sampling. It does not scale the final actions or
change the contrastive-loss temperature. The first experiment uses this fixed rule:

```text
score <= 0.0:       noise scale T = 2.0
0.0 < score < 0.5:  noise scale T = 2.0 - 2.0 * score
score >= 0.5:       noise scale T = 1.0
incomplete history: noise scale T = 1.0
initial noise:     epsilon_T = T * epsilon, epsilon ~ N(0, I)
```

The latest completed five-pair window controls the next **naturally scheduled**
chunk request. COAP continues scoring every executed action, but never clears the
action queue in this mode. With the default schedule, a new 50-action chunk is
requested after each 10 executed actions, using the latest real observation.
There is no early replan, OGG, threshold calibration, or extra random draw for
temperature control. `T=1` leaves the original sampled-noise tensor unchanged.
The two towers remain frozen; this initial mapping is fixed, not learned.
Noise scaling is a sampling intervention, not an exact softmax-temperature or
probability transformation of the flow model's output distribution.

```bash
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64_temperature.yaml --mode evaluate
```

Each episode records the score, applied noise scale and completed-step index for
every chunk request in `generation_requests`; `coap_summary.json` aggregates the
number of chunks and the min/max/mean scale. Warm-up starts fresh each episode.
The score bounds, temperature bounds and warm-up temperature are configurable in
the `generation` section. Defaults keep generation control disabled, and existing
replanning/calibration fingerprints are preserved. Use an isolated source snapshot
when scheduling this experiment behind runs that must keep using their old code.

## Observation/action timing

The MetaWorld evaluator builds a policy input from the current observation,
postprocesses the selected command, and calls `env.step(action_np)`. The cache
extractor similarly stores the image and action from the same dataset row.
Thus the five pairs are:

```text
observation:      o1      o2      o3      o4      o5
executed action:  a1      a2      a3      a4      a5
                 |------- five same-timestep pairs -------|
score:                                           after a5 returns
```

`o5` is captured **before** `a5`. Its outcome is not visible in this window.
COAP preserves that convention: it never adds `o6`, shifts the actions, or uses
unexecuted future commands. Each action must return from `env.step` before its
pair is committed. Initial scoring begins after five executed pairs; subsequent
scores use the rolling five-pair window after every completed action. There is
no padding. Only episode reset clears the history; ordinary replans retain it.

Trace steps count completed actions, starting at one. `observation_steps` uses
the evaluator's zero-based observation indices, so the first score contains
`[0,1,2,3,4]` at completed step 5. During warm-up the normal policy chunk schedule
continues. Terminal pairs can be scored, but a terminal episode is not replanned.

## Features, model and contrastive objective

The initial integration supports **PI0.5 on MetaWorld**. It reuses the existing
per-frame int8 visual cache and its spatial-pooling layout. Within each frame,
dequantized tokens are spatially averaged to a feature vector; the five frame
vectors remain ordered. This initial detector does not consume proprioception
or task text. Two independent projection/GRU/projection towers encode the visual
sequence and action sequence into equal-dimensional, L2-normalized embeddings.
Hidden size, embedding size, GRU depth/dropout and window size are configurable
in [the example config](../configs/coap/metaworld_pool64.yaml); window size defaults
to five. The VLA is frozen and is not loaded for cached-feature training.

For embeddings `O` and `A`, the training logits are `O @ A.T / temperature`.
The loss averages observation-to-action and action-to-observation cross entropy.
Matching pairs form the positive diagonal. The temperature is configurable.

Episodes are split **before** constructing windows, using 80/10/10 fractions
by default. `training.split_seed` controls the episode split independently of
`training.seed`, which controls initialization/shuffling. Both default to 42.
The split manifest and train-only action mean/std are saved with the checkpoint.
Test episodes are reserved and do not participate in model selection or fitting.

`EpisodeBatchSampler` chooses at most one window from an episode per batch.
The contrastive mask additionally excludes overlapping windows from the same
episode and exact duplicate visual or action sequences, including duplicates
from different episodes. Anchors without eligible negatives are excluded;
an epoch without usable contrastive pairs fails rather than training a trivial
zero loss. A final singleton-episode tail cannot form a contrastive batch and is
omitted. The sampler and eligibility mask are separate extension points for
task-aware hard negatives or multi-positive objectives.

Other unmatched windows may still contain valid interchangeable actions.
Their negative labels are an approximation, not verified execution failures.

## Reuse an existing cache without leaking action normalization

Existing Corrector caches typically store policy-quantile-normalized, clipped
actions using dataset-wide statistics. COAP does not invert those clipped values
or fit another normalizer on top of them. Its raw commands must be in the same
coordinates as the action actually dispatched to the environment.

For an existing **local-parquet MetaWorld cache**, export raw actions from the
same unchanged demonstration dataset in the extractor's row order:

```bash
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml \
  --mode prepare-actions \
  --dataset-path datasets/metaworld_mt50
```

This reads the raw action, episode and frame-index columns, checks frame counts,
episode order and contiguous within-episode frame indices, and saves
`coap_raw_actions.npy` plus a checksum/provenance JSON in the cache directory.
Images are not decoded and visual features are not re-extracted. The action
sidecar is about 3.3 MB for 204,806 four-dimensional float32 commands.
The adapter assumes the cache was extracted from this same unmodified parquet
dataset; old caches do not record per-frame image hashes. Filtered/reordered
sources require an explicit aligned export rather than guessing the pairing.
For a newly extracted `raw_action` cache, COAP can use `actions.npy` directly.
Offline pairing assumes the demonstrations use the same pre-action observation
convention; frame/episode checks cannot establish physical sensor/actuator latency.

Action normalization is then fitted using **training episodes only**, with a
configurable standard-deviation floor, and saved in `best_model.pt` and
`action_normalization.json`. Runtime uses the postprocessed command passed to
`env.step`, not `raw_action_plan` or any remaining future chunk actions.

## Train, collect, calibrate and evaluate

Run from the repository root in the existing MetaWorld/PI0.5 environment. These
commands are for a future real experiment; implementation testing does not launch
the jobs. Each output must be fresh. `--dry-run` prints configs/evaluation commands
without loading the dataset, model, policy or simulator.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

# After prepare-actions (or with an already-raw action cache):
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode train

# Frozen policy, no interruptions; 10 episodes per task, collection seed 10000.
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode collect-calibration \
  --set evaluation.episodes=10 --set evaluation.save_video=false

python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode calibrate

# 20 episodes per task, evaluation seed 1000; COAP + plain replanning.
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode evaluate

python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode summarize
```

The default checkpoint is `outputs/experiments/metaworld_coap_pool64/best_model.pt`,
selected by validation contrastive loss. `last_model.pt`, `metrics.jsonl`, the
configuration, episode split and action-normalization files are also saved.

Calibration prefers separate **successful policy rollouts** with no interventions.
It takes the configured lower quantile (default 5th percentile) of the minimum
similarity in each successful collection episode. At least 50 successful episodes
with complete windows are required. This is an empirical threshold-selection
rule, not a statistical guarantee of a 5% false-alarm rate. Artifacts bind the
COAP checkpoint, policy preprocessing and rollout schedule; evaluation rejects
task/seed episodes overlapping calibration collection.

When policy collection is unavailable, a deliberately provisional fallback can
use demonstration-validation episodes:

```bash
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode calibrate \
  --source demonstrations \
  --set monitor.artifact=outputs/experiments/metaworld_coap_pool64/demo_calibration.json

python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode evaluate \
  --set monitor.artifact=outputs/experiments/metaworld_coap_pool64/demo_calibration.json \
  --set monitor.allow_provisional_threshold=true
```

An explicitly supplied `monitor.threshold` also requires
`monitor.allow_provisional_threshold=true` and is logged as provisional.
Runtime uses **direct cosine similarity**; `1 − similarity` is logged as mismatch.
Neither batch-softmax values nor cosine scores are interpreted as probabilities.
Persistence and cooldown are configurable. An alarm at a natural chunk boundary
is logged but is not counted as an extra interruption. A real trigger discards
the unexecuted queue remainder, and the next loop requests a fresh chunk from
the latest real observation. It does not reset the environment.

## Matched comparisons

COAP's no-detector control uses the same evaluator, checkpoint, action cap,
chunk length and evaluation seeds, with the detector and OGG disabled:

```bash
python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode evaluate --comparison policy
```

The existing Original Corrector remains a separate dynamics detector. After its
training, compare the usual OGG recovery and the same detector with plain replanning:

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml --mode evaluate \
  --set evaluation.policy=pretrained_weights/pi05_metaworld_eval_compat \
  --set evaluation.seed=1000 \
  --set evaluation.output=outputs/experiments/eval_coap_comparison_original_ogg

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml --mode evaluate \
  --set evaluation.policy=pretrained_weights/pi05_metaworld_eval_compat \
  --set evaluation.seed=1000 --set evaluation.guidance=false \
  --set evaluation.output=outputs/experiments/eval_coap_comparison_original_replan
```

These comparisons match online evaluation conditions. The preserved legacy
Original trainer uses a sample-level split, whereas COAP uses episode-level
splits; keep that difference explicit when interpreting results. Vary both
`training.seed` and `training.split_seed` if reproducing our previous combined
initialization/split repeats; keep `split_seed` fixed to isolate initialization
and shuffle variation. Choose distinct checkpoint, artifact and evaluation
outputs for every new run.

## Fresh features, logs and compute cost

COAP performs a **fresh frozen PI0.5 vision-encoder pass on every real observation**,
including the warm-up steps. This extra vision pass is required between VLA calls;
the old demonstration cache or the image from the last chunk request cannot
represent the new observation. At chunk boundaries, the policy also performs its
normal vision/inference work. Spatial averaging reduces tower input size, not
the frozen vision encoder's work. Training uses quantized cached features while
online scoring uses freshly encoded features, so calibration must measure the
actual policy-rollout distribution.

`coap_episodes.jsonl` records per-window cosine/mismatch, observation indices,
threshold, alarms, cooldowns, trigger times and discarded-action counts. Episode
records include task success, interventions, VLA calls/inference time, fresh-feature
time, scoring time and episode wall time. `coap_summary.json` aggregates these
metrics; the existing evaluator also writes its usual task/episode summaries.
Training logs parameter count/device and records elapsed time and usable/skipped
contrastive batches. No full-policy compute-cost measurements are claimed from
the synthetic smoke tests.

Task success/failure alone does not label which windows are abnormal. Summaries
therefore leave false-alarm/detection-delay metrics unavailable unless labels
exist. Optional trace `abnormal: false` labels enable false alarms per labelled
normal scored window. Optional per-episode `anomaly_intervals` containing inclusive
`start_step`/`end_step` values enable intervention delays (in completed actions)
and missed-interval counts. Use the same labeling protocol across detectors.

## Tests and current validation limits

```bash
PYTHONPATH=src python -m unittest discover -s tests/coap -v

python -m siglip_dynamics.coap.run \
  --config configs/coap/metaworld_pool64.yaml --mode evaluate --dry-run
```

The CPU tests cover data alignment, train-only normalization, contrastive gradients
and duplicate/overlap masking, no scores before five completed pairs, rolling order,
queued-action exclusion, resets, persistence/cooldown, high-score continuation,
terminal/boundary handling, checkpoint reload and threshold provenance.
Evaluator smoke tests execute the actual evaluator function with a fake policy
and environment, including clearing the queue and checking that the next policy
call uses the latest observation. They load its Python AST to avoid importing
unrelated optional robot packages; they do not simulate real MetaWorld physics.

The local checkout currently lacks the real MT50 cache/demonstrations and PI0.5
policy weights. Its general Python environment also lacks an optional `serial`
dependency pulled in by the full evaluator import. Real training and live
MetaWorld evaluation have not been run; use the existing complete cloud/runtime
environment for those future experiments. No real-robot integration is enabled.
