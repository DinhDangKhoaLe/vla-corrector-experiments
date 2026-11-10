# Random-replanning baseline

This baseline uses the frozen PI0.5 policy with a random execution controller.
It requires only policy weights and the MetaWorld runtime: no demonstrations,
latent cache, Corrector/COAP checkpoint, learned trigger, calibration or OGG.
Generation uses the policy's normal sampling noise.

For an explicit plain-policy control with no random-controller bookkeeping,
use `configs/corrector/metaworld_pi05_baseline_pool64.yaml` with
`python -m siglip_dynamics.experiments.run --config <config> --mode evaluate`.
It passes `--disable_corrector=true` and enables per-episode policy reseeding.
The older `p=0` configuration below retains the protocol of the completed
2026-11-04 comparison. Set `evaluation.reseed_policy_each_episode: true` for
both random and plain configurations when comparing against the new Corrector
protocol; choose fresh output folders. Reseeding happens after `env.reset`,
before the first policy sample of each episode.

After each action completes, the controller draws a Bernoulli decision with
probability `p` **only when** the episode is nonterminal, queued actions remain,
warm-up has finished, and cooldown has elapsed. A successful draw clears both
unexecuted action queues. The next evaluator iteration calls PI0.5 using the
latest real observation. Executed actions are never changed and the environment
is never reset by the controller.

The [example configuration](../configs/baselines/metaworld_random_replan.yaml)
matches the COAP evaluation schedule:

- PI0.5 predicts 50 actions and normally executes up to 10 before replanning.
- Five completed warm-up actions; first possible extra replan is after action 6,
  matching the earliest trigger with COAP's default five-pair window/persistence.
- After an interruption, the next 10 completed actions cannot trigger another.
  Natural chunk boundaries do not reset this cooldown or count as interventions.
- Evaluation seed 1000; 20 episodes per task across 50 tasks; maximum 300 steps.
- Default probability 0.02 **per eligible step**, not per episode or chunk.

The 2% probability is a configurable starting point, not a calibrated match to
COAP's intervention count. Report the actual interventions and VLA calls, and
compare multiple probabilities or choose a rate using held-out rollouts when
testing whether learned trigger timing helps at a similar replanning budget.
Do not interpret differences in both trigger frequency and success as evidence
about trigger timing alone.

## Run

From the repository root in the existing MetaWorld/PI0.5 environment:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless

python -m siglip_dynamics.random_replan \
  --config configs/baselines/metaworld_random_replan.yaml --dry-run

python -m siglip_dynamics.random_replan \
  --config configs/baselines/metaworld_random_replan.yaml --mode evaluate
```

Each evaluation requires a fresh output folder. To change the probability or
controller seed, also select a separate output:

```bash
python -m siglip_dynamics.random_replan \
  --config configs/baselines/metaworld_random_replan.yaml \
  --set random.probability=0.05 \
  --set random.seed=43 \
  --set evaluation.output=outputs/experiments/eval_metaworld_random_replan_p005_seed43
```

`random.seed` controls only the independent random interruption stream.
`evaluation.seed` retains the existing policy/environment seeding convention.
Per-episode interruption streams are deterministically derived from the
controller seed, task ID and episode seed, using a local NumPy generator.
Decisions do not consume the global NumPy, Python or Torch RNG streams.
After an extra replan, additional VLA predictions naturally draw additional
policy noise; identical evaluation seeds do not imply identical action sequences
between different replanning schedules.

For a matched no-extra-replanning control, set `p=0`:

```bash
python -m siglip_dynamics.random_replan \
  --config configs/baselines/metaworld_random_replan.yaml \
  --set random.probability=0 \
  --set evaluation.output=outputs/experiments/eval_metaworld_policy_matched_e10
```

This keeps the normal 10-action execution schedule, with no extra visual
encoder passes, learned detectors, OGG or temperature modification. `p=1` is
also supported for deterministic interruption at every eligible opportunity.

The dedicated plain-policy configuration is
`configs/baselines/metaworld_pi05_plain.yaml`. It sets `p=0`, preserves the
evaluation protocol above and writes to
`outputs/experiments/eval_metaworld_policy_matched_e10_seed1000`. This control
was launched alongside the random baseline on 2026-11-04 using systemd unit
`vla-pi05-plain-e10-seed1000-20261104.service`, in the same isolated runtime.
Concurrent evaluations share the GPU, so their wall times should not be used
to compare inference speed with experiments that ran alone.

## Outputs and validation

The evaluator writes its normal analysis/success summaries plus:

- `resolved_random_replan.yaml`: the exact settings used.
- `random_replan_episodes.jsonl`: episode success, seeds, eligible decisions,
  random draws, trigger times, discarded actions, VLA calls and timing.
- `random_replan_summary.json`: success by difficulty, interventions, eligible
  decision count, observed trigger fraction, VLA calls and timing.

The independent random controller is mutually exclusive with Corrector, COAP,
PACE and the separate `policy_only` flag. The launcher disables OGG and paired
baseline outputs. The first full MetaWorld evaluation was launched on Brev
`primitive-pink-bovid` on 2026-11-04, using the example configuration above and
the isolated runtime `/data/vla-random-replan-20261104`. Its systemd unit is
`vla-random-replan-p002-seed42-20261104.service`. Existing evaluations retain
their own code. This run measures the random baseline; it does not include
the separate `p=0` matched control, which has its own output folder and service.

```bash
PYTHONPATH=src python -m unittest discover -s tests/coap -v
```

Tests execute the actual evaluator control flow with a fake policy/environment,
check `p=0` against plain-policy execution, check forced random replans against
the latest observation, and verify timing, cooldown, RNG isolation, logging and
CLI mode isolation. These CPU checks do not measure real MetaWorld success.


## Uniform execution horizons (50 predicted actions)

`configs/baselines/metaworld_pi05_plain_50act.yaml` is the plain-policy control:
predict 50 actions, execute all 50, then query the latest observation. It passes
`disable_corrector=true`; no learned detector, PACE, COAP, OGG, or noise scaling
is active.

`configs/baselines/metaworld_pi05_uniform_10_50.yaml` uses the same policy and
50-action chunks, but samples one discrete uniform integer horizon from 10 through
50 (inclusive) for each chunk. It executes that prefix, discards the unexecuted
remainder, and requests a new chunk using the latest real observation. The mean
planned horizon is 30 actions. There is no Bernoulli per-step decision, warmup,
or cooldown in this strategy. A sampled horizon of 50 is a natural boundary,
not an extra intervention. Terminal episodes never trigger another replan.

Both configs use per-episode policy reseeding, evaluation seed 1000, 20 episodes
per task, all MT50 difficulty groups, and a 300-step episode limit. Controller
seed 42 determines a separate reproducible per-task/per-episode random stream;
it does not consume policy, environment, Python, or global NumPy RNG draws.
The existing Bernoulli strategy is unchanged and remains the default.

```bash
python -m siglip_dynamics.experiments.run \
  --config configs/baselines/metaworld_pi05_plain_50act.yaml --mode evaluate
python -m siglip_dynamics.random_replan \
  --config configs/baselines/metaworld_pi05_uniform_10_50.yaml --mode evaluate
```

The uniform strategy records `planned_horizon`, `chunk_action_index`,
`chunk_start_step`, discarded-action count, and trigger status in the existing
per-step `random_replan_episodes.jsonl` traces. The initial horizon is drawn on
reset; subsequent horizons are drawn at chunk boundaries before the next policy
query. Neither experiment needs a demonstration dataset or feature cache.
