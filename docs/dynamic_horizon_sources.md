# Dynamic horizon methods on MetaWorld

The independent `src/dynamic_horizon` package uses the downloaded MetaWorld
π0.5 checkpoint and the existing LeRobot preprocessing/MetaWorld wrapper.
It does not alter VLA-Corrector. Entry points are documented in `training.txt`
and `evaluation.txt` at the repository root.

## Sources and implementation scope

Sources were inspected on 2026-11-10/11. These are MetaWorld adaptations, not
claims to reproduce the papers' benchmark scores or latency.

| Method | Primary source | Implemented behavior |
| --- | --- | --- |
| AutoHorizon | [Paper](https://arxiv.org/abs/2602.21445), [official PyTorch model](https://github.com/hatchetProject/AutoHorizon/blob/c7504f1756109103f2cfcc2e23f1b1a23841c885/src/openpi/models_pytorch/pi0_pytorch.py) | Bidirectional soft attention pointers, entropy reliability mask, earliest stable pointer; full-horizon overlap rule. |
| AAC | [Paper](https://arxiv.org/abs/2604.04161), [official entropy/horizon implementation](https://github.com/Adaptive-Action-Chunking/libero-pro/blob/1c83fa78a172707ab04e6a11d8b148d027f5fb23/action_optimization/action_entropy_pi05.py) | Gaussian position entropy plus Bernoulli gripper entropy; prefix-mean entropy elbow; minimum movement floor; sampled trajectory nearest the sample mean. |
| PACE | [Paper](https://arxiv.org/abs/2606.00537), [official standalone module](https://github.com/niejunnan25/PACE/tree/5d0f4a92415f8899963cba7f0f929f1b9c5f75db/pace/src/pace) | Savitzky–Golay speed profile; low-speed valleys; contrast against the chunk start; earliest accepted boundary. Per-task fifth-percentile demonstration calibration. |
| A³ | [Paper](https://arxiv.org/abs/2605.11567), [official decoder](https://github.com/INCEPTIONwang/A3/blob/96a85c698f3848f6e684e36a81caf6e275db2ecc/rlinf/models/embodiment/openpi/openpi_action_model.py) | Cumulative-XYZ trajectory consensus draft; confidence and sequential conditional verification; intersected contiguous prefix; optional replacement at first rejection. |
| BCP | [Paper and appendix](https://arxiv.org/html/2608.03483v1), [project](https://fleetfootwork.github.io/BCP/) | Frozen policy plus Transformer continuation head; first-stop Bernoulli distribution; reference-relative reward and group-relative clipped policy optimization. |

BCP's [repository](https://github.com/fleetfootwork/BCP/tree/2ceec1045a2daa435c3549c8b3ccf177b67da1b3)
contains website assets only. `bcp.py` and `train_bcp.py` implement the paper;
they are not a port of unreleased official training code. The paper specifies
two Transformer layers/eight heads; width 256 and learned position embeddings
are explicit implementation choices because the full architecture is not released.

A³ follows the inspected released implementation. The newer paper describes
mode-aware clustering/quantized verification, while that code uses cumulative
trajectory mean/variance and a median-confidence vote. This distinction is
intentional: no unsupported claim of implementing unpublished quantized code.
Its original MetaWorld configuration uses a shorter prediction horizon; here
both orders span the requested 50 actions. The released XYZ thresholds
`[2, 2, 3.4]` are exposed in YAML, in decoded action units. They are not tuned
on evaluation episodes and may need training-side validation for this checkpoint.

## MetaWorld-specific adaptations

MetaWorld commands contain XYZ displacement controls plus gripper, not absolute
joint positions or six-dimensional end-effector poses. PACE therefore computes
speed as `norm(clip(action_xyz, -1, 1) * 0.01) * 80`, following MetaWorld's
Cartesian command scale and controller FPS. Differencing consecutive delta
commands would measure acceleration and incorrectly identify constant movement
as stationary. Calibration uses the same action-derived speed on all 2,500
downloaded demonstrations, grouped by canonical `task_id`. It reads no images
or evaluation outcomes. Double precision and a machine-roundoff guard prevent
flat saturated commands from generating numerical contrast thresholds.
This adaptation measures commanded speed; it does not pretend to recover joint
velocity, collisions, or actual end-effector motion from the action alone.

AAC omits rotational entropy/motion because MetaWorld has no rotation command.
Its movement floor is 0.01 metres with gripper weight 0.2, configurable. It
retains the author's first-sample movement floor and entropy-elbow indexing.

A³ compares decoded XYZ before clipping and does not compare the gripper.
Conditioned normalized actions are clamped to their forward-noised values at
each denoising step, then exactly restored at the final step. All action
dimensions of an accepted position are anchored. Candidate sampling and
verification are serial, with an observation-prefix KV cache, to fit one GPU.
Both verification orders remain active; this changes batching/random-number
order and throughput, not the prefix acceptance conditions.

BCP uses the paper's H=50 candidate set `[15,20,25,30,35,40,45,50]` and runs
groups serially: one fixed-50 reference plus seven adaptive trajectories from
the same initial task instance. All eight outcomes contribute to advantage
normalization; only adaptive decisions receive GRPO/entropy updates. Terminal
reward is broadcast to each decision and the loss is averaged over all adaptive
decisions. Only the continuation head is trained. Offline demonstrations cannot
replace the online counterfactual rollouts required by this objective. The local
checkpoint already supplies the pretrained action policy; no additional VLA
fine-tuning is performed.

## Runtime and evaluation conventions

- Policy checkpoint: `pretrained_weights/pi05_metaworld_compat`; this existing
  symlink view removes unsupported config metadata without changing weights.
- Strict local weight loading fails on missing keys/shapes, instead of accepting
  the upstream loader's random-weight fallback after a load error.
- Tokenizer: cached public OpenPI PaliGemma SentencePiece model, exact BOS/token
  IDs and right padding. Saved state normalization and action unnormalization
  remain in use; missing wrist cameras retain the policy's masked padding.
- AutoHorizon reads all action-expert layers/heads at zero-based denoising step 3.
  BCP reads valid visual-language context and final expert velocity features.
- This runner uses eager attention for vision, language and action modules.
  The installed PyTorch 2.7.1/CUDA 12.8 SigLIP SDPA path produced differing
  features on repeated identical images on the local RTX 5060 Ti. Eager vision
  attention removes that variation; native-sampler parity is checked with the
  same eager backend. This can change latency relative to SDPA-based runs.
- Prediction length is always 50; maximum execution length is 50. A method's
  selected prefix can be shortened further by episode termination/time limits,
  and both selected and executed counts are logged.
- Default evaluation: all 50 tasks, 20 episodes/task, 300 environment steps,
  base seed 1000 with the repository's difficulty/task ordering. NumPy is seeded
  before reset because this installed MetaWorld ignores the `reset(seed)` argument.
  MuJoCo data is reset before hand initialization to remove residual physical
  state. Policy RNG is reseeded after reset. This fixes reproducibility in this
  runner; historical results from the old reset behavior are not bitwise matched.
- BCP logs and checks physical-state fingerprints for identical initial group
  states; its training seed starts at 100000, separate from evaluation seeds.
- Default inference precision is bfloat16 (vision/norm exceptions follow this
  repository's π0.5 implementation) for the local 16 GB GPU. This is a precision
  choice, not a claim of matching float32 baseline results.
- Required simulator version is the repository-pinned `mujoco==3.6.0`.
  MuJoCo 4.0 changes equality-data layout and breaks the installed MetaWorld 3.0.

## Verification

Run `python -m unittest discover -s tests/dynamic_horizon -v` for algorithm
contracts. `python -m dynamic_horizon.smoke --output NEW_DIRECTORY` performs
real-model/native-sampler parity, conditional-anchor, attention/feature extraction,
full-dataset PACE calibration, all-method simulator rollouts, and BCP training,
checkpoint resume, reload and evaluation checks. Its BCP head is deliberately
stored separately and must not be reported as a fully trained benchmark model.
Full 50-task performance evaluations and production BCP training are separate
commands in the two root text files.
