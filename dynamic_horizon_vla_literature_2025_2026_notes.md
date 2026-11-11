# Dynamic execution horizons for VLA policies, 2025–2026

Checked: 2026-11-10. This updates the existing 13-paper comparison; it is not a claim of an exhaustive systematic review. Full-text methods and evaluation sections were checked, with official repositories inspected for implementation availability and the A³ action-update distinction. Published results were not reproduced.

Year means first public preprint year. Thus MoH is listed as 2025, although its conference publication is ICML 2026. Benchmark/backbone coverage reflects the paper versions linked below, including later revisions.

"Simulation dataset used" lists simulation benchmarks and explicitly named associated training splits, not all backbone pretraining data. "Backbone policy" lists policies evaluated with each method, not every comparator.

**Updated chunks — user's definition**
- **No:** choose or truncate an existing action prefix, stop execution, and make an ordinary policy query. New observations, selecting one independently sampled candidate, or generating a fresh ordinary chunk do not by themselves count as updating.
- **Yes:** change action values by fusion/correction, or condition/guide regeneration beyond an ordinary new observation-conditioned query.
- PACE is No: speed smoothing is used to choose the boundary, not to replace the action values.
- MoH is Yes for the complete method because it fuses predictions; its horizon-consensus truncation rule alone is No.
- A³ is Yes for the inspected released implementation: it can append the conditionally re-decoded action at the rejection boundary. The paper's formal description emphasizes accepting an unchanged verified prefix; this implementation qualification should not be dropped.

"Not found" means no public author implementation was located in the paper/project links and targeted searches on the check date. It is not proof that no repository exists. "Yes" does not promise that every experiment is reproducible from the release.

The BCP repository checked at commit 2ceec1045a2daa435c3549c8b3ccf177b67da1b3 contains index.html and static website assets, not the method implementation:
https://github.com/fleetfootwork/BCP/tree/2ceec1045a2daa435c3549c8b3ccf177b67da1b3

A³ implementation evidence (lines 1630–1659, file blob SHA 06b51d90a80eb0b5f58c09f9b7b8298a1c8aaee6):
https://github.com/INCEPTIONwang/A3/blob/main/rlinf/models/embodiment/openpi/openpi_action_model.py#L1630

## Paper sources

- EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation: https://arxiv.org/html/2511.05397v1
- Mixture of Horizons in Action Chunking (MoH): https://arxiv.org/html/2511.19433v2
- VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies (AutoHorizon): https://arxiv.org/html/2602.21445v2
- Adaptive Action Chunking at Inference-time for Vision-Language-Action Models (AAC): https://arxiv.org/html/2604.04161v2
- Dynamic Execution Commitment of Vision-Language-Action Models (A³): https://arxiv.org/html/2605.11567v4
- PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking: https://arxiv.org/html/2606.00537v2
- Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies (DVAC): https://arxiv.org/html/2606.03847v1
- VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon: https://arxiv.org/html/2607.01804
- CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation: https://arxiv.org/html/2607.26789v1
- Continue or Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon Execution (BCP): https://arxiv.org/html/2608.03483v1
- Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs: https://arxiv.org/html/2609.00908v3
- GeoAAC: Geometry-Based Adaptive Action Chunking from Denoising Trajectories in VLA Policies: https://arxiv.org/html/2609.20776v1
- Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution (PAV): https://arxiv.org/html/2610.21434v1

## Comparison

| Year | Name | Dynamic horizon mechanism | Simulation dataset used | Backbone policy | Updated chunks | Code available |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | [EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation](https://arxiv.org/html/2511.05397v1) | Discrete/continuous prediction disagreement determines how much of the discrete chunk to execute. | LIBERO | EveryDayVLA: Prismatic-7B with discrete and continuous action heads | No — selects a discrete-action prefix; no action fusion or corrective regeneration. | Yes — https://github.com/everydayvla/EveryDayVLA |
| 2025 | [Mixture of Horizons in Action Chunking (MoH)](https://arxiv.org/html/2511.19433v2) | Fuses multiple horizon branches; cross-horizon agreement determines the executable prefix. | LIBERO; RoboTwin 2.0; RoboCasa (five selected tasks) | π0; π0.5; πreg; StarVLA-GR00T (Qwen3-VL) | Yes — learned gating fuses action predictions across horizons; its horizon selector alone only truncates. | Yes — https://github.com/Timsty1/MixtureOfHorizons |
| 2026 | [VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies (AutoHorizon)](https://arxiv.org/html/2602.21445v2) | Action self-attention anchors estimate the reliable execution prefix. | LIBERO; RoboTwin (seven tasks) | π0.5; GR00T N1.5 | No — selects an unchanged prefix, then makes a normal policy query. | Yes — https://github.com/hatchetProject/AutoHorizon |
| 2026 | [Adaptive Action Chunking at Inference-time for Vision-Language-Action Models (AAC)](https://arxiv.org/html/2604.04161v2) | Multi-sample action entropy selects the horizon, subject to a minimum-motion constraint. | RoboCasa Kitchen; LIBERO; LIBERO-Pro | GR00T N1.5; π0.5 | No — selects a sampled candidate and its prefix; no guided regeneration or action-value correction. | Yes — https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample ; simulation clients: https://github.com/Adaptive-Action-Chunking |
| 2026 | [Dynamic Execution Commitment of Vision-Language-Action Models (A³)](https://arxiv.org/html/2605.11567v4) | Sample consensus selects a draft; conditional re-decoding verifies its executable prefix. | LIBERO; MetaWorld; ManiSkill | π0; π0.5; GR00T | Yes (released implementation) — can append a conditionally re-decoded action at the first rejection; paper framing emphasizes prefix verification. | Yes — https://github.com/INCEPTIONwang/A3 |
| 2026 | [PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking](https://arxiv.org/html/2606.00537v2) | Selects the earliest prominent low-speed valley as a phase boundary; thresholds calibrated from demonstrations. | RoboTwin 2.0 (50 tasks) | π0.5 | No — executes a speed-selected prefix, discards the suffix, and makes a normal policy query; does not change action values. | Yes — https://github.com/niejunnan25/PACE |
| 2026 | [Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies (DVAC)](https://arxiv.org/html/2606.03847v1) | Late-denoising action variance identifies a stable prefix; rolling statistics adapt the threshold. | LIBERO; RoboTwin; CALVIN | π0.5; π0; Qwen2.5-VL-π; Qwen3-VL-GR00T | No — executes the stable prefix, then makes a normal policy query. | Not found — no public author implementation located as of 2026-11-10. |
| 2026 | [VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon](https://arxiv.org/html/2607.01804) | Observed latent-dynamics drift interrupts execution and triggers gradient-guided recovery. | MetaWorld; LIBERO; RoboTwin 2.0 | π0.5; SmolVLA; X-VLA | Yes — Online Gradient Guidance changes the recovery policy query after truncation. | Yes — https://github.com/ZJU-OmniAI/vla-corrector |
| 2026 | [CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation](https://arxiv.org/html/2607.26789v1) | Action-conditioned world-model risk triggers latency-aware suffix repair. | RoboCasa365; Human300 policy-training split | π0.5-based policy with separate mobility/manipulation experts | Yes — keeps the committed prefix and regenerates the deployable suffix using risk-dependent guidance. | Not found — no public author implementation located as of 2026-11-10. |
| 2026 | [Continue or Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon Execution (BCP)](https://arxiv.org/html/2608.03483v1) | An RL-trained continuation head chooses a prefix through sequential continue-or-replan decisions. | RoboTwin 2.0; LIBERO; LIBERO-Pro | LingBot-VLA; ABot-M0; π0.5; ACT (non-VLA baseline) | No — changes execution length only; action generation remains unchanged. | No implementation found — https://github.com/fleetfootwork/BCP contains project-website assets only as checked on 2026-11-10. |
| 2026 | [Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs](https://arxiv.org/html/2609.00908v3) | Truncates when action-to-observation attention reaches a sustained high-entropy plateau. | RoboTwin 2.0; LIBERO | π0.5; X-VLA | No — attention determines the truncation point; subsequent policy queries are ordinary. | Not found — no public author implementation located as of 2026-11-10. |
| 2026 | [GeoAAC: Geometry-Based Adaptive Action Chunking from Denoising Trajectories in VLA Policies](https://arxiv.org/html/2609.20776v1) | Denoising-trajectory geometry across prefixes selects a horizon from one generation. | LIBERO; LIBERO-Pro; RoboCasa365 | GR00T N1.5; π0.5 | No — geometric reliability selects a prefix without changing the generated actions. | Not found — no public author implementation located as of 2026-11-10. |
| 2026 | [Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution (PAV)](https://arxiv.org/html/2610.21434v1) | Predicts prefix consequences in policy-latent space and selects the longest valid prefix. | LIBERO; RoboTwin 2.0 | π0; π0.5 | No — evaluates existing action proposals; does not generate or correct action values. | Not found — no public author implementation located as of 2026-11-10. |
