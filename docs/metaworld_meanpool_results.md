# MetaWorld results: full tokens and spatial mean pooling

Recorded through **2 November 2026**. The 256-token values were supplied as
prior original-code results trained on an RTX 8000. Their training seeds and
durations were not provided. All pooled values below are completed L40S runs.

| Method | Tokens | Training seed | Easy | Medium | Hard | Very hard | Overall | Training time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plain π0.5 | 256 | — | 44.46% | 38.18% | 39.17% | 27.00% | 40.70% | — |
| Original VLA-Corrector | 256 | — | 81.25% | 50.91% | 50.00% | 28.00% | 65.50% | — |
| Action-prefix | 256 | — | 79.29% | 51.36% | 52.50% | 35.00% | 65.50% | — |
| PACE π0.5, no Corrector | 256 | — | 82.14% | 46.82% | 52.50% | 38.00% | 66.40% | — |
| Uncertainty | 256 | — | 80.54% | 49.55% | 57.50% | 38.00% | 66.70% | — |
| Original VLA-Corrector | 64 | 42 | 80.18% | 49.09% | 49.17% | 35.00% | 65.10% | ~92 min |
| Action-prefix | 64 | 42 | 78.21% | 47.27% | 47.50% | 44.00% | 64.30% | ~68 min |
| Uncertainty | 64 | 42 | 79.46% | 51.36% | 55.00% | 43.00% | 66.70% | ~68 min |
| Uncertainty | 64 | 43 | 79.11% | 47.27% | 50.00% | 38.00% | 64.50% | ~43 min |
| Uncertainty | 64 | 44 | 80.89% | 44.55% | 59.17% | 37.00% | 65.90% | ~43 min |
| Uncertainty | 16 | 42 | 79.46% | 44.55% | 51.67% | 36.00% | 64.10% | ~13 min |
| Uncertainty | 16 | 43 | 78.75% | 43.64% | 55.00% | 44.00% | 64.70% | ~13 min |
| Uncertainty | 16 | 44 | 79.29% | 51.36% | 49.17% | 34.00% | 65.00% | ~14 min |

The plain-policy and PACE rows belong to the prior full-token setup; they do
not use a Corrector cache. Pooling changes only Corrector features. PI0.5
continues to process its full visual representation.

## Pooled evaluation and training protocol

- MT50, evaluation seed 1000, 20 episodes per task, at most 300 steps per episode.
- Overall success is total successes / 1,000 episodes, rather than an unweighted mean of the four difficulty groups.
- Group denominators: Easy 560 episodes; Medium 220; Hard 120; Very hard 100.
- Correctors use h=1, K=10, an execution cap of 10 and a policy chunk size of 50.
- All listed pooled runs completed 30 training epochs.
- Original uses its legacy current-action trainer, cosine loss and sample-level train/validation split.
- Prefix uses ordered action-prefix conditioning and episode-disjoint splits.
- Uncertainty adds an expected-RMSE head and MSE loss. Each run has its own 500-episode calibration collection (seed 10000) and calibration artifact.
- Training seeds control initialization, shuffle and splits. Evaluation seed stays fixed across runs.
- The 64-token seed-42 runs used disk caching. The 64-token uncertainty seeds 43/44 and all 16-token runs used GPU VRAM caching.
- Approximate training times include initial cache loading; extraction, calibration and evaluation are excluded. Hardware, seed and split differences prevent interpreting all timing differences as cache-residency effects.

Additional Original and Prefix seed-43/44 configs are available, but their
results are not included until their evaluations have been retrieved and verified.

## Uncertainty mean and standard deviation

Across training seeds 42, 43 and 44, using sample standard deviation (n−1 denominator):

| Tokens | Easy | Medium | Hard | Very hard | Overall |
| --- | --- | --- | --- | --- | --- |
| 64 (8×8) | 79.82 ± 0.94 | 47.73 ± 3.43 | 54.72 ± 4.59 | 39.33 ± 3.21 | **65.70 ± 1.11** |
| 16 (4×4) | 79.17 ± 0.37 | 46.52 ± 4.22 | 51.94 ± 2.93 | 38.00 ± 5.29 | **64.60 ± 0.46** |

Success rates are percentages; standard deviations are percentage points.
These describe training-seed variation. They are not 95% confidence intervals.

Generated PNG/CSV/JSON exports and downloaded evaluation summaries remain
under the Git-ignored `outputs/experiments/` directory. This Markdown table
preserves the recorded measurements in the repository.

See [README.md](../README.md) for extraction, training and evaluation commands.
