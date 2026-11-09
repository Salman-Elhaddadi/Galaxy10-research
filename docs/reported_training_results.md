# User-reported training results — 2026-11-05

Update 2026-11-09: new saved-artifact clean and augmented seed-0 reruns reproduce the selected epoch-30 validation macro-F1 values below. Their checkpoints, configurations, normalization and 20-condition validation evaluations are preserved and checked as recorded in `exploratory_validation_results.md`. The original runs listed in this historical log were not recovered; clean seeds 1/2 remain console-only. Do not pool the reruns as additional independent seeds. No frozen-test result is available.

These results were transcribed from user-supplied console logs. They have not been reproduced locally, and the corresponding checkpoints, resolved configurations, data fingerprints and protocol manifests have not yet been inspected. They are clean-validation diagnostics, not frozen-test results.

## Model A clean condition

Checkpoint selection follows the protocol: maximize fixed-ten-class clean-validation macro-F1, using the earliest epoch on a tie.

| Seed | Run ID | Selected epoch | Macro-F1 | Accuracy | Final train loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `model-a-seed0-20261105T104653987608Z-c47ddcb64822416c9be2c4dcec071e13` | 30 | 0.7665 | 0.7940 | 0.6384 |
| 1 | `model-a-seed1-20261105T105851883059Z-e7ec974b482349d79955176f38b88670` | 30 | 0.7514 | 0.7843 | 0.6772 |
| 2 | `model-a-seed2-20261105T114530574442Z-6ba66ecc23844a1ba78d62c7a13dfc6e` | 28 | 0.7596 | 0.7923 | 0.6656 |
| **Mean** | — | — | **0.7592** | **0.7902** | — |
| **Sample standard deviation** | — | — | **0.0076** | **0.0052** | — |

The best individual macro-F1 is seed 0's 0.7665. The across-seed macro-F1 range is 0.0151. Seed 1's maximum accuracy is 0.7855 at epoch 28, but its selected checkpoint remains epoch 30 because macro-F1 is the primary selection metric. Seed 2's epoch-30 values are not used because its macro-F1 peaked at epoch 28.

## Interpretation boundary

All three training losses generally decrease while validation macro-F1 fluctuates, so checkpoint selection matters. The small across-seed sample standard deviation indicates reasonably consistent clean-validation performance for these three reported runs. It does not establish generalization to the held-out test set, robustness under degradation, calibration, or formal protocol eligibility.

Before aggregation with evaluation results, verify each run's `summary.json`, `config.json`/resolved configuration, `best.pt`, normalization and split fingerprints, condition metadata, and frozen-protocol manifest status.

## Model A augmentation condition (exploratory single seed)

The user elected to stop augmentation training after seed 0 and proceed to degradation diagnostics. This is a scope reduction from the protocol's three-seed design, so augmentation comparisons remain exploratory and cannot establish the pre-specified H3 seed-variance criterion.

| Seed | Run ID | Selected epoch | Macro-F1 | Accuracy | Final train loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `model-a-aug-seed0-20261105T224043534656Z-c01504439ecf4d819c307b68b542a34e` | 30 | 0.7400 | 0.7656 | 0.7314 |

Compared with the reported clean seed-0 checkpoint (macro-F1 0.7665, accuracy 0.7940), the augmented run is lower by 0.0265 macro-F1 and 0.0284 accuracy on clean validation data. Thus this single run does not satisfy H3's clean-retention requirement of at most a 0.01 macro-F1 loss. Subsequent degradation diagnostics are recorded in `exploratory_validation_results.md`. These values were transcribed from user-supplied console output; artifacts and provenance have not been inspected.
