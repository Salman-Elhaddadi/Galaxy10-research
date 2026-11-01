# Research Protocol

**Galaxy morphology classification: robustness, reliability and explainability of lightweight CNNs under image-quality degradation**

- Version: `protocol-v1` (**DRAFT** until every item in Section 12 is closed, then tag it in git)
- Date: 2026-11-01
- Author: Salman El haddadi

This document is written **before any model is trained**. It fixes what will be done and how results will be judged. Anything not fixed here is exploratory and must be labeled as exploratory in the paper. Changes after tagging go in the amendment log (Section 13), never silently.

---

## 1. Research question and hypotheses

**Question.** How robust and reliable are lightweight CNNs for galaxy morphology classification under different levels of image-quality degradation, and can targeted augmentation improve robustness?

| ID | Hypothesis | Judged by (decision rule, fixed now) |
|---|---|---|
| H1 | Macro-F1 decreases as degradation severity increases | Per degradation type: macro-F1 at the highest level is lower than clean by more than the seed-to-seed std, and the curve is non-increasing within seed noise. Otherwise H1 is reported as not supported for that type. |
| H2 | Degradation types affect classes differently | Per-class F1 drop at a fixed level: H2 is supported if the bootstrap CIs of the drop do not overlap for at least two classes. Effect sizes always reported. |
| H3 | Targeted augmentation improves degraded performance | Criterion in Section 8 (Experiment 3). |
| H4 | Confidence becomes less reliable under degradation | ECE and Brier score of Model A increase with severity (same rule as H1). |

Hypotheses are tested, not assumed. Negative results are reported.

---

## 2. Dataset (facts verified in notebook 01)

Galaxy10 DECaLS: 17,736 images, 256x256x3, `uint8`, 10 classes, HDF5 (gzip). The SHA256 of the downloaded file matches the value published in the astroNN documentation.

| Class | Name | Images |
|---|---|---|
| 0 | Disturbed | 1081 |
| 1 | Merging | 1853 |
| 2 | Round Smooth | 2645 |
| 3 | In-between Round Smooth | 2027 |
| 4 | Cigar Shaped Smooth | 334 |
| 5 | Barred Spiral | 2043 |
| 6 | Unbarred Tight Spiral | 1829 |
| 7 | Unbarred Loose Spiral | 2628 |
| 8 | Edge-on without Bulge | 1423 |
| 9 | Edge-on with Bulge | 1873 |

Imbalance: 7.9:1 (class 2 vs class 4). Accuracy alone is not a valid metric; macro-F1 is primary.

---

## 3. Data audit and policies

| # | Finding (notebook 01) | Policy |
|---|---|---|
| 1 | **Exact duplicate images** (identical pixels): 60 groups, 121 images (0.68%). 60 of 60 groups conflict; 121 images. | **P-DUP** [D1]: every image in a duplicate group whose labels disagree is **excluded** from train, validation and test (no reliable label). |
| 2 | **Same sky position**: 65 pairs within 3 arcsec (5 same label, 60 different), 130 images. | **P-POS** [D2]: all pairs within 3 arcsec are merged into one group, so they never cross splits. This replaces the hardcoded list of manually reviewed pairs. Distinct images at the same position are not excluded (they can be real companions) unless they are also exact duplicates with conflicting labels. |
| 3 | **Near-constant images**: 3 (std < 2 in every channel; probably the same 3 with no detected object). These 3 images are the triplicate already excluded by P-DUP. | **P-BLANK**: excluded after visual confirmation in notebook 01 (cell 7d). |
| 4 | **Saturated images**: 31 with more than 1% saturated pixels. | Kept. Listed as a limitation. |
| 5 | **Pixel scale**: two values, 0.262 and 0.524 arcsec/pixel, present in every class. Images at 0.524: 820 (4.6%). | **P-PX** [D2b]: keep all images; do not stratify on it; report every metric overall **and** by pixel scale; log the share per split in `split_report.json`; per-class results by scale only where n >= 20. |
| 6 | **Redshift**: 92 NaN, 3 values <= 0. | Not used as input, filter or stratification key. No action. |
| 7 | **Intrinsic noise**: border-MAD median 10.87/255 (about 0.043 on the [0,1] scale), 1st-99th percentile 5.9-16.3/255. Measured on the grayscale mean at 256x256. | **P-NOISE**: degradation severity is reported as added sigma **and** as a ratio to intrinsic sigma, re-measured per channel on the 128x128 cache (check P5). |
| 8 | **Sky level**: median 33/255 (1st-99th percentile 31-38). Suggests display-stretched composites. Not verified (check P1). | Until verified, every degradation is described as a **display-space perturbation**, not a physical noise model. |

---

## 4. Preprocessing

1. Resize 256 -> 128 with `cv2.INTER_AREA`. Cache: `data/processed/galaxy10_128x128_uint8.npy` and `galaxy10_labels.npy`. Center-cropping is not used (the object-size proxy in notebook 01 is inconclusive); this is listed as a limitation.
2. Convert to float32 in [0, 1] (`uint8 / 255`).
3. Normalize per channel with mean and std computed on the **training split only** (after exclusions), stored in `train_normalization.npz`.
4. Order inside the Dataset: `uint8 -> float [0,1] -> geometric augmentation (train only) -> degradation (augmentation or evaluation) -> clip [0,1] -> normalize`.

---

## 5. Splits

1. Compute image hashes (script, committed digests), build groups with P-DUP and P-POS, apply the exclusions (P-DUP conflicts, P-BLANK).
2. `StratifiedGroupKFold`, seed 42, stratified by class, grouped by group id: 10 folds, one fold is test; from the remaining data, 9 folds, one fold is validation. Result: about 80/10/10.
3. Committed outputs in `splits/`: `train_idx.npy`, `val_idx.npy`, `test_idx.npy`, `excluded_idx.npy`, `split_report.json` (class counts and pixel-scale share per split, group counts). Indices refer to the full 17,736-image cache; nothing is re-cached.
4. Tests (`tests/test_splits.py`): partitions are disjoint, no group spans two splits, every non-excluded image appears exactly once.
5. The splits created before this protocol existed are **discarded**. No model was trained on them.
6. **Test-set rules:** never used for training, normalization statistics, model selection, hyperparameters, early stopping or choosing degradation levels. Each frozen configuration is evaluated on test once per seed.

---

## 6. Models

**Model A (baseline)**: kernel 3x3, stride 1, padding 1, no conv bias (BatchNorm follows), MaxPool 2x2, PyTorch default initialization.

| Layer | Output shape |
|---|---|
| Input | 3 x 128 x 128 |
| Conv(3->32) + BN + ReLU + MaxPool | 32 x 64 x 64 |
| Conv(32->64) + BN + ReLU + MaxPool | 64 x 32 x 32 |
| Conv(64->128) + BN + ReLU + MaxPool | 128 x 16 x 16 |
| Conv(128->256) + BN + ReLU | 256 x 16 x 16 |
| AdaptiveAvgPool(1) + Flatten + Dropout(0.4) | 256 |
| Linear(256->10) | 10 |

Hand-computed: about 0.39 M parameters and about 0.24 GMACs per forward pass (verify in code, check P8).

**Model B (optional, separate experiment)**: a lightweight residual CNN, defined in an amendment **before** it is used, with at most 2x the parameters of Model A. It is only compared to Model A in an architecture comparison, never inside the augmentation experiment (architecture stays fixed there).

The residual network currently in `src/galaxy10/model/CNN.py` (about 7.8 M parameters) is **not** part of v1.

---

## 7. Training

- Loss: cross-entropy, **no class weights or resampling** [D3]. Optimizer Adam, lr 1e-3 constant, weight decay 0, batch size 32, 30 epochs.
- Model selection: the epoch with the highest macro-F1 on the **clean** validation set, for both A and B [D7].
- Geometric augmentation, training only, **identical for A and B** [D4]: random horizontal flip, random vertical flip, rotation by a random multiple of 90 degrees (no interpolation). No test-time augmentation.
- Seeds 0, 1, 2 for every configuration (initialization, shuffling, augmentation). The split seed (42) is independent.
- Determinism: seed Python, NumPy and PyTorch, seed the DataLoader generator and workers, record library versions and hardware per run. GPU non-determinism may remain and is reported.
- Hyperparameters are fixed to the values above for v1. Any tuning is done on validation only, applies identically to A and B, and is logged.
- Before any real run: initial loss is close to ln(10) = 2.30, and the model can overfit a single batch.
- Compute: development on CPU, full runs on Kaggle GPU.

---

## 8. Experiments

### Experiment 1: baseline
Train Model A on clean data. Report on validation and test: macro-F1 (primary), accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrix, and the same metrics split by pixel scale.

### Experiment 2: degradation robustness
Evaluate trained models on degraded **test** images (validation only for diagnostics). Degradations are applied to the [0,1] 128x128 tensor before normalization, then clipped to [0,1]. Each image is degraded deterministically with an RNG seeded by (global seed, image index, type, level). Levels are frozen in `configs/degradations.yaml`.

| Type | Definition | Levels [D6] |
|---|---|---|
| A. Gaussian noise | `x + N(0, sigma^2)`, independent per pixel and channel | sigma in {0, 0.02, 0.05, 0.10, 0.15} |
| B. Gaussian blur | kernel size `2*ceil(3*sigma)+1`, reflect padding | sigma_px in {0, 0.5, 1, 2, 3} |
| C. Resolution | downsample 128 -> 128/f with `INTER_AREA`, upsample to 128 with `INTER_LINEAR` | f in {1, 2, 4, 8} |
| D. Contrast | `m + c*(x - m)`, m = mean over all pixels and channels of the image | c in {1.0, 0.75, 0.5, 0.25, 0.1} |
| E. Poisson noise (optional) | only if check P1 justifies it | to be defined by amendment |

Outputs: macro-F1 versus severity per type, per-class F1 heatmaps versus severity.

### Experiment 3: augmentation (Model A vs Model B-aug)
Same architecture, hyperparameters, seeds and geometric augmentation. The only difference: during training, each image is degraded with probability 0.5 using **one** type chosen uniformly from {Gaussian noise, Gaussian blur}, severity drawn uniformly in [0, max_train], with max_train = 0.10 (noise sigma) and 2.0 (blur sigma_px).

Results are reported in four separate groups, because gains on seen types are expected by construction:
1. seen types at seen severities (noise <= 0.10, blur <= 2),
2. seen types at **extrapolated** severities (noise 0.15, blur 3),
3. **unseen types** (resolution, contrast),
4. clean images.

**Success criterion for H3** [D5]: on groups 2 and 3, the mean macro-F1 over non-zero levels of B exceeds A by more than the seed-to-seed std **and** the clean macro-F1 of B is at most 1 percentage point below A.

### Experiment 4: calibration
For every model, type and level: ECE (15 equal-width bins, top-1 confidence), multi-class Brier score, NLL, mean confidence of correct versus incorrect predictions. Reliability diagrams for clean data and the highest level of each type. No post-hoc recalibration in v1 (temperature scaling may come later as an amendment).

### Experiment 5: Grad-CAM
Model A, seed 0 checkpoint, target layer = output of the last conv block (256x16x16), target class = predicted class. Images are selected by seeded random sampling (seed 42) **before looking at any heatmap**: per class, up to 4 correct and 4 misclassified test images, shown clean and at one mid-level of each degradation. Include a sanity check with randomized model weights (Adebayo et al., 2018, verify the reference before citing). Qualitative only; Grad-CAM is not evidence of astrophysical understanding.

---

## 9. Metrics and statistics

- Primary metric: **macro-F1**. Secondary: accuracy, balanced accuracy, per-class F1, ECE, Brier, NLL.
- Uncertainty: 95% percentile bootstrap CI over test images (1000 resamples) per seed, plus mean and std over the 3 seeds. The bootstrap captures test-sampling noise, not training variance, so both are reported.
- A claimed difference must exceed the seed-to-seed std and have a bootstrap CI excluding zero in all 3 seeds. No significance tests on 3 seeds.
- Comparisons are paired: same test images, same seeds.

---

## 10. Reproducibility and integrity rules

1. Every run writes `results/<run_id>/` with the resolved config, seed, git commit hash, metrics and figures, and `checkpoints/<run_id>/` for weights.
2. No result is edited by hand. Figures are generated by scripts from saved metrics.
3. No fabricated results or citations. No novelty claim without literature support.
4. Every deviation from this protocol is an amendment (Section 13), dated and justified.
5. Report negative results.
6. Distinguish synthetic degradation from real observing conditions in every figure caption and in the paper.

---

## 11. Limitations and literature position

Known limitations: label noise (conflicting duplicates; volunteer-vote labels), mixed pixel scales, saturated images, resized and probably display-stretched composites, synthetic degradations, a single dataset, class imbalance, and a small test set (about 1.7k images, so wide per-class CIs, especially class 4).

Literature (status, to be completed in `docs/literature.md`):
- Pandya et al., 2023, arXiv:2311.01500: robustness of equivariant networks on Galaxy10 DECaLS with Poisson noise and one-pixel attacks (read).
- A&A 683, A42 (2024), CvT galaxy morphology classification with Gaussian and salt-and-pepper noise: **noise section not yet read**; verify before citing.
- Ciprijanovic et al., arXiv:2112.14299 (DeepAdversaries): perturbations and domain adaptation for galaxy morphology; verify before citing.
- DeepAstroUDA, arXiv:2302.02005 (listed in the astroNN docs).

Because noise robustness on Galaxy10 DECaLS is already studied, **H1 is an expected result, not a contribution**. A possible contribution is a reproducible study combining several degradation types, calibration under degradation, per-class sensitivity and targeted augmentation with lightweight CNNs. **No novelty claim is made** until a systematic literature search (check P6) supports it.

---

## 12. Open items before tagging `protocol-v1`

**Decisions to confirm or change (defaults written above):**
- D1 exclude conflicting-label exact duplicates
- D2 merge all pairs within 3 arcsec into one group; D2b keep both pixel scales and report by scale
- D3 plain cross-entropy without class weights
- D4 geometric augmentation: flips and 90-degree rotations for both models
- D5 success criterion thresholds (1 percentage point, seed std)
- D6 degradation levels for types B, C and D
- D7 model selection on clean validation macro-F1

**Checks to complete:**
- P1 verify how the images were produced (Legacy Surveys documentation): is the stretch known?
- P2 count images at pixel scale 0.524 per class
- P3 count conflicting-label duplicate groups and images, per class
- P4 inspect the 3 near-constant images
- P5 measure intrinsic noise per channel on the 128x128 cache
- P6 read the CvT paper's noise section; search for calibration-under-degradation work on galaxy classifiers
- P7 update `make_splits.py` to implement Section 5, regenerate and commit the splits
- P8 benchmark one forward/backward pass of Model A on the CPU

---

## 13. Amendment log

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-11-01 | protocol-v1 draft | Initial draft | n/a |
