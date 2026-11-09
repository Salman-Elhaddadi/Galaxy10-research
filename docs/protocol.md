# Research Protocol

Status update 2026-11-09: matched clean and augmented Model A seed-0 draft checkpoints and 20-condition validation evaluations are now saved and verified; see `docs/exploratory_validation_results.md`. This does not freeze or amend the protocol. Missing seeds, deliberate freeze and formal test access remain open.

**Galaxy morphology classification: robustness, reliability and explainability of lightweight CNNs under image-quality degradation**

- Version: `protocol-v1` (**DRAFT** until every item in Section 12 is closed, then tag it in git)
- Date: 2026-11-01
- Author: Salman El haddadi

The initial draft preceded the reported draft training. It fixes what will be done and how results will be judged; subsequent engineering decisions are logged in Section 13. The reported Model A seed-0 epoch-28 validation macro-F1 0.7767 / accuracy 0.8053 remains preliminary, not a final test result. Anything not fixed here is exploratory and must be labeled as exploratory in the paper. Changes after tagging go in the amendment log, never silently.

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
| 8 | **Sky level**: median 33/255 (1st-99th percentile 31-38). The official builder downloaded DR8 viewer JPEG cutouts (`bands=grz`); exact viewer stretch parameters are not preserved (P1; `docs/literature.md`). | Every degradation is a **display-space perturbation**, not a physical noise model. Poisson noise is not enabled. |

---

## 4. Preprocessing

1. Resize 256 -> 128 with `cv2.INTER_AREA`. Cache: `data/processed/galaxy10_128x128_uint8.npy` and `galaxy10_labels.npy`. Center-cropping is not used (the object-size proxy in notebook 01 is inconclusive); this is listed as a limitation.
2. Convert to float32 in [0, 1] (`uint8 / 255`).
3. Normalize per channel with mean and std computed on the **training split only** (after exclusions), stored in `train_normalization.npz`.
4. Order inside the Dataset: `uint8 -> float [0,1] -> geometric augmentation (train only) -> degradation (augmentation or evaluation) -> clip [0,1] -> normalize`.

---

## 5. Splits

1. Compute image hashes (script, committed digests), build groups with P-DUP and P-POS, apply the exclusions (P-DUP conflicts, P-BLANK).
2. `StratifiedGroupKFold`, seed 42, shuffle enabled, stratified by class, grouped by group id: 10 folds, the **first yielded fold** is test; from the remaining data, 9 folds, the **first yielded fold** is validation. Result: about 80/10/10. Group id is the minimum original index in each transitive exact-duplicate/sky-position component. Exclusions are applied before splitting; retained members of a component cannot cross splits.
3. Committed outputs in `splits/`: `train_idx.npy`, `val_idx.npy`, `test_idx.npy`, `excluded_idx.npy`, `split_report.json` (class counts and pixel-scale share per split, group counts). Indices refer to the full 17,736-image cache; nothing is re-cached.
4. Tests (`tests/test_research.py`): partitions are disjoint, no group spans two splits, every non-excluded image appears exactly once, and excluded images occur in no split. Commit per-image SHA256 digests, group ids and pixel scales with the report. Raw digests are recomputed independently of notebook caches; previous split artifacts are preserved in content-addressed history before regeneration.
5. The 2026-11-04 raw audit reproduced all four existing index arrays byte for byte (14,091 / 1,762 / 1,762, 121 exclusions). The existing normalization file did not reproduce training-only statistics and lacked provenance; its preserved SHA256 is `31ac15e742482cfc3764733c7cee41e8cc7cdcf0316119130a18d5026a2794c0`. Corrected normalization SHA256 is `21ccf07cc433c5373f8f961fc88a80e2359c6646792925e3d2c5b74be987aa7b`. Existing results using old/missing normalization provenance are preliminary and cannot be pooled with corrected runs. No historical run/checkpoint is modified.
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

The current Model B in `src/galaxy10/model/residual.py` has 4,904,618 parameters, versus Model A's 391,466 (cap: 782,932). It is preserved as **exploratory**, excluded from protocol-compliant architecture comparisons. No compliant residual Model B is currently implemented. `src/galaxy10/model/CNN.py` contains the unchanged Model A baseline. See the 2026-11-04 amendment.

---

## 7. Training

- Loss: cross-entropy, **no class weights or resampling** [D3]. Optimizer Adam, lr 1e-3 constant, weight decay 0, batch size 32, 30 epochs.
- Model selection: the epoch with the highest macro-F1 on the **clean** validation set, for both A and B [D7].
- Macro-F1 uses the fixed 10 classes, with undefined class F1 set to zero. Exact ties retain the earliest epoch. No gradient clipping or learning-rate schedule is applied.
- Geometric augmentation, training only, **identical for A and B** [D4]: random horizontal flip, random vertical flip, rotation by a random multiple of 90 degrees (no interpolation). No test-time augmentation.
- Seeds 0, 1, 2 for every configuration (initialization, shuffling, augmentation). The split seed (42) is independent.
- Determinism: seed Python, NumPy and PyTorch (CPU and CUDA), seed the DataLoader generator and workers, record library versions and hardware per run. Enable deterministic PyTorch algorithms, disable cuDNN benchmarking and configure deterministic cuBLAS; unsupported operations fail rather than silently use nondeterministic kernels. Reproducibility is scoped to the same software, hardware and worker count; cross-platform equality is not guaranteed.
- The training entry point feeds only training and clean validation examples to the model. Test evaluation is a separate command after configuration freezing; training does not load test indices or use test examples/metrics for learning or selection. The shared complete cache is byte-hashed for provenance; this includes test-image bytes solely for integrity, without test predictions or test-derived training statistics. It verifies development group/exclusion isolation and normalization provenance. Formal training must record the reviewed frozen manifest via `--protocol-manifest`; default runs stay draft. Model B stays exploratory even with a manifest.
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

Implementation convention: `configs/degradations.yaml` is the authoritative frozen level list, including identity levels in each curve, global evaluation seed 42 and Grad-CAM mid-levels (noise 0.05, blur 1, resolution 4, contrast 0.5). Noise seed is derived with SHA256 from `(seed, original index, type, float severity hex)`, independent of Python hash randomization, worker or iteration order. Blur uses a normalized discrete Gaussian and reflect padding. All are display-space perturbations. Intrinsic reference noise is measured on **training images only**, using per-channel `1.4826 * MAD` on an 8px border at 128px; this descriptive reference does not select severity levels.

### Experiment 3: augmentation (Model A-clean vs Model A-aug)
Same architecture, hyperparameters, seeds and geometric augmentation. The only difference: during training, each image is degraded with probability 0.5 using **one** type chosen uniformly from {Gaussian noise, Gaussian blur}, severity drawn uniformly in [0, max_train], with max_train = 0.10 (noise sigma) and 2.0 (blur sigma_px).

The augmentation condition uses a separate seeded worker RNG stream for its Bernoulli/type/severity/noise draws, preserving the geometric RNG sequence of the clean condition. Every config, checkpoint and summary identifies `condition=clean|aug`; B-aug is rejected.

Results are reported in four separate groups, because gains on seen types are expected by construction:
1. seen types at seen severities (noise <= 0.10, blur <= 2),
2. seen types at **extrapolated** severities (noise 0.15, blur 3),
3. **unseen types** (resolution, contrast),
4. clean images.

**Success criterion for H3** [D5]: on groups 2 and 3, the mean macro-F1 over non-zero levels of A-aug exceeds A-clean by more than the seed-to-seed std **and** the clean macro-F1 of A-aug is at most 1 percentage point below A-clean. These labels refer to training conditions on the same Model A architecture, never the exploratory residual Model B.

### Experiment 4: calibration
For every model, type and level: ECE (15 equal-width bins, top-1 confidence), multi-class Brier score, NLL, mean confidence of correct versus incorrect predictions. Reliability diagrams for clean data and the highest level of each type. No post-hoc recalibration in v1 (temperature scaling may come later as an amendment).

ECE bins are left-closed/right-open except the final bin includes confidence 1. Brier is the mean **sum** of ten squared probability errors. NLL is computed from log-softmax; means for empty correct/incorrect subsets are `null`, with valid bootstrap resample counts recorded. Balanced accuracy averages recalls over classes with nonzero support; macro-F1 always averages all ten. Pixel-scale per-class precision/recall/F1 is suppressed (`null`) below support 20, with support retained.

### Experiment 5: Grad-CAM
Model A, seed 0 checkpoint, target layer = output of the last conv block (256x16x16), target class = predicted class. Images are selected by seeded random sampling (seed 42) **before looking at any heatmap**: per class, up to 4 correct and 4 misclassified test images, shown clean and at one mid-level of each degradation. Include a sanity check with randomized model weights (Adebayo et al., 2018, verify the reference before citing). Qualitative only; Grad-CAM is not evidence of astrophysical understanding.

Use saved clean evaluation predictions to sample sorted candidate indices within true class and clean correctness, persist `selection.json` before generating heatmaps, hook final block ReLU `features[14]` (before adaptive pooling), and target each view's predicted class. Randomization is a fresh complete Model A initialization including BatchNorm reset, seed 42, evaluation mode; save targets for trained/randomized views. Validation examples are permitted for workflow diagnostics and explicitly labeled validation; final qualitative views require the same frozen-test authorization as evaluation. This implementation does not assert a passed scientific sanity-check interpretation.

---

## 9. Metrics and statistics

- Primary metric: **macro-F1**. Secondary: accuracy, balanced accuracy, per-class F1, ECE, Brier, NLL.
- Uncertainty: 95% percentile bootstrap CI over test images (1000 resamples) per seed, plus mean and std over the 3 seeds. The bootstrap captures test-sampling noise, not training variance, so both are reported.
- A claimed difference must exceed the seed-to-seed std and have a bootstrap CI excluding zero in all 3 seeds. No significance tests on 3 seeds.
- Comparisons are paired: same test images, same seeds.

Bootstrap uses NumPy `default_rng(42)` and 1,000 uniform samples of size N with replacement over selected images. Save 2.5/97.5 percentiles for scalar and per-class metrics, and paired **clean-minus-degraded** intervals with identical resampling indices for F1 drops. No class-stratified or group bootstrap is substituted. Aggregate saved-best-checkpoint results with arithmetic mean and **sample** std (`ddof=1`); singleton std is `null`. Reject duplicate seeds within a comparison group and label missing seeds. Never pool architecture, condition, source/protocol/data version, settings, split or severity. No hypothesis conclusion is inferred automatically.

---

## 10. Reproducibility and integrity rules

1. Every run writes `results/<run_id>/` with the resolved config, seed, git commit hash, metrics and figures, and `checkpoints/<run_id>/` for weights.
   The trainer reserves fresh directories using model, condition, seed, UTC timestamp and UUID, and never reuses a run directory. It writes `config.json`, per-epoch `metrics.json`, `summary.json`, and `best.pt` (model and optimizer states, selected epoch, validation metrics and config). Configs record dirty Git status, source/protocol/input SHA256 hashes, library versions and hardware. Evaluation, aggregation and Grad-CAM also reserve unique result directories. All current real runs remain draft; Model B additionally carries an explicit exploratory exclusion label.
2. No result is edited by hand. Figures are generated by scripts from saved metrics.
3. No fabricated results or citations. No novelty claim without literature support.
4. Every deviation from this protocol is an amendment (Section 13), dated and justified.
5. Report negative results.
6. Distinguish synthetic degradation from real observing conditions in every figure caption and in the paper.
7. Frozen-test access requires explicit `--frozen-test` and a manifest generated by `scripts/freeze_protocol.py --confirm-frozen` from a reviewed evidence-backed checklist. It binds source, protocol, configuration, cache, normalization and split fingerprints. Missing/open/stale manifests fail before test indices/pixels are loaded. Incompatible checkpoint input provenance cannot be overridden for test. Each checkpoint/freeze reserves a test-access ledger; use the combined clean+degradation suite once per seed. Failed access reservations remain auditable. No real test evaluation is performed while Section 12 is open.

---

## 11. Limitations and literature position

Known limitations: label noise (conflicting duplicates; volunteer-vote labels), mixed pixel scales, saturated images, resized and probably display-stretched composites, synthetic degradations, a single dataset, class imbalance, and a small test set (about 1.7k images, so wide per-class CIs, especially class 4).

The scoped primary-source review is recorded in `docs/literature.md`. Direct prior art includes Pandya et al. (2023) on Galaxy10 DECaLS under Poisson noise and one-pixel attacks, Cao et al. (2024) on galaxy classification under Gaussian and salt-and-pepper noise, and Ćiprijanović et al. (2022) on simulated-galaxy perturbations and domain adaptation. Hendrycks and Dietterich (2019), Guo et al. (2017), Ovadia et al. (2019), and Adebayo et al. (2018) establish the common-corruption, calibration-under-shift, and saliency-sanity context used by this protocol.

Because noise robustness on Galaxy10 DECaLS is already studied, **H1 is an expected result, not a contribution**. The defensible contribution, conditional on results, is the combined reproducible treatment of four display-space degradation types, calibration, per-class and pixel-scale sensitivity, paired uncertainty, controlled augmentation, and randomized-weight qualitative checks in a lightweight model. No generic novelty claim is made for noise robustness.

---

## 12. Open items before tagging `protocol-v1`

**Decisions applied on 2026-11-04 (written defaults retained):**
- D1 exclude conflicting-label exact duplicates
- D2 merge all pairs within 3 arcsec into one group; D2b keep both pixel scales and report by scale
- D3 plain cross-entropy without class weights
- D4 geometric augmentation: flips and 90-degree rotations for both models (confirmed by the 2026-11-04 workflow request)
- D5 success criterion thresholds (1 percentage point, seed std)
- D6 degradation levels for types B, C and D
- D7 model selection on clean validation macro-F1 (confirmed by the 2026-11-04 workflow request)

**Checks and actual status (evidence also in `configs/protocol_checklist.json`):**
- P1 **VERIFIED WITH LIMITATION**: the official Galaxy10 builder downloads Legacy Survey DR8 viewer JPEG cutouts with `bands=grz`; these are rendered RGB products, not linear photon counts. Exact viewer stretch parameters are not preserved, so no physical Poisson model is enabled. See `docs/literature.md`.
- P2 **VERIFIED** from raw metadata/notebook: at scale 0.524 per class `[34,103,74,30,2,211,147,107,63,49]`; see split report.
- P3 **VERIFIED** from independently recomputed raw SHA256: 60 conflicting groups / 121 images; per-class images `[30,14,10,6,1,5,10,33,4,8]`.
- P4 **VERIFIED**: saved notebook panel inspected; indices 11492,12659,15422 are featureless repeated frames, already excluded by P-DUP. New unreviewed blank candidates fail preparation.
- P5 **MEASURED**, training only at 128px: per-channel median border-MAD sigma `[0.05232707,0.04069882,0.04069882]`; full percentiles and added/intrinsic ratios in the preparation audit.
- P6 **VERIFIED FOR PROTOCOL-V1 SCOPE**: direct Galaxy10 and galaxy-robustness work, common-corruption evaluation, calibration under shift, and saliency sanity checks are reviewed and positioned in `docs/literature.md`. Noise robustness itself is explicitly not claimed as novel.
- P7 **IMPLEMENTED/REGENERATED/TESTED**: all original index hashes reproduced, group/exclusion invariants pass; split and provenance outputs, including history, are committed. Protocol review and tagging still remain before freezing.
- P8 **MEASURED**: CPU one-thread, synthetic batch 1, 128px: forward 0.0263 s, backward 0.0435 s (single unwarmed measurement), 391,466 parameters, 240,650,752 Conv/Linear MACs. Not a throughput benchmark or a real-data overfit test.

Evidence: `results/preparation-audit-20261104T121403331729Z-0f858d79184644fbab03d1d6cb8e3e00/audit.json`, `splits/split_report.json`, `results/blank_inspection_from_notebook.png`, `docs/engineering_validation.md`, and `docs/literature.md`. The existing real-training single-batch preflight **passed on CPU**: initial loss 2.3018, final loss 0.0372 / batch accuracy 1.0 at step 40. Its first-16 default is all class 0, so this is explicitly a limited memorization check, not a multiclass learning result. Evidence is in `results/single-batch-sanity-20261104T124012471573Z-9b7039568d6a4292a0cd90805bed5eb6/sanity.json`. **GPU validation, review/commit/tag, and the deliberate frozen-test decision remain pending.** This protocol stays draft and no real freeze manifest was created in this phase.

---

## 13. Amendment log

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-11-01 | protocol-v1 draft | Initial draft | n/a |
| 2026-11-04 | protocol-v1 draft | Preserve Model A and the existing residual Model B unchanged; identify B as exploratory and exclude it from protocol-compliant comparisons because it exceeds the unchanged 2x cap. Correct the stale CNN.py residual reference. | Retain the baseline and exploratory work without silently substituting an architecture or relaxing the cap. |
| 2026-11-04 | protocol-v1 draft | Align training to clean validation macro-F1, Adam lr 1e-3 / weight decay 0, seeds 0/1/2, batch size 32, 30 epochs and identical geometric augmentation. Remove the unprescribed gradient-value clipping. Define fixed ten-class macro-F1, zero-division handling and earliest-epoch tie selection. | Correct implementation drift and make selection unambiguous before any formal results. |
| 2026-11-04 | protocol-v1 draft | Require deterministic algorithms in the trainer, seed loaders/workers, record resolved settings and provenance, reserve unique result/checkpoint directories, and separate training from future test evaluation. | Make runs traceable and repeatable and protect the test-set freeze rule. |
| 2026-11-04 | protocol-v1 draft | Rename Experiment 3 conditions to Model A-clean and Model A-aug; keep its architecture, degradation policy and success thresholds unchanged. | Remove the ambiguous B-aug label, which could be mistaken for the separate residual architecture. |
| 2026-11-04 | protocol-v1 draft | Apply D1-D7 defaults, select the first nested SGKF folds (10 then 9, shuffle, seed 42), canonicalize transitive group ids, recompute raw SHA256 digests, preserve old artifacts and validate exclusions/groups before saving. Retain only the three repository-confirmed P-BLANK exclusions. | Make Section 5 deterministic/auditable without notebook dependencies; existing split arrays reproduced byte for byte. |
| 2026-11-04 | protocol-v1 draft | Recompute training-only normalization with provenance; preserve old file; mark old/missing-provenance runs preliminary. | Existing normalization differs numerically despite unchanged indices; prevent silent comparability claims. |
| 2026-11-04 | protocol-v1 draft | Freeze the written degradation levels and explicit mid-levels in YAML, deterministic SHA256-derived per-image seeds, pre-normalization clipping; measure intrinsic noise on training images with an 8px border at 128px. | Implement display-space experiments without consulting test performance or changing severity defaults. |
| 2026-11-04 | protocol-v1 draft | Implement separate strict-best-checkpoint evaluation with explicit split selection, evidence-backed freeze manifest and test flag, unique outputs and test-access ledger. Require matching checkpoint data provenance; validation override stays preliminary. | Prevent accidental test development, repeated passes and promotion of incompatible/draft checkpoints to formal results. |
| 2026-11-04 | protocol-v1 draft | Define fixed metrics/calibration bin conventions, 1,000 seed-42 image bootstrap, paired clean-minus-degraded drop intervals, support threshold 20 for per-class pixel-scale results, and sample std across separated seed groups. | Resolve metric/statistical implementation ambiguity without changing primary hypotheses. |
| 2026-11-04 | protocol-v1 draft | Implement Model A-aug exactly as written with separate augmentation RNG and explicit condition metadata; reject B-aug. | Preserve architecture/settings/geometric draws while changing only the prescribed training degradation. |
| 2026-11-04 | protocol-v1 draft | Preselect Grad-CAM examples from saved predictions; persist selection before maps; target final block ReLU and each view's prediction; fully reinitialize weights/BN at seed 42 for sanity views. | Make selection and randomization auditable; validation smoke maps are diagnostic, with no scientific interpretation claimed. |
| 2026-11-04 | protocol-v1 draft | Add six-run A matrix, separately labelled exploratory B commands, best-checkpoint aggregation, fingerprint-checked Kaggle preparation and synthetic CPU workflow checks. Leave P1/P6, GPU runs and deliberate freeze pending. | Complete implementable engineering without inventing literature findings, scientific performance or completed external checks. |
| 2026-11-04 | protocol-v1 draft | Preserve the existing first-16 single-batch preflight default and early-stop criterion, save unique machine-readable evidence, and fail the command if overfitting is not achieved. Record the passed CPU check and its all-class-0 selection limitation. Keep paired-bootstrap NLL in log space even when probabilities underflow. | Prevent silent preflight success and inaccurate confidence intervals without changing research training settings or inventing a multiclass sanity result. |
| 2026-11-05 | protocol-v1 draft | Close P1 with evidence that Galaxy10 uses rendered Legacy Survey DR8 `grz` viewer JPEG cutouts; retain display-space language and reject a physical Poisson interpretation. Close P6 for the scoped v1 review and record direct prior art, calibration-under-shift context, saliency sanity checks, and a narrowed contribution boundary in `docs/literature.md`. | Resolve the two external evidence checks without overstating pixel physics or novelty. |
