# Galaxy10 Research

Reproducible galaxy morphology classification, display-space degradation, calibration and qualitative Grad-CAM. The [protocol](docs/protocol.md) remains **draft**. No frozen test evaluation has been performed in this engineering phase.

## Current status and comparability

The user-reported Model A seed 0 draft selected epoch 28, with **clean validation** macro-F1 **0.7767** and accuracy **0.8053**. Those values were supplied as prior results, not reproduced here. They are not test results. The preserved `notebooks/kaggle_latest/` notebooks embed an older source snapshot; keep them as historical artifacts, and use the commands below for new runs.

The raw-data audit reproduced the existing split arrays **byte for byte**: train 14,091, validation 1,762, test 1,762; 121 excluded images, 60 conflicting duplicate groups, 65 sky-position pairs, no group crossing partitions. Original split/report snapshots are retained under `splits/history/`. The report and new group/digest/scale artifacts add provenance; no image indices moved.

**Normalization changed numerically.** The old file had SHA256 `31ac15e742482cfc3764733c7cee41e8cc7cdcf0316119130a18d5026a2794c0`. The corrected training-only file has SHA256 `21ccf07cc433c5373f8f961fc88a80e2359c6646792925e3d2c5b74be987aa7b`. Maximum absolute changes: mean `0.00005249`, std `0.00015838`. The old file is retained in `data/processed/history/`. Its original split provenance was absent. Consequently the existing seed-0 draft, and any other runs using that normalization or lacking matching provenance, remain **preliminary** and must not be pooled with corrected runs. Existing Kaggle notebooks/runs/checkpoints were not overwritten. Image-cache and label fingerprints are recorded in `configs/prepared_data.json`; this phase did not rebuild them.

| Result category | Meaning |
| --- | --- |
| Validation diagnostic | Clean validation model selection or diagnostic evaluation; never a final test result |
| Preliminary/draft | Protocol prerequisites remain open, old normalization, or missing/mismatched provenance |
| Exploratory | Model B (4,904,618 parameters) exceeds the unchanged 782,932 cap; always excluded from formal architecture comparisons |
| Formal protocol-eligible | Model A, fixed settings, recorded frozen protocol at training, matching fingerprints and deliberate frozen test access; none newly produced here |
| Frozen test result | Separate evaluator with `--frozen-test` and a matching freeze manifest; this alone does not promote a draft checkpoint to formal eligibility |

Model A remains unchanged at 391,466 parameters. Architecture and augmentation conditions are separate fields in every new training artifact. Protocol eligibility is never inferred from a final-epoch score.

## Install and prepare

Use a CPython environment with an appropriate PyTorch build (GPU build on Kaggle). From the writable repository root:

```bash
python -m pip install -e '.[research,test]'
python -m pytest -q
```

Minimum declared Python is 3.9; reproducing numeric artifacts also requires compatible library versions. Runs record versions, hardware, seeds and worker count. `python` on the audited Windows machine defaults to MSYS Python without project dependencies; the validation log identifies the scientific CPython environment actually used.

For fresh preparation, place the verified raw dataset at `data/raw/Galaxy10_DECals.h5`. Only run `make_dataset.py` if the image/label cache is absent: it creates the 128px `INTER_AREA` cache. It must not be used to silently replace a prepared cache.

```bash
python scripts/make_dataset.py
python scripts/make_splits.py
python scripts/compute_normalisation.py
python scripts/check_spatial_leakage.py
python scripts/check_data.py
python scripts/audit_preparation.py
```

`make_splits.py` independently recomputes SHA256 digests from raw pixels; it needs no notebook statistics/pickle. Conflicting exact duplicates and the three reviewed blanks are excluded; all exact duplicates and all sky matches within 3 arcsec are transitively grouped. It uses the **first** fold of shuffled 10-fold `StratifiedGroupKFold`, then the first fold of 9-fold SGKF, both seed 42. It validates before saving, snapshots previous files, writes deterministic `splits/split_report.json` plus committed `image_digests.csv`, `group_ids.npy`, `pixel_scale.npy`, and puts old/new fingerprint comparisons in a unique `results/split-audit-*` directory. New unreviewed blank candidates cause failure. Regeneration is an audit operation; **use committed splits on Kaggle** to avoid library-version drift.

Normalization uses only current training indices and records cache/split/file hashes in `data/processed/normalization.json`. Training rejects missing group audit or stale normalization provenance. `check_data.py` reads development images by default; metadata-only split auditing can inspect test indices without using test images or metrics for development. `audit_preparation.py` measures per-channel border MAD **on training images only**, using an 8px border at 128px, and benchmarks one synthetic CPU forward/backward pass. It does not establish the real-data overfit prerequisite.

## Exact next Kaggle GPU commands

Start in a **writable checkout of this updated repository** under `/kaggle/working`, with Galaxy10 attached as an input. These commands do not modify historical notebooks or runs:

```bash
python -m pip install -e '.[research,test]'
python scripts/prepare_kaggle.py
python -m pytest -q
python scripts/check_model_sanity.py

python experiments/train.py --model a --condition clean --seed 0 --device cuda --num-workers 0
python experiments/train.py --model a --condition clean --seed 1 --device cuda --num-workers 0
python experiments/train.py --model a --condition clean --seed 2 --device cuda --num-workers 0
python experiments/train.py --model a --condition aug --seed 0 --device cuda --num-workers 0
python experiments/train.py --model a --condition aug --seed 1 --device cuda --num-workers 0
python experiments/train.py --model a --condition aug --seed 2 --device cuda --num-workers 0
```

Preparation discovers the sole `/kaggle/input/**/Galaxy10_DECals.h5` (or pass `--raw-source /actual/path`), verifies the raw and cached data, uses committed splits, recomputes/validates normalization and checks development batches. Fingerprint differences fail explicitly for review. Keep installed Kaggle PyTorch; avoid replacing it with a CPU build. The sanity command prints initial loss, attempts to overfit 16 real training examples, saves a unique preflight JSON and exits with failure if the existing loss/accuracy criterion is not met; inspect/retain its output before full runs. It passed locally on CPU (initial loss 2.3018, loss 0.0372 at step 40); the existing first-16 selection is all class 0, so this is a limited memorization preflight. Repeat it on the GPU environment. These six runs remain **draft** until the protocol is deliberately frozen. For formal runs after freezing, add `--protocol-manifest configs/frozen_protocol.json` to each training command; do not retroactively relabel earlier runs.

Optional residual runs remain explicitly exploratory:

```bash
python experiments/train.py --model b --condition clean --seed 0 --device cuda --num-workers 0
python experiments/train.py --model b --condition clean --seed 1 --device cuda --num-workers 0
python experiments/train.py --model b --condition clean --seed 2 --device cuda --num-workers 0
```

`python experiments/run_matrix.py` prints the six commands without executing. Add `--execute` to run them sequentially, `--include-exploratory-b` to append B seeds, and, only after freezing, `--protocol-manifest configs/frozen_protocol.json`. It stops on failure.

Training: unweighted cross-entropy, Adam lr `1e-3`, weight decay `0`, batch `32`, `30` epochs, no scheduler/clipping. Selection maximizes **clean validation fixed-ten-class macro-F1**, zero division = 0, earliest epoch on ties. Both A conditions use identical independent horizontal/vertical flips at 0.5 and uniform rot90. A-aug alone applies, with probability **0.5**, exactly **one** uniformly chosen noise/blur degradation, with sigma uniform `[0,0.10]` / `[0,2.0]`, before clipping and normalization. Its separate RNG preserves the geometric RNG sequence.

Each run creates unique `results/<run_id>/{config,metrics,summary}.json` and `checkpoints/<run_id>/best.pt`, recording condition, selection epoch, settings, source/data/protocol hashes, Git status, versions and hardware. Training does not load test indices or feed test examples to the model. The shared full cache is byte-hashed for integrity, including bytes belonging to test images; hashing produces no test predictions/statistics and does not influence training or selection. Determinism is scoped to the same software, hardware and worker count; unsupported deterministic kernels fail.

## Saved-checkpoint validation and degradations

Replace `BEST` with the actual **saved `best.pt`** from one completed run, and `EVAL` with the new printed evaluation directory:

```bash
BEST=checkpoints/ACTUAL_RUN_ID/best.pt
python experiments/evaluate.py --checkpoint "$BEST" --split val --device cuda
python experiments/evaluate.py --checkpoint "$BEST" --split val --degradations --device cuda
```

For an old checkpoint with missing/different input provenance, validation-only diagnostics can explicitly add `--allow-preliminary`. This does not reproduce its original normalization or make scores comparable. Raw state dictionaries/final-epoch checkpoints without saved selection/config metadata are rejected; architecture states load strictly. No provided historical Kaggle checkpoint was evaluated here.

`configs/degradations.yaml` freezes Gaussian noise `[0,.02,.05,.10,.15]`, blur `[0,.5,1,2,3]`, resolution factors `[1,2,4,8]`, contrast `[1,.75,.5,.25,.1]`; zero/identity levels remain in each curve. Perturbations act on RGB float `[0,1]`, **before normalization**. Blur uses radius `ceil(3*sigma)` and reflect padding; resolution uses OpenCV `INTER_AREA` down / `INTER_LINEAR` up; contrast uses the image-wide RGB mean. Noise uses a stable SHA256-derived seed from global seed 42, original image index, type and numeric severity, independent of loading order. These are display-space perturbations; no physical observing-noise or Poisson claim is made.

Every unique evaluation directory contains resolved config/provenance, predictions (`NPZ` and `CSV`, all ten probabilities and original indices), loss, accuracy, balanced accuracy, fixed-ten-class macro-F1, per-class precision/recall/F1, confusion matrix, support, calibration, confidence intervals and summary. The suite adds `severity_metrics.csv`, per-level JSON, macro-F1 curves, per-class F1 heatmaps and clean/highest-severity reliability diagrams. Exposure labels distinguish clean, seen types/seen severity, seen types/extrapolated severity and unseen types. Scores are fractions.

Calibration: 15 equal-width top-confidence ECE bins (confidence 1 belongs to the last bin), multiclass Brier = mean sum of squared probability errors, NLL from log-softmax, and mean confidence for correct/incorrect cases (`null` if no such cases). Balanced accuracy averages recalls over classes present; macro-F1 always averages ten classes. Metrics are also reported per pixel scale; per-class scale results with fewer than 20 images are `null`.

95% percentile bootstrap uses **1,000 image resamples**, seed 42, including per-class intervals. Degraded runs also save paired **clean-minus-degraded** drop intervals using identical resample indices. It quantifies image sampling uncertainty, not seed variance. No automatic scientific hypothesis conclusions are generated.

## Deliberate protocol freeze and test access

`configs/protocol_checklist.json` records completed checks and unresolved evidence. **P1 (image production/stretch), P6 (literature) and the deliberate test decision remain open.** Read primary sources, document evidence, review/update/commit/tag the protocol and supply a truthful completed checklist before freezing. Do not simply turn pending checks true. The recorded CPU preflight passed its existing criterion; its single-class batch limitation remains explicit.

After those decisions, the commands are:

```bash
python scripts/freeze_protocol.py --checklist configs/protocol_checklist.json --output configs/frozen_protocol.json --confirm-frozen
# Train each formal Model A run with --protocol-manifest configs/frozen_protocol.json first.
python experiments/evaluate.py --checkpoint "$BEST" --split test --frozen-test --freeze-manifest configs/frozen_protocol.json --degradations --device cuda
```

The freeze records protocol, source, config and all preparation/split hashes. Any subsequent change invalidates it. Real test evaluation requires both explicit flags, a matching manifest and checkpoint input provenance; `--allow-preliminary` cannot bypass test compatibility. The evaluator reserves a once-per-checkpoint/freeze test-access ledger under `results/test_access/`; failed reservations remain auditable. Run the **full clean + degradation suite once per seed** to avoid consuming a clean-only pass first. Grad-CAM then reuses saved predictions; it does not rerun model selection. No test command above was executed on real data during this phase.

## Grad-CAM and aggregation

```bash
EVAL=results/ACTUAL_VALIDATION_EVALUATION_DIRECTORY
python experiments/gradcam.py --evaluation-dir "$EVAL" --device cuda
```

For frozen-test Grad-CAM, supply the frozen-test evaluation directory plus `--frozen-test --freeze-manifest configs/frozen_protocol.json`. Only Model A training seed 0 is accepted. Selection is seed 42, per true class up to four correct and four incorrect clean examples, saved to `selection.json` **before heatmap generation**. The target is the final block ReLU output (`features[14]`, 256x16x16); the target class is each view's prediction. Clean and configured mid-level noise `.05`, blur `1`, resolution `4`, contrast `.5` are shown with trained and fully randomized weights (seed 42, fresh initialization/BN). Outputs include selected indices/settings, per-view predicted targets, numeric heatmaps and figures. This is a qualitative sanity workflow; literature verification and interpretation remain outstanding.

Explicitly supply each completed run's best checkpoint (paths below are placeholders):

```bash
python experiments/aggregate.py --checkpoints checkpoints/RUN_SEED0/best.pt checkpoints/RUN_SEED1/best.pt checkpoints/RUN_SEED2/best.pt
python experiments/aggregate.py --checkpoints checkpoints/RUN_SEED0/best.pt checkpoints/RUN_SEED1/best.pt checkpoints/RUN_SEED2/best.pt --evaluations results/EVAL_SEED0 results/EVAL_SEED1 results/EVAL_SEED2
```

Aggregation reads saved selected-epoch validation metrics, never the final history row. JSON/CSV groups separate architecture, augmentation condition, source/protocol/data fingerprints, training settings, split and severity. Duplicate seeds in a group fail; missing seeds are explicit. Mean and **sample** standard deviation (`ddof=1`) are saved; singleton std is `null`. Model B is never formal. Seed variance and bootstrap uncertainty are distinct.

## Validation and remaining work

```bash
python -m pytest -q
python -m pytest tests/test_research.py -q
python experiments/run_matrix.py --include-exploratory-b
python experiments/evaluate.py --help
python experiments/gradcam.py --help
python experiments/aggregate.py --help
```

Tests use synthetic CPU fixtures for split/exclusion/group isolation, byte-reproducible reports, deterministic perturbations, targeted policy/RNG isolation, hand-calculated metrics, reproducible bootstrap, freeze protection, strict checkpoints, one tiny augmented Model A epoch, all 20 evaluation views, output schemas/figures, randomized Grad-CAM, aggregation and entry points. They do not establish scientific performance. The [engineering validation record](docs/engineering_validation.md) records actual outcomes and limitations.

Remaining: external primary-source and literature checks; GPU seed matrix and validation diagnostics; deliberate protocol/split freeze; then one frozen test suite per eligible seed and qualitative interpretation. No literature findings, new full-run model performance or real test scores are claimed.
