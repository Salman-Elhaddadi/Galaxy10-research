# Exploratory validation results — updated 2026-11-09

## Verified saved-artifact update (2026-11-09)

The new clean seed-0 run `model-a-seed0-20261108T204027801805Z-9db1a3778b9b478aada60ddc853e86dc` and augmented seed-0 run `model-a-aug-seed0-20261109T005633820349Z-bf3f0163f5084ed5833222bd5b285c8f` are preserved in the two `*_complete.zip` archives named in the README. Each archive passed ZIP integrity verification and contains `best.pt`, training summary/config/history, its matching `train_normalization.npz`, and a completed 20-condition validation evaluation with predictions, metrics, uncertainty intervals and plots. Evaluation checkpoint SHA256 values match the checkpoint bytes in their respective archives. Their training provenance matches except for recorded Git working-tree status; model, seed, settings, source and data fingerprints agree. Both are marked `draft_baseline`, `formal_results_eligible: false`, and selected epoch 30.

The clean evaluation was regenerated locally on CPU from the saved Kaggle checkpoint: `evaluation-val-20261109T003827245882Z-204a7b39e2e34cf99118b1fedf2f8294`. Its clean macro-F1/accuracy are 0.766491/0.793984. The augmented archive includes Kaggle evaluation `evaluation-val-20261109T011005246916Z-45a72642a2a84a2fa15572d9a13be58e`; clean validation macro-F1/accuracy are 0.740009/0.765607. Selected rows of both archived severity CSVs match the previously transcribed values below: at noise 0.10, clean/augmented macro-F1 is 0.2909/0.6824; at blur 2, 0.5890/0.6906; at resolution 4, 0.5957/0.6813. The full CSVs are the authoritative machine-readable results for these new runs. These reruns reproduce reported seed-0 numbers but are not additional independent seeds. No frozen-test evaluation has occurred; augmented seeds 1/2 and verified clean seeds 1/2 remain unavailable.

## Status and evidence

The following section preserves the 2026-11-06 research checkpoint as historical context. Its tables were transcribed from conversation output at that time. The 2026-11-09 saved artifacts above now reproduce the selected seed-0 values, while the original historical artifacts remain unrecovered. These are validation diagnostics, not frozen-test results. The protocol remains draft. Augmentation seeds 1 and 2 were intentionally skipped, so the original three-seed H3 criterion cannot be assessed.

The clean seed-0 model was retrained after its original checkpoint could not be found: `model-a-seed0-20261105T235514921339Z-16208f0551e14f51a16d13de639b0d2d`, selected epoch 30, clean validation macro-F1 0.7665 and accuracy 0.7940. Its evaluation directory was `evaluation-val-20261106T003848579855Z-a73ebafcf9c844fc8be92b0873baf001`.

The augmented model was `model-a-aug-seed0-20261105T224043534656Z-c01504439ecf4d819c307b68b542a34e`, selected epoch 30, clean validation macro-F1 0.7400 and accuracy 0.7656. Its evaluation directory was `evaluation-val-20261105T231017214095Z-f1f3e1b04a634c9696ebcb01d56a8423`. Both runs identified themselves as `draft_baseline`.

The later Kaggle session contained no `best.pt` or `severity_metrics.csv` under `/kaggle/working` or `/kaggle/input`. The only saved notebook version was a Save & Run All that failed after 32 seconds; its displayed output contained no checkpoints or results folders. The trained weights, evaluation files and configurations were not recovered. Hence matching dataset, split, normalization, source and training provenance is not independently confirmed. Identical reported seed and evaluation levels alone do not establish comparability. Historical training summaries appear in `reported_training_results.md`; the retrained clean seed 0 must not be counted as an additional independent seed.

## Reported validation metrics

F1 is fixed-ten-class macro-F1; accuracy and calibration metrics are fractions. Rows show clean-trained / augmented values. Identity rows repeat the clean result and are omitted here.

| Degradation | Level | Macro-F1 | Accuracy | ECE (15 bins) | Brier | NLL |
|---|---:|---|---|---|---|---|
| Clean | — | .766491 / .740009 | .793984 / .765607 | .022888 / .030006 | .299754 / .333316 | .629313 / .710870 |
| Noise | .02 | .759582 / .736825 | .787174 / .761067 | .023632 / .029268 | .304344 / .334062 | .640564 / .714233 |
| Noise | .05 | .722212 / .725926 | .759364 / .751419 | .032529 / .028550 | .352609 / .350101 | .747838 / .753573 |
| Noise | .10 | .290859 / .682368 | .381385 / .706583 | .221917 / .039736 | .812678 / .407030 | 1.900745 / .869091 |
| Noise | .15 | .033683 / .616997 | .122020 / .639614 | .673425 / .055935 | 1.406370 / .499192 | 4.031159 / 1.067638 |
| Blur | .5 | .732943 / .732703 | .770148 / .760499 | .022788 / .034663 | .330227 / .334561 | .685160 / .714975 |
| Blur | 1 | .719716 / .730824 | .762202 / .761067 | .030392 / .036907 | .339986 / .340591 | .707898 / .736442 |
| Blur | 2 | .588987 / .690610 | .639614 / .721907 | .020636 / .066702 | .476121 / .396340 | 1.004836 / .851255 |
| Blur | 3 | .399626 / .538453 | .455165 / .565834 | .088236 / .025866 | .694013 / .564673 | 1.560184 / 1.218634 |
| Resolution | 2 | .733871 / .731141 | .763905 / .761067 | .036337 / .035969 | .334110 / .340818 | .696842 / .737886 |
| Resolution | 4 | .595708 / .681350 | .644722 / .710556 | .046016 / .075142 | .479006 / .421400 | 1.020756 / .904894 |
| Resolution | 8 | .204557 / .295636 | .229852 / .308740 | .146385 / .094547 | .876610 / .822863 | 2.316184 / 2.082194 |
| Contrast | .75 | .714817 / .708953 | .749149 / .740636 | .082790 / .105438 | .361214 / .382630 | .754280 / .811591 |
| Contrast | .50 | .383576 / .374991 | .468218 / .437003 | .039802 / .076217 | .671590 / .731779 | 1.514227 / 1.694977 |
| Contrast | .25 | .119123 / .093264 | .216799 / .164586 | .060175 / .211311 | .885499 / .987135 | 2.201662 / 3.014335 |
| Contrast | .10 | .043143 / .060051 | .124291 / .126561 | .125117 / .241834 | .909187 / 1.028528 | 2.318465 / 3.704864 |

Noise and blur were augmentation types; noise .15 and blur 3 exceeded their training ranges. Resolution and contrast were unseen types. All corruptions are synthetic display-space perturbations, not physical observing forecasts.

## Reported per-class F1

Vectors are ordered by class 0–9: disturbed, merging, round smooth, in-between round smooth, cigar-shaped smooth, barred spiral, unbarred tight spiral, unbarred loose spiral, edge-on without bulge, edge-on with bulge. Values are transcribed to six decimal places, clean-trained first and augmented second.

| Condition | Clean-trained vector | Augmented vector |
|---|---|---|
| Clean | .385093, .841310, .906667, .868421, .750000, .839329, .706587, .640777, .855219, .871508 | .381503, .773723, .888469, .852679, .732394, .795604, .670927, .602105, .859107, .843575 |
| Noise .02 | .379747, .825000, .891386, .866667, .750000, .828235, .710843, .634921, .843854, .865169 | .386740, .766265, .890977, .845133, .739726, .803571, .646865, .587738, .856164, .845070 |
| Noise .05 | .281481, .766265, .868941, .840855, .701754, .798186, .678899, .606426, .830986, .848329 | .380435, .756501, .878957, .836601, .720000, .787037, .612795, .588983, .843206, .854749 |
| Noise .10 | .018868, .494297, .502646, .561680, .262295, .477833, .021739, .044444, .130719, .394068 | .342857, .668058, .835539, .777778, .676056, .726852, .594406, .558242, .808664, .835227 |
| Noise .15 | .000000, .000000, .123404, .000000, .000000, .000000, .000000, .000000, .000000, .213425 | .301370, .564797, .735075, .717435, .637681, .660714, .519298, .454545, .770428, .808625 |
| Blur .5 | .303448, .831099, .889328, .826722, .648649, .834951, .685535, .648370, .855263, .806061 | .363636, .758105, .892655, .849558, .720000, .784314, .655629, .610442, .854167, .838527 |
| Blur 1 | .253731, .827957, .916350, .864989, .640000, .782609, .671171, .565489, .830671, .844193 | .291391, .758950, .893939, .853333, .771429, .784141, .639731, .628906, .846416, .840000 |
| Blur 2 | .260870, .744745, .803954, .771499, .300000, .619718, .559184, .318408, .762195, .749296 | .367347, .736573, .870370, .781316, .582278, .723192, .638298, .608696, .834951, .763077 |
| Blur 3 | .196203, .501992, .566038, .522613, .153846, .252874, .408978, .094595, .664835, .634286 | .303207, .603390, .743925, .651163, .333333, .433048, .565333, .372642, .777778, .600707 |
| Resolution 2 | .253731, .825137, .910816, .847007, .779661, .781095, .672986, .594488, .841424, .832370 | .318471, .760976, .897727, .843267, .739726, .781250, .646667, .627680, .853242, .842407 |
| Resolution 4 | .307692, .736196, .814815, .794258, .341463, .614973, .565401, .283379, .732143, .766756 | .379147, .739726, .867647, .764228, .571429, .692308, .607029, .588983, .850649, .752351 |
| Resolution 8 | .175627, .107843, .279221, .019608, .000000, .150235, .339943, .094697, .589235, .289157 | .216696, .298387, .338983, .196721, .000000, .217207, .495495, .207407, .687943, .297521 |
| Contrast .75 | .411483, .696864, .902998, .885906, .603774, .759494, .673219, .590517, .763636, .860274 | .431718, .675958, .871609, .832272, .575758, .773723, .674352, .581741, .807818, .864583 |
| Contrast .50 | .207972, .021505, .703755, .656151, .000000, .456747, .483384, .205298, .440367, .660584 | .203762, .000000, .666667, .645833, .232558, .302158, .385246, .175084, .479263, .659341 |
| Contrast .25 | .134851, .000000, .585859, .000000, .000000, .000000, .010929, .015152, .000000, .444444 | .133956, .000000, .226300, .094118, .000000, .105882, .000000, .000000, .027211, .345178 |
| Contrast .10 | .074310, .000000, .304274, .000000, .000000, .000000, .000000, .022388, .000000, .030457 | .000000, .000000, .285156, .088623, .000000, .008403, .000000, .000000, .000000, .218324 |

## Interpretation and next steps

The reported augmented model performs substantially better under stronger noise and blur, and at resolution factors 4 and 8; contrast results are mixed. Clean macro-F1 is lower by .026482, exceeding the protocol's .01 clean-retention limit in this single comparison. This is not a formal rejection or confirmation of H3, whose full seed-variance and uncertainty requirements cannot be assessed. ECE is not monotonic; at augmented blur 3 it is lower than clean despite worse F1 and Brier. Per-class output was shown in the conversation, but uncertainty intervals and full prediction files are unavailable; avoid claims of significant class differences.

The user chose to proceed with this exploratory study without further Kaggle training. Subsequent work can develop the results and discussion using these numbers, with explicit limitations: one augmented seed, validation only, unverified artifact provenance, unavailable checkpoint/prediction files, and no frozen test evaluation. Formal confirmation would require recoverable or recreated artifacts, a reviewed protocol and an appropriately designed evaluation.
