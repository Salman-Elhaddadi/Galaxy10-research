# Literature and image-production evidence

Reviewed 2026-11-05 for protocol checks P1 and P6. This is a scoped evidence review for the preregistered study, not a claim of an exhaustive systematic review.

## P1: what the Galaxy10 DECaLS pixels represent

The official Galaxy10 repository identifies the released array as 17,736 256x256 colour images made from the Legacy Surveys `g`, `r`, and `z` bands, with source imaging from DECaLS, BASS, and MzLS and morphology labels from Galaxy Zoo. The repository supplies the released HDF5 SHA256 and links its construction code.

The official downloader code is more specific: it requests Legacy Survey DR8 `jpeg-cutout` products with `layer=dr8&bands=grz`, initially at 512x512 pixels and 0.262 arcsec/pixel for the DECaLS campaign-AB path. Thus the compiled Galaxy10 image tensor contains rendered, quantized colour JPEG products rather than calibrated linear photon counts or per-band inverse-variance maps. The Legacy Survey file documentation separately describes calibrated coadds as nanomaggies per pixel and its JPEG product as a calibrated `g,r,z` colour image, confirming that these are different product types.

The dataset construction repository does not freeze the viewer's precise colour-transfer/stretch parameters or expose gain, sky, read-noise, exposure, PSF, or inverse-variance data for each released RGB pixel. Consequently:

- all planned corruptions remain explicitly **display-space perturbations**;
- Gaussian noise is an operational stress test in normalized RGB space, not a detector-noise simulation;
- Poisson noise is not enabled in protocol v1 because applying it to the rendered 8-bit RGB values would not constitute a justified photon-count model; and
- conclusions must not be phrased as observing-time, flux, or signal-to-noise forecasts.

Primary sources:

- [Official Galaxy10 dataset repository and construction notes](https://github.com/henrysky/Galaxy10)
- [Official Galaxy10 DR8 downloader source](https://github.com/henrysky/Galaxy10/blob/master/downloader_gzdes_ab_des_dr8.py)
- [Galaxy10 DECaLS Zenodo record](https://zenodo.org/records/10845026)
- [Dey et al. (2019), Overview of the DESI Legacy Imaging Surveys](https://arxiv.org/abs/1804.08657)
- [Legacy Survey DR8 file descriptions](https://www.legacysurvey.org/dr8/files/)

## P6: prior work and positioning

### Galaxy morphology under perturbation

Pandya et al. (2023) directly evaluate ordinary and E(2)-equivariant CNNs on Galaxy10 DECaLS under synthetic Poisson noise and one-pixel adversarial attacks. They report that equivariant models are more robust than their matched conventional CNN. This makes a generic claim that noise harms Galaxy10 classification expected rather than novel. Their perturbation definition and architecture differ from this study's display-space Gaussian noise, blur, resolution, and contrast suite.

Cao et al. (2024) evaluate a convolutional vision transformer for galaxy morphology with multiple severities of Gaussian and salt-and-pepper noise. Reported accuracy declines under stronger noise. This further establishes galaxy-classifier noise robustness as prior art, though it does not provide this protocol's fixed-ten-class macro-F1, calibration, paired bootstrap, or per-class degradation analysis.

Ćiprijanović et al. (2022) study simulated galaxy morphology under higher observational Poisson noise and one-pixel attacks, and test domain adaptation as mitigation. The data are IllustrisTNG-based simulated observations rather than Galaxy10 rendered cutouts, so the physical noise interpretation does not transfer to the present RGB dataset. It nevertheless supports evaluating distribution shift and robustness interventions.

Primary papers:

- [Pandya et al. (2023), E(2) Equivariant Neural Networks for Robust Galaxy Morphology Classification](https://arxiv.org/abs/2311.01500)
- [Cao et al. (2024), galaxy morphology classification with a convolutional vision transformer, A&A 683 A42](https://doi.org/10.1051/0004-6361/202348544)
- [Ćiprijanović et al. (2022), DeepAdversaries](https://arxiv.org/abs/2112.14299)

### Common corruptions and calibration under shift

Hendrycks and Dietterich (2019) establish severity-indexed common-corruption benchmarks and motivate evaluating a fixed trained classifier across corruption types and levels. The present study follows that broad stress-test pattern but reports unnormalized task metrics rather than ImageNet-C's AlexNet-normalized corruption error.

Guo et al. (2017) show that classification accuracy and confidence calibration are distinct and use the widely adopted 15-bin ECE evaluation alongside NLL. Ovadia et al. (2019) show that calibration and predictive uncertainty often degrade under dataset shift and that post-hoc calibration fitted in-distribution need not remain reliable under increasing shift. These findings justify reporting ECE, NLL, Brier score, confidence, and reliability diagrams at each severity instead of accuracy alone. Protocol v1 deliberately measures native calibration and does not add post-hoc recalibration.

Primary papers:

- [Hendrycks and Dietterich (2019), Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261)
- [Guo et al. (2017), On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Ovadia et al. (2019), Can You Trust Your Model's Uncertainty?](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

### Grad-CAM interpretation

Adebayo et al. (2018) demonstrate that visual appeal alone is insufficient to validate saliency methods and propose model-parameter and data-randomization sanity checks. Grad-CAM passed their tested model-parameter sensitivity checks, whereas Guided Grad-CAM did not. This supports the protocol's fully randomized-weight comparison, but it does not make a Grad-CAM map a causal or astrophysical explanation. The study therefore treats Grad-CAM as qualitative model-behaviour evidence only.

- [Adebayo et al. (2018), Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)

## Defensible contribution boundary

The evidence does **not** support novelty for applying noise to galaxy images or for observing performance degradation. The defensible study contribution, subject to final results, is the combined and reproducible evaluation of a lightweight Galaxy10 classifier across four display-space degradation families, with fixed severity grids, per-class and pixel-scale analysis, native calibration, paired image-bootstrap uncertainty, seed variance, a controlled augmentation comparison, and randomized-weight Grad-CAM sanity views. Any final novelty statement must remain conditional and narrower than the implemented feature list.

## Review decision

P1 is closed with the explicit limitation that exact viewer stretch parameters are not preserved and no physical Poisson model is justified. P6 is closed for protocol-v1 scope: the directly relevant Galaxy10, galaxy-robustness, common-corruption, calibration-under-shift, and saliency-sanity literature above has been read and positioned. The protocol remains draft until the GPU preflight/results are reviewed and the deliberate freeze/test-access decision is made.
