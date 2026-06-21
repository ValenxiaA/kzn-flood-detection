# Deep Learning for Flood Detection in KwaZulu-Natal, South Africa

**Athindothe Valencia Marubini**
MEng Satellite Systems and Applications, Cape Peninsula University of Technology
Supervisor: Prof. Innocent Davidson | Co-supervisor: Dr. OP Babalola

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ValenxiaA/kzn-flood-detection/HEAD)

---

## Overview

This project develops and evaluates a multi-source deep learning pipeline for
automated flood inundation mapping, applied to the April 2022 KwaZulu-Natal
(KZN) flood event in South Africa.

A U-Net convolutional neural network was trained on a twelve-channel feature
stack combining Sentinel-2 optical bands and spectral indices (NDVI, NDWI,
MNDWI), Sentinel-1 SAR VV backscatter (pre- and post-flood), and SRTM
elevation. The study area, comprising four Sentinel-2 MGRS tiles (T36JTM,
T36JUM, T36JTN, T36JUN), was derived directly from the spatial extent of the
UNOSAT FL20220418ZAF emergency flood mapping product, rather than assumed in
advance.

A controlled optical-only configuration (nine channels) was trained under
identical conditions to isolate the contribution of the SAR and elevation
channels to detection performance.

**[Live results dashboard →](https://YOUR-APP-NAME.streamlit.app)**

---

## What This Pipeline Demonstrates

1. **Study area derivation** — verifying MGRS tile requirements directly
   against UNOSAT ground-truth flood polygon coverage, rather than assuming
   tile boundaries in advance
2. **Multi-source data acquisition** — Sentinel-1 SAR and Sentinel-2 optical
   imagery sourced via Google Earth Engine and Copernicus, with documented
   cloud-cover and SAR swath-coverage verification
3. **Mosaic assembly** — reprojection and mosaicking of four tiles into a
   single aligned coordinate grid, including correction of a Sentinel-1 UTM
   zone mismatch introduced during export
4. **Feature stack construction** — twelve-channel stack combining optical
   reflectance, spectral indices, SAR backscatter, and normalised elevation
5. **Patch extraction** — 128×128 patch generation with flood-region dense
   sampling, nodata-aware non-flood filtering, and 8x geometric augmentation
6. **U-Net training** — fused (12-channel) and optical-only (9-channel)
   configurations trained under identical conditions for a controlled sensor
   contribution comparison
7. **Spatial zone analysis** — detailed flood-zone evaluation at two
   representative locations (Durban Harbour, uMngeni River corridor),
   comparing model predictions directly against UNOSAT ground truth

---

## Results Summary

| Metric | Fusion (12-channel) | Optical-only (9-channel) |
|---|---|---|
| F1-Score | 0.8443 | 0.8210 |
| Cohen's Kappa | 0.8378 | 0.8134 |
| AUC-ROC | 0.9900 | 0.9852 |

Evaluated on a held-out test set of 2,897 patches against the UNOSAT
FL20220418ZAF reference. The fused configuration outperformed the
optical-only configuration on every metric, indicating that the Sentinel-1
SAR and SRTM elevation channels provided a measurable benefit for this
event.

---

## Installation

```bash
git clone https://github.com/ValenxiaA/kzn-flood-detection.git
cd kzn-flood-detection
pip install -r requirements.txt
streamlit run app.py
```

Or view the live results dashboard with no installation required:

**[https://YOUR-APP-NAME.streamlit.app](https://YOUR-APP-NAME.streamlit.app)**

To run the full data pipeline notebooks (study area definition, data
acquisition, mosaicking, stack assembly, patch extraction, training, and
figure generation), open them directly from the `notebooks/` folder in
Google Colab (recommended, GPU available) or launch via Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ValenxiaA/kzn-flood-detection/HEAD)

**Note:** the model training notebook (B6) requires a GPU and will run very
slowly on Binder's free CPU-only environment. It is best run in Google
Colab with a GPU runtime.

---

## Repository Structure

```
kzn-flood-detection/
├── app.py                                  # Streamlit results dashboard
├── requirements.txt                        # Python dependencies
├── models/
│   ├── fusion_unet_4tile_best.pt           # Trained fusion model (12-channel)
│   └── optical_unet_4tile_best.pt          # Trained optical-only model (9-channel)
├── figures/                                 # Result figures (flood zone analysis)
├── notebooks/
│   ├── B1_study_area_definition_final.ipynb
│   ├── B2_acquisition_final.ipynb
│   ├── B3_mosaic_final.ipynb
│   ├── B4_stack_assembly_final.ipynb
│   ├── B5_patch_extraction_final.ipynb
│   ├── B6_model_trainingfinal.ipynb
│   └── B7_final_chapter4_figures_final.ipynb
└── README.md
```

---

## Data Sources

| Dataset | Source | Resolution |
|---|---|---|
| Sentinel-1 GRD | ESA Copernicus (via Google Earth Engine) | 10 m |
| Sentinel-2 L2A | ESA Copernicus | 10–20 m |
| SRTM DEM | NASA (via Google Earth Engine) | 30 m |
| UNOSAT flood extent | UNOSAT Humanitarian Data Exchange (FL20220418ZAF) | — |

---

## Citation

This repository accompanies the MEng thesis:

> Marubini, A.V. (2026). *Development and Evaluation of a Deep Learning
> Model for Flood and Drought Detection Using Satellite Remote Sensing Data
> in South Africa.* MEng thesis, Cape Peninsula University of Technology.

---

## Contact

**Athindothe Valencia Marubini**
Cape Peninsula University of Technology
