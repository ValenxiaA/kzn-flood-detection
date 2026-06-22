# Flood & Drought Detection Using Deep Learning

**MEng Thesis — Cape Peninsula University of Technology (CPUT) / F'SATI**  
Athindothe Valencia Marubini · Student no. 219160643  
Supervisor: Prof. I. Davidson · Co-supervisor: Dr. O.P. Babalola

---

## Overview

This project develops and evaluates a deep learning pipeline for flood and drought
detection over South Africa using satellite remote sensing data.

**Flood detection** — A 12-channel fusion U-Net trained on Sentinel-1 SAR and
Sentinel-2 optical imagery detects flood extent from the April 2022 KwaZulu-Natal
flood event (tiles 36JTM, 36JUM, 36JTN, 36JUN). An optical-only 9-channel
configuration serves as a sensor-contribution baseline.

**Drought prediction** — A two-layer LSTM predicts the Standardised Precipitation
Index at 3-month scale (SPI-3) one month ahead using CHIRPS rainfall and MODIS NDVI
anomaly, trained separately for KwaZulu-Natal and the Northern Cape over 2000–2024.

---

## Live Dashboard

[Launch Dashboard](https://fzmgxujyvc9vzfbinq9r4i.streamlit.app)

---

## Results

### Flood Detection (test set, 2 897 patches)

| Model | F1 | IoU | Kappa | AUC-ROC |
|---|---|---|---|---|
| Fusion U-Net (12-ch) | 0.8344 | 0.7159 | 0.8258 | 0.9804 |
| Optical U-Net (9-ch) | 0.8908 | 0.8032 | 0.8856 | 0.9852 |

### Drought Prediction (LSTM, held-out test set)

| Province | RMSE | R² | Pearson r |
|---|---|---|---|
| KwaZulu-Natal | 0.800 | 0.132 | 0.525 |
| Northern Cape | 0.829| 0.389 | 0.750 |



---

## Repository Structure

```
kzn-flood-detection/
├── app.py
├── requirements.txt
├── index.html
├── models/
│   ├── fusion_unet_4tile_best.pt
│   ├── optical_unet_4tile_best.pt
│   ├── lstm_KZN.pth
│   └── lstm_NC.pth
├── data/
│   └── drought/
│       ├── spi3_KZN.csv
│       ├── spi3_NC.csv
│       ├── rainfall_KZN.csv
│       ├── rainfall_NC.csv
│       ├── ndvi_KZN.csv
│       ├── ndvi_NC.csv
│       ├── ndvi_anomaly_KZN.csv
│       └── ndvi_anomaly_NC.csv
├── figures/
├── sample_patches/
└── notebooks/
    ├── B1_study_area_definition_final.ipynb
    ├── B2_acquisition_final.ipynb
    ├── B3_mosaic_final.ipynb
    ├── B4_stack_assembly_final.ipynb
    ├── B5_patch_extraction_final.ipynb
    ├── B6_model_trainingfinal.ipynb
    ├── B7_final_chapter4_figures_final.ipynb
    └── KZN_NC_Drought_Thesis_Final.ipynb
```

## Notebooks

You do not need to run the full pipeline to explore results. Start at **B7** to load
trained weights and reproduce all metrics and figures without retraining.

| Notebook | Purpose | GPU needed |
|---|---|---|
| B1 | Study area definition from UNOSAT geometry | No |
| B2 | Sentinel-1 + Sentinel-2 GEE export | No |
| B3 | Mosaic and reprojection (EPSG:32736) | No |
| B4 | 12-channel raster stack assembly | No |
| B5 | 128×128 patch extraction | No |
| B6 | U-Net training (fusion + optical configs) | **Yes** |
| B7 | Results figures and evaluation — start here | No |
| Drought | Full LSTM pipeline (CHIRPS + MODIS + SPI-3) | No |

---

## Data Sources

| Dataset | Source | Use |
|---|---|---|
| Sentinel-1 SAR | ESA via Google Earth Engine | Flood detection input |
| Sentinel-2 MSI | ESA via Google Earth Engine | Flood detection input |
| SRTM DEM | NASA via Google Earth Engine | Elevation channel |
| UNOSAT FL20220418ZAF | UNOSAT | Flood labels |
| CHIRPS v2.0 | Climate Hazards Group | Drought rainfall input |
| MODIS MOD13A3 | NASA | Drought NDVI input |

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires the model weights in `models/` and CSV data in `data/drought/`.
