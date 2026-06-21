"""
KZN Flood Detection — Results Dashboard
Loads trained U-Net model weights and displays evaluation results.
No training happens here — this is a results viewer for already-trained models.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# =====================================================================
# Page setup
# =====================================================================
st.set_page_config(
    page_title="KZN Flood Detection Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 KwaZulu-Natal Flood Detection — Results Dashboard")
st.markdown(
    """
    **Multi-source deep learning flood detection for the April 2022 KwaZulu-Natal flood event.**
    This dashboard loads the trained U-Net model weights and displays evaluation results.
    For the full data acquisition, preprocessing, and training pipeline, see the notebook links below.
    """
)

GITHUB_REPO = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME"  # <-- update this

st.markdown(f"📂 [View full project on GitHub]({GITHUB_REPO})")

st.divider()

# =====================================================================
# Model architecture (must match training exactly)
# =====================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out_conv = nn.Conv2d(base, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.up4(b);  d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


# =====================================================================
# Load models (cached so this only happens once per session)
# =====================================================================
@st.cache_resource
def load_models():
    device = torch.device('cpu')  # Streamlit Cloud free tier has no GPU

    fusion_path = "models/fusion_unet_4tile_best.pt"
    optical_path = "models/optical_unet_4tile_best.pt"

    status = {"fusion": False, "optical": False}

    fusion_model = None
    optical_model = None

    if os.path.exists(fusion_path):
        fusion_model = UNet(in_channels=12)
        fusion_model.load_state_dict(torch.load(fusion_path, map_location=device))
        fusion_model.eval()
        status["fusion"] = True

    if os.path.exists(optical_path):
        optical_model = UNet(in_channels=9)
        optical_model.load_state_dict(torch.load(optical_path, map_location=device))
        optical_model.eval()
        status["optical"] = True

    return fusion_model, optical_model, status


with st.spinner("Loading trained model weights..."):
    fusion_model, optical_model, model_status = load_models()

col1, col2 = st.columns(2)
with col1:
    if model_status["fusion"]:
        n_params = sum(p.numel() for p in fusion_model.parameters())
        st.success(f"✅ Fusion model (12-channel) loaded — {n_params:,} parameters")
    else:
        st.error("❌ Fusion model file not found at models/fusion_unet_4tile_best.pt")

with col2:
    if model_status["optical"]:
        n_params = sum(p.numel() for p in optical_model.parameters())
        st.success(f"✅ Optical model (9-channel) loaded — {n_params:,} parameters")
    else:
        st.error("❌ Optical model file not found at models/optical_unet_4tile_best.pt")

st.divider()

# =====================================================================
# Results — Quantitative metrics
# =====================================================================
st.header("📊 Test Set Performance")

st.markdown(
    "Evaluated on a held-out test set of **2,897 patches**, against the "
    "UNOSAT FL20220418ZAF reference for the April 2022 KwaZulu-Natal flood event."
)

metrics = {
    "Metric": ["F1-Score", "Cohen's Kappa", "AUC-ROC"],
    "Fusion (12-channel)": [0.8443, 0.8378, 0.9900],
    "Optical-only (9-channel)": [0.8210, 0.8134, 0.9852],
}

import pandas as pd
df = pd.DataFrame(metrics)
st.dataframe(df, use_container_width=True, hide_index=True)

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Fusion F1-Score", "0.8443", delta="+0.0233 vs optical")
mcol2.metric("Fusion Cohen's Kappa", "0.8378", delta="+0.0244 vs optical")
mcol3.metric("Fusion AUC-ROC", "0.9900", delta="+0.0048 vs optical")

st.caption(
    "The fused (12-channel) configuration outperformed the optical-only (9-channel) "
    "configuration on every metric, indicating that the Sentinel-1 SAR and SRTM elevation "
    "channels provided a measurable benefit for this event."
)

st.divider()

# =====================================================================
# Results — Figures (optional, only shows if files are present)
# =====================================================================
st.header("🗺️ Flood Zone Analysis")

st.markdown(
    "Detailed spatial comparison at two representative locations: the Durban Harbour "
    "area (Zone 1) and the uMngeni River corridor (Zone 2)."
)

figure_dir = "figures/"
figure_files = {
    "Zone 1 — UNOSAT Ground Truth": "flood_area_9_03_ground_truth.png",
    "Zone 1 — Fusion Prediction": "flood_area_9_04_fusion.png",
    "Zone 1 — Error Analysis": "flood_area_9_06_error_analysis.png",
    "Zone 2 — UNOSAT Ground Truth": "flood_area_2_03_ground_truth.png",
    "Zone 2 — Fusion Prediction": "flood_area_2_04_fusion.png",
    "Zone 2 — Error Analysis": "flood_area_2_06_error_analysis.png",
}

zone_choice = st.selectbox("Select a zone to view:", ["Zone 1 (Durban Harbour)", "Zone 2 (uMngeni River)"])

if zone_choice == "Zone 1 (Durban Harbour)":
    keys = ["Zone 1 — UNOSAT Ground Truth", "Zone 1 — Fusion Prediction", "Zone 1 — Error Analysis"]
else:
    keys = ["Zone 2 — UNOSAT Ground Truth", "Zone 2 — Fusion Prediction", "Zone 2 — Error Analysis"]

fcols = st.columns(3)
for i, key in enumerate(keys):
    fpath = os.path.join(figure_dir, figure_files[key])
    with fcols[i]:
        st.markdown(f"**{key.split('— ')[-1]}**")
        if os.path.exists(fpath):
            st.image(fpath, use_container_width=True)
        else:
            st.info(f"Figure not found: {figure_files[key]}\n\n(Upload to `{figure_dir}` to display)")

st.divider()

# =====================================================================
# Links to full pipeline notebooks
# =====================================================================
st.header("📓 Full Pipeline — Run the Notebooks Yourself")

st.markdown(
    """
    This dashboard shows results from already-trained models. To see or reproduce the
    full data acquisition, preprocessing, and training pipeline, open the notebooks
    directly on GitHub or launch them live via Binder (no installation required).

    **Note:** training notebooks require a GPU and will run very slowly on Binder's
    free CPU-only environment. They are best run in Google Colab with a GPU runtime.
    """
)

notebooks = [
    ("B1 — Study Area Definition", "B1_study_area_definition_final.ipynb"),
    ("B2 — Data Acquisition", "B2_acquisition_final.ipynb"),
    ("B3 — Mosaic Assembly", "B3_mosaic_final.ipynb"),
    ("B4 — Stack Assembly", "B4_stack_assembly_final.ipynb"),
    ("B5 — Patch Extraction", "B5_patch_extraction_final.ipynb"),
    ("B6 — Model Training (requires GPU)", "B6_model_trainingfinal.ipynb"),
    ("B7 — Results Figures", "B7_final_chapter4_figures_final.ipynb"),
]

BINDER_BASE = f"https://mybinder.org/v2/gh/YOUR_USERNAME/YOUR_REPO_NAME/HEAD?filepath="  # <-- update this

for name, filename in notebooks:
    gh_link = f"{GITHUB_REPO}/blob/main/notebooks/{filename}"
    binder_link = f"{BINDER_BASE}notebooks/{filename}"
    st.markdown(f"**{name}**  \n[View on GitHub]({gh_link}) · [Launch on Binder]({binder_link})")

st.divider()
st.caption(
    "Athindothe Valencia Marubini — MEng Satellite Systems and Applications, "
    "Cape Peninsula University of Technology. Supervisor: Prof. Innocent Davidson, "
    "Co-supervisor: Dr. OP Babalola."
)
