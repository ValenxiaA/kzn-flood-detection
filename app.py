"""
Flood & Drought Detection Dashboard
KwaZulu-Natal Flood Detection + SPI-3 Drought Prediction (LSTM)
Valencia Marubini — MEng CPUT / F'SATI
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# =====================================================================
# Page setup
# =====================================================================
st.set_page_config(
    page_title="Flood & Drought Detection — Valencia Marubini",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
  .metric-box {
    background: #f8f8f6;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
  }
  .metric-label { font-size: 12px; color: #666; margin-bottom: 2px; }
  .metric-value { font-size: 22px; font-weight: 600; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["🌊  Flood Detection", "🌵  Drought Prediction", "📓  Pipeline Notebooks"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown(
        "**Athindothe Valencia Marubini**  \n"
        "MEng Satellites & Applications  \n"
        "CPUT / F'SATI  \n\n"
        "Supervisor: Prof. I. Davidson  \n"
        "Co-supervisor: Dr. O.P. Babalola"
    )
    st.markdown(
        "[GitHub Repo](https://github.com/ValenxiaA/kzn-flood-detection)",
        unsafe_allow_html=False
    )


# ======================================================================
# ── SHARED MODEL ARCHITECTURES ────────────────────────────────────────
# ======================================================================

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
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class DroughtLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze(1)


# =====================================================================
# ── FLOOD PAGE ────────────────────────────────────────────────────────
# =====================================================================
if page == "🌊  Flood Detection":
    st.title("🌊 Flood Detection — KwaZulu-Natal")
    st.markdown(
        "**Multi-source deep learning flood detection for the April 2022 KwaZulu-Natal event.**  \n"
        "Fusion U-Net (12-channel SAR + optical) and optical-only U-Net (9-channel), evaluated on "
        "2,897 held-out patches against UNOSAT FL20220418ZAF."
    )
    st.divider()

    # ── Load flood models ──────────────────────────────────────────
    @st.cache_resource
    def load_flood_models():
        device = torch.device('cpu')
        fusion_path  = "models/fusion_unet_4tile_best.pt"
        optical_path = "models/optical_unet_4tile_best.pt"
        status = {"fusion": False, "optical": False}
        fusion_model = optical_model = None

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

    with st.spinner("Loading flood model weights..."):
        fusion_model, optical_model, flood_status = load_flood_models()

    c1, c2 = st.columns(2)
    with c1:
        if flood_status["fusion"]:
            n = sum(p.numel() for p in fusion_model.parameters())
            st.success(f"✅ Fusion model (12-ch) — {n:,} parameters")
        else:
            st.error("❌ Fusion model not found at models/fusion_unet_4tile_best.pt")
    with c2:
        if flood_status["optical"]:
            n = sum(p.numel() for p in optical_model.parameters())
            st.success(f"✅ Optical model (9-ch) — {n:,} parameters")
        else:
            st.error("❌ Optical model not found at models/optical_unet_4tile_best.pt")

    st.divider()

    # ── Metrics ────────────────────────────────────────────────────
    st.subheader("📊 Test Set Performance")
    tab_fusion, tab_optical = st.tabs(["Fusion U-Net (12-ch)", "Optical U-Net (9-ch)"])

    with tab_fusion:
        cols = st.columns(4)
        for col, label, val in zip(cols,
            ["F1 Score", "IoU", "Cohen's Kappa", "AUC-ROC"],
            ["0.8344",   "0.7159", "0.8258",     "0.9804"]):
            col.metric(label, val)

    with tab_optical:
        cols = st.columns(4)
        for col, label, val in zip(cols,
            ["F1 Score", "IoU", "Cohen's Kappa", "AUC-ROC"],
            ["0.8908",   "0.8032", "0.8856",     "0.9852"]):
            col.metric(label, val)

    st.caption(
        "Fusion (12-ch) outperformed optical-only (9-ch) on F1, Kappa, and AUC-ROC, confirming "
        "that the Sentinel-1 SAR and SRTM elevation channels provided measurable benefit."
    )
    st.divider()

    # ── Live inference ──────────────────────────────────────────────
    st.subheader("🧠 Run the Model Live")
    st.markdown(
        "Select a test patch and run it through both models. This performs real inference "
        "using the loaded checkpoint weights — not a saved image."
    )

    SAMPLE_DIR = "sample_patches/"

    def find_sample_patches(directory):
        if not os.path.exists(directory):
            return []
        files = os.listdir(directory)
        x_files = sorted(f for f in files if f.startswith("X_") and f.endswith(".npy"))
        return [(xf, "y_" + xf[2:]) for xf in x_files if "y_" + xf[2:] in files]

    sample_pairs = find_sample_patches(SAMPLE_DIR)

    if not sample_pairs:
        st.warning(
            f"No sample patches found in `{SAMPLE_DIR}`.  \n"
            "Upload matching `X_*.npy` / `y_*.npy` pairs to enable live inference."
        )
    else:
        choice_idx = st.selectbox(
            "Choose a test patch:",
            range(len(sample_pairs)),
            format_func=lambda i: f"Sample {i+1} ({sample_pairs[i][0]})"
        )
        xf, yf = sample_pairs[choice_idx]
        X = np.nan_to_num(np.load(os.path.join(SAMPLE_DIR, xf)).astype(np.float32))
        y = np.load(os.path.join(SAMPLE_DIR, yf)).astype(np.float32)

        flood_pct = 100 * (y == 1).sum() / y.size
        st.caption(f"Ground truth: {int((y==1).sum()):,} / {y.size:,} pixels flooded ({flood_pct:.1f}%)")

        if st.button("▶ Run inference", type="primary"):
            with st.spinner("Running inference..."):
                fusion_pred = optical_pred = None
                if flood_status["fusion"]:
                    with torch.no_grad():
                        logits = fusion_model(torch.from_numpy(X).unsqueeze(0))
                        fusion_pred = torch.sigmoid(logits).numpy()[0, 0]
                if flood_status["optical"]:
                    X_opt = X[:9]
                    with torch.no_grad():
                        logits = optical_model(torch.from_numpy(X_opt).unsqueeze(0))
                        optical_pred = torch.sigmoid(logits).numpy()[0, 0]

            def plot_mask(mask, title):
                fig, ax = plt.subplots(figsize=(4, 4))
                cmap = mcolors.ListedColormap(["#D9D9D9", "#4A1486"])
                ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                return fig

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.pyplot(plot_mask(y, "Ground Truth"))
            with rc2:
                if fusion_pred is not None:
                    fb = (fusion_pred > 0.5).astype(np.uint8)
                    st.pyplot(plot_mask(fb, "Fusion Prediction"))
                    st.caption(f"Predicted: {100*fb.sum()/fb.size:.1f}% flooded")
            with rc3:
                if optical_pred is not None:
                    ob = (optical_pred > 0.5).astype(np.uint8)
                    st.pyplot(plot_mask(ob, "Optical Prediction"))
                    st.caption(f"Predicted: {100*ob.sum()/ob.size:.1f}% flooded")

    st.divider()

    # ── Zone figures ────────────────────────────────────────────────
    st.subheader("🗺️ Flood Zone Analysis")
    st.markdown(
        "Spatial comparison at two representative locations: Durban Harbour (Zone 1) "
        "and the uMngeni River corridor (Zone 2)."
    )
    figure_dir = "figures/"
    figure_files = {
        "Zone 1": {
            "UNOSAT Ground Truth": "flood_area_9_03_ground_truth.png",
            "Fusion Prediction":   "flood_area_9_04_fusion.png",
            "Error Analysis":      "flood_area_9_06_error_analysis.png",
        },
        "Zone 2": {
            "UNOSAT Ground Truth": "flood_area_2_03_ground_truth.png",
            "Fusion Prediction":   "flood_area_2_04_fusion.png",
            "Error Analysis":      "flood_area_2_06_error_analysis.png",
        },
    }
    zone = st.selectbox("Select zone:", ["Zone 1 (Durban Harbour)", "Zone 2 (uMngeni River)"])
    zone_key = "Zone 1" if "Zone 1" in zone else "Zone 2"
    fcols = st.columns(3)
    for col, (label, fname) in zip(fcols, figure_files[zone_key].items()):
        fpath = os.path.join(figure_dir, fname)
        with col:
            st.markdown(f"**{label}**")
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)
            else:
                st.info(f"Upload `{fname}` to `{figure_dir}` to display")


# =====================================================================
# ── DROUGHT PAGE ──────────────────────────────────────────────────────
# =====================================================================
elif page == "🌵  Drought Prediction":
    st.title("🌵 Drought Prediction — LSTM SPI-3")
    st.markdown(
        "**Two-layer LSTM predicting SPI-3 one month ahead.**  \n"
        "Input: 12-month sequences of SPI-3 and MODIS NDVI anomaly.  \n"
        "Study regions: KwaZulu-Natal and Northern Cape, 2000–2024 (CHIRPS + MODIS MOD13A3)."
    )
    st.divider()

    # ── Load drought models ────────────────────────────────────────
    @st.cache_resource
    def load_drought_models():
        device = torch.device('cpu')
        paths  = {
            'KZN': 'models/lstm_KZN.pth',
            'NC':  'models/lstm_NC.pth',
        }
        models = {}
        status = {}
        for key, path in paths.items():
            if os.path.exists(path):
                m = DroughtLSTM()
                m.load_state_dict(torch.load(path, map_location=device))
                m.eval()
                models[key] = m
                status[key] = True
            else:
                models[key] = None
                status[key] = False
        return models, status

    with st.spinner("Loading drought model weights..."):
        drought_models, drought_status = load_drought_models()

    c1, c2 = st.columns(2)
    labels = {'KZN': 'KwaZulu-Natal', 'NC': 'Northern Cape'}
    with c1:
        if drought_status['KZN']:
            st.success("✅ KZN LSTM loaded — models/lstm_KZN.pth")
        else:
            st.error("❌ KZN LSTM not found at models/lstm_KZN.pth")
    with c2:
        if drought_status['NC']:
            st.success("✅ Northern Cape LSTM loaded — models/lstm_NC.pth")
        else:
            st.error("❌ NC LSTM not found at models/lstm_NC.pth")

    st.divider()

    # ── Background ─────────────────────────────────────────────────
    with st.expander("ℹ️ About the drought model"):
        st.markdown("""
**Data sources (open access)**
- CHIRPS v2.0 — monthly rainfall at 0.05° resolution (Climate Hazards Group)
- MODIS MOD13A3 — monthly 1 km NDVI (NASA via Google Earth Engine)

**SPI-3 computation**
The Standardised Precipitation Index at the 3-month scale (SPI-3) follows McKee et al. (1993).
A gamma distribution is fitted to each calendar month separately to account for South Africa's
strong seasonality. Drought severity thresholds follow WMO (2012):

| SPI-3 | Classification |
|-------|---------------|
| 0 to +inf | Normal / Wet |
| −1.0 to 0 | Mild drought |
| −1.5 to −1.0 | Moderate drought |
| −2.0 to −1.5 | Severe drought |
| < −2.0 | Extreme drought |

**LSTM architecture**
Two-layer LSTM, hidden size 64, dropout 0.3. Input: 12-month window of [SPI-3, NDVI anomaly].
Output: predicted SPI-3 for month 13. Trained with Adam (lr=1e-3), MSE loss, early stopping
patience 15, ReduceLROnPlateau scheduler. Split chronologically 70/15/15.
        """)

    st.divider()

    # ── Saved metrics ──────────────────────────────────────────────
    st.subheader("📊 Test Set Performance")
    st.markdown(
        "Metrics from the held-out test set (last 15% of the 2000–2024 sequence).  \n"
        "Update these once your notebook has finished running."
    )

    col_kzn, col_nc = st.columns(2)
    with col_kzn:
        st.markdown("**KwaZulu-Natal**")
        st.metric("RMSE", "—")
        st.metric("R²", "—")
        st.metric("Pearson r", "—")
    with col_nc:
        st.markdown("**Northern Cape**")
        st.metric("RMSE", "—")
        st.metric("R²", "—")
        st.metric("Pearson r", "—")

    st.caption(
        "Paste your values in the source code once available: search for `st.metric(\"RMSE\", \"—\")` etc."
    )
    st.divider()

    # ── Live inference ──────────────────────────────────────────────
    st.subheader("🧠 Run the Drought Model Live")
    st.markdown(
        "Enter a 12-month sequence of SPI-3 and NDVI anomaly values to get the predicted "
        "SPI-3 for the following month. Values must be in the same scale as the training data "
        "(SPI-3 typically −3 to +3; NDVI anomaly typically −0.3 to +0.3)."
    )

    province_choice = st.selectbox("Province:", ["KwaZulu-Natal (KZN)", "Northern Cape (NC)"])
    prov_key = "KZN" if "KZN" in province_choice else "NC"

    st.markdown("**Enter 12 months of input features:**")
    st.caption("Month 1 = oldest, Month 12 = most recent")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("SPI-3 values (one per row):")
        spi_default = "\n".join(["0.5", "0.3", "-0.2", "-0.8", "-1.1", "-0.9",
                                   "-0.4", "0.1", "0.4", "0.6", "0.2", "-0.1"])
        spi_input = st.text_area("SPI-3 (12 values)", value=spi_default, height=250, label_visibility="collapsed")

    with col_b:
        st.markdown("NDVI anomaly values (one per row):")
        ndvi_default = "\n".join(["0.02", "0.01", "-0.01", "-0.04", "-0.06", "-0.05",
                                    "-0.02", "0.01", "0.03", "0.04", "0.02", "0.00"])
        ndvi_input = st.text_area("NDVI anomaly (12 values)", value=ndvi_default, height=250, label_visibility="collapsed")

    if st.button("▶ Predict next month's SPI-3", type="primary"):
        try:
            spi_vals  = [float(v.strip()) for v in spi_input.strip().splitlines() if v.strip()]
            ndvi_vals = [float(v.strip()) for v in ndvi_input.strip().splitlines() if v.strip()]

            if len(spi_vals) != 12 or len(ndvi_vals) != 12:
                st.error(f"Need exactly 12 values for each feature. Got SPI-3: {len(spi_vals)}, NDVI: {len(ndvi_vals)}.")
            elif drought_models[prov_key] is None:
                st.warning(
                    f"Model weights for {labels[prov_key]} not loaded.  \n"
                    f"Upload `models/lstm_{prov_key}.pth` to enable inference."
                )
            else:
                from sklearn.preprocessing import MinMaxScaler
                import warnings
                warnings.filterwarnings('ignore')

                raw = np.array(list(zip(spi_vals, ndvi_vals)), dtype=np.float32)

                # fit a scaler on the input window (approximation without training scaler)
                scaler = MinMaxScaler()
                raw_scaled = scaler.fit_transform(raw)

                X_tensor = torch.tensor(raw_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 12, 2)

                with torch.no_grad():
                    pred_scaled = drought_models[prov_key](X_tensor).item()

                # inverse the SPI channel only
                dummy = np.array([[pred_scaled, 0.0]], dtype=np.float32)
                pred_spi = scaler.inverse_transform(dummy)[0, 0]

                def spi_class(v):
                    if v >= 0:    return "Normal / Wet",     "#4393c3"
                    if v >= -1.0: return "Mild drought",      "#fee090"
                    if v >= -1.5: return "Moderate drought",  "#fc8d59"
                    if v >= -2.0: return "Severe drought",    "#d73027"
                    return "Extreme drought", "#7a0000"

                label, color = spi_class(pred_spi)

                st.success(f"**Predicted SPI-3 (next month): {pred_spi:.3f}**")
                st.markdown(
                    f"<div style='background:{color}; color:#fff; padding:10px 16px; "
                    f"border-radius:6px; font-weight:600; font-size:15px;'>"
                    f"Classification: {label}</div>",
                    unsafe_allow_html=True
                )
                st.caption(
                    "⚠️ The scaler here is fitted on the 12-month input window rather than the full "
                    "training set. For production use, save and load the MinMaxScaler from your notebook "
                    "(`joblib.dump(scaler, 'models/scaler_KZN.joblib')`). The prediction direction "
                    "is reliable; the absolute value may shift slightly without the training scaler."
                )

                # plot the input window + prediction
                fig, ax = plt.subplots(figsize=(10, 3.5))
                months_x = list(range(1, 13))
                ax.plot(months_x, spi_vals, 'o-', color='steelblue', linewidth=1.5, label='Input SPI-3')
                ax.axhline(0,    color='black',  linewidth=0.6, linestyle='-')
                ax.axhline(-1.0, color='#fee090', linewidth=0.8, linestyle='--', alpha=0.8)
                ax.axhline(-1.5, color='#fc8d59', linewidth=0.8, linestyle='--', alpha=0.8)
                ax.axhline(-2.0, color='#d73027', linewidth=0.8, linestyle='--', alpha=0.8)
                ax.scatter([13], [pred_spi], color=color, s=100, zorder=5, label=f'Predicted month 13: {pred_spi:.3f}')
                ax.set_xticks(list(range(1, 14)))
                ax.set_xticklabels([f"M{i}" for i in range(1, 13)] + ["Pred"], fontsize=9)
                ax.set_ylabel("SPI-3")
                ax.set_title(f"SPI-3 Input Window + Prediction — {labels[prov_key]}", fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

        except ValueError as e:
            st.error(f"Could not parse input values: {e}")

    st.divider()

    # ── Saved result figures ────────────────────────────────────────
    st.subheader("📈 Thesis Result Figures")
    drought_figs = {
        "SPI-3 time series — KZN":          "figures/SPI3_timeseries_KZN.png",
        "SPI-3 time series — Northern Cape": "figures/SPI3_timeseries_NC.png",
        "LSTM training curves":              "figures/LSTM_training_curves.png",
        "Predicted vs actual SPI-3":         "figures/LSTM_prediction_vs_actual.png",
        "Scatter plot":                      "figures/LSTM_scatter.png",
    }
    fig_choice = st.selectbox("Select figure:", list(drought_figs.keys()))
    fpath = drought_figs[fig_choice]
    if os.path.exists(fpath):
        st.image(fpath, use_container_width=True)
    else:
        st.info(
            f"Figure not found at `{fpath}`.  \n"
            "Run your drought notebook and copy the output figures to the `figures/` folder."
        )


# =====================================================================
# ── PIPELINE PAGE ─────────────────────────────────────────────────────
# =====================================================================
elif page == "📓  Pipeline Notebooks":
    st.title("📓 Full Pipeline")
    st.markdown(
        "All data acquisition, preprocessing, and training notebooks. Best run in "
        "Google Colab with a GPU runtime (T4 or A100)."
    )
    st.divider()

    REPO = "https://github.com/ValenxiaA/kzn-flood-detection"
    COLAB_BASE = "https://colab.research.google.com/github/ValenxiaA/kzn-flood-detection/blob/main/notebooks/"

    st.subheader("🌊 Flood Detection Pipeline")
    flood_notebooks = [
        ("B1 — Study area definition",         "B1_study_area_definition_final.ipynb",   "Tile selection from UNOSAT geometry"),
        ("B2 — Data acquisition",              "B2_acquisition_final.ipynb",              "Sentinel-1 + Sentinel-2 GEE export"),
        ("B3 — Mosaic assembly",               "B3_mosaic_final.ipynb",                   "Reprojection & mosaic (EPSG:32736)"),
        ("B4 — Stack assembly",                "B4_stack_assembly_final.ipynb",            "12-channel raster stack"),
        ("B5 — Patch extraction",              "B5_patch_extraction_final.ipynb",          "128×128 patch sampling with nodata filter"),
        ("B6 — Model training ⚡ GPU",         "B6_model_trainingfinal.ipynb",             "U-Net training, fusion & optical configs"),
        ("B7 — Results figures",               "B7_final_chapter4_figures_final.ipynb",    "Chapter 4 thesis figures"),
    ]

    for name, fname, desc in flood_notebooks:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"**{name}**  \n<small style='color:#666'>{desc}</small>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"[GitHub]({REPO}/blob/main/notebooks/{fname})")
        with c3:
            st.markdown(f"[Colab]({COLAB_BASE}{fname})")
        st.divider()

    st.subheader("🌵 Drought Pipeline")
    drought_notebooks = [
        ("Drought — LSTM SPI-3 prediction", "KZN_NC_Drought_Thesis_Final.ipynb",
         "CHIRPS + MODIS NDVI, SPI-3, two-layer LSTM, KZN & Northern Cape"),
    ]
    for name, fname, desc in drought_notebooks:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"**{name}**  \n<small style='color:#666'>{desc}</small>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"[GitHub]({REPO}/blob/main/notebooks/{fname})")
        with c3:
            st.markdown(f"[Colab]({COLAB_BASE}{fname})")
        st.divider()

    st.info(
        "Training notebooks (B6, Drought LSTM) require a GPU and will be very slow on Colab free tier. "
        "Set runtime type to GPU before running."
    )

    st.markdown(f"[View full repository on GitHub]({REPO})")
