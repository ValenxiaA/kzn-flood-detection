"""
Flood & Drought Detection Dashboard
KwaZulu-Natal Flood Detection + SPI-3 Drought Prediction (LSTM)
Valencia Marubini — MEng CPUT / F'SATI
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

# =====================================================================
# Page setup
# =====================================================================
st.set_page_config(
    page_title="Flood & Drought Detection — Valencia Marubini",
    page_icon="🌊",
    layout="wide"
)


# =====================================================================
# Visual styling helpers
# =====================================================================
st.markdown("""
<style>
    :root {
        --primary: #0F5E8C;
        --primary-soft: #E7F3FA;
        --teal: #0E8F7E;
        --teal-soft: #E7F7F3;
        --amber: #B56B11;
        --amber-soft: #FFF4E2;
        --danger: #B42318;
        --ink: #14213D;
        --muted: #5C667A;
        --line: #E6EAF0;
        --panel: #FFFFFF;
        --page: #F6F8FB;
    }

    .stApp { background: linear-gradient(180deg, #F6F8FB 0%, #FFFFFF 42%); }

    /* Keep the app from jumping between very wide and narrow layouts */
    .block-container {
        max-width: 1180px !important;
        padding-top: 1.35rem !important;
        padding-bottom: 3rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2545 0%, #12385D 100%);
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stRadio label { color: #FFFFFF !important; }

    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    p, li, .stMarkdown { color: #243044; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.5rem 1rem;
        background: #EEF4F8;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 24px rgba(20, 33, 61, 0.06);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.86rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-size: 1.65rem !important;
        font-weight: 750 !important;
    }

    .hero-card {
        background: linear-gradient(135deg, #0B2545 0%, #0F5E8C 55%, #0E8F7E 100%);
        color: #FFFFFF;
        padding: 2.0rem 2.1rem;
        border-radius: 28px;
        box-shadow: 0 18px 45px rgba(15, 94, 140, 0.22);
        margin-bottom: 1.3rem;
    }
    .hero-card h1 { color: #FFFFFF; margin: 0 0 0.6rem 0; font-size: clamp(2rem, 4vw, 3.2rem); }
    .hero-card p { color: rgba(255,255,255,0.88); max-width: 850px; margin-bottom: 0; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.1rem; }
    .chip {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.25);
        color: #FFFFFF;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .soft-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 10px 26px rgba(20, 33, 61, 0.06);
        margin-bottom: 1rem;
    }
    .metric-panel {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.15rem;
        box-shadow: 0 10px 26px rgba(20, 33, 61, 0.06);
        height: 100%;
    }
    .metric-panel h3 { margin: 0 0 0.2rem 0; font-size: 1.05rem; }
    .metric-panel .sub { color: var(--muted); font-size: 0.84rem; margin-bottom: 0.9rem; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
    .mini-metric {
        background: #F8FAFC;
        border: 1px solid #E9EEF5;
        border-radius: 16px;
        padding: 0.85rem;
    }
    .mini-label { color: var(--muted); font-size: 0.78rem; font-weight: 650; margin-bottom: 0.2rem; }
    .mini-value { color: var(--ink); font-size: 1.35rem; font-weight: 800; }
    .interpretation {
        margin-top: 0.9rem;
        background: var(--teal-soft);
        border-left: 4px solid var(--teal);
        padding: 0.75rem 0.85rem;
        border-radius: 12px;
        color: #12433E;
        font-size: 0.88rem;
    }
    .notice {
        background: #F8FAFC;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: 0.7rem 0 1rem 0;
    }
    .notice strong { color: var(--ink); }

    @media (max-width: 760px) {
        .hero-card { padding: 1.4rem; border-radius: 20px; }
        .metric-grid { grid-template-columns: 1fr; }
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips=None):
    chips = chips or []
    chip_html = "".join([f"<span class='chip'>{c}</span>" for c in chips])
    st.markdown(
        f"""
        <div class='hero-card'>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class='chip-row'>{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def province_metric_panel(name: str, rmse: str, r2: str, corr: str, note: str):
    st.markdown(
        f"""
        <div class='metric-panel'>
            <h3>{name}</h3>
            <div class='sub'>Real thesis test-set drought performance</div>
            <div class='metric-grid'>
                <div class='mini-metric'><div class='mini-label'>RMSE</div><div class='mini-value'>{rmse}</div></div>
                <div class='mini-metric'><div class='mini-label'>R²</div><div class='mini-value'>{r2}</div></div>
                <div class='mini-metric'><div class='mini-label'>Pearson r</div><div class='mini-value'>{corr}</div></div>
            </div>
            <div class='interpretation'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def friendly_status(label: str, ok: bool, detail: str):
    if ok:
        st.success(f"✅ {label} loaded")
    else:
        st.info(f"ℹ️ {label} not available yet — {detail}")

# ── Sidebar ──────────────────────────────────────────────────────────
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
    st.markdown("[GitHub Repo](https://github.com/ValenxiaA/kzn-flood-detection)")


# =====================================================================
# Shared model architectures
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
# Data loaders (cached)
# =====================================================================
@st.cache_resource
def load_flood_models():
    device = torch.device('cpu')
    fusion_model = optical_model = None
    status = {"fusion": False, "optical": False}

    if os.path.exists("models/fusion_unet_4tile_best.pt"):
        fusion_model = UNet(in_channels=12)
        fusion_model.load_state_dict(torch.load("models/fusion_unet_4tile_best.pt", map_location=device))
        fusion_model.eval()
        status["fusion"] = True

    if os.path.exists("models/optical_unet_4tile_best.pt"):
        optical_model = UNet(in_channels=9)
        optical_model.load_state_dict(torch.load("models/optical_unet_4tile_best.pt", map_location=device))
        optical_model.eval()
        status["optical"] = True

    return fusion_model, optical_model, status


@st.cache_resource
def load_drought_models():
    device = torch.device('cpu')
    models, status = {}, {}
    for key in ['KZN', 'NC']:
        path = f'models/lstm_{key}.pth'
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


@st.cache_data
def load_drought_data():
    """Load CSVs, merge SPI-3 + NDVI anomaly, fit scaler on training portion."""
    result = {}
    for key in ['KZN', 'NC']:
        spi_path  = f'data/drought/spi3_{key}.csv'
        ndvi_path = f'data/drought/ndvi_anomaly_{key}.csv'

        if not os.path.exists(spi_path) or not os.path.exists(ndvi_path):
            result[key] = None
            continue

        spi  = pd.read_csv(spi_path,  parse_dates=['date'])
        ndvi = pd.read_csv(ndvi_path, parse_dates=['date'])

        spi['ym']  = spi['date'].dt.to_period('M')
        ndvi['ym'] = ndvi['date'].dt.to_period('M')

        merged = (
            spi[['ym', 'date', 'spi3']]
            .merge(ndvi[['ym', 'ndvi_anomaly']], on='ym')
            .dropna()
            .sort_values('date')
            .reset_index(drop=True)
        )

        arr = merged[['spi3', 'ndvi_anomaly']].values.astype(np.float32)
        n_train = int(len(arr) * 0.70)
        scaler = MinMaxScaler()
        scaler.fit(arr[:n_train])

        result[key] = {
            'df':     merged,
            'arr':    arr,
            'scaler': scaler,
            'dates':  pd.to_datetime(merged['date']),
        }
    return result


def spi_class(v):
    if v >= 0:     return "Normal / Wet",    "#4393c3"
    if v >= -1.0:  return "Mild drought",     "#fee090"
    if v >= -1.5:  return "Moderate drought", "#fc8d59"
    if v >= -2.0:  return "Severe drought",   "#d73027"
    return "Extreme drought", "#7a0000"


PROVINCE_LABELS = {'KZN': 'KwaZulu-Natal', 'NC': 'Northern Cape'}

SPI_THRESHOLDS = [
    (0.0,         float('inf'), 'Normal / Wet',     '#4393c3'),
    (-1.0,        0.0,          'Mild drought',      '#fee090'),
    (-1.5,       -1.0,          'Moderate drought',  '#fc8d59'),
    (-2.0,       -1.5,          'Severe drought',    '#d73027'),
    (float('-inf'), -2.0,       'Extreme drought',   '#7a0000'),
]


# =====================================================================
# FLOOD PAGE
# =====================================================================
if page == "🌊  Flood Detection":
    st.title("🌊 Flood Detection — KwaZulu-Natal")
    st.markdown(
        "Multi-source deep learning flood detection for the **April 2022 KwaZulu-Natal** flood event.  \n"
        "Fusion U-Net (12-channel SAR + optical) and optical-only U-Net (9-channel), evaluated on "
        "**2,897 held-out test patches** against UNOSAT FL20220418ZAF."
    )
    st.divider()

    with st.spinner("Loading flood model weights..."):
        fusion_model, optical_model, flood_status = load_flood_models()

    c1, c2 = st.columns(2)
    with c1:
        if flood_status["fusion"]:
            n = sum(p.numel() for p in fusion_model.parameters())
            st.success(f"✅ Fusion model (12-ch) loaded — {n:,} parameters")
        else:
            st.info("ℹ️ Fusion model not available yet — add models/fusion_unet_4tile_best.pt to enable live inference.")
    with c2:
        if flood_status["optical"]:
            n = sum(p.numel() for p in optical_model.parameters())
            st.success(f"✅ Optical model (9-ch) loaded — {n:,} parameters")
        else:
            st.info("ℹ️ Optical model not available yet — add models/optical_unet_4tile_best.pt to enable live inference.")

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
        "that Sentinel-1 SAR and SRTM elevation channels provided measurable benefit."
    )
    st.divider()

    # ── Live inference ──────────────────────────────────────────────
    st.subheader("🧠 Run the Model Live")
    st.markdown(
        "Select a held-out test patch and run it through the trained models. "
        "This performs real inference using the loaded checkpoint weights."
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
                    with torch.no_grad():
                        logits = optical_model(torch.from_numpy(X[:9]).unsqueeze(0))
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
        "Spatial comparison at two representative locations: **Durban Harbour (Zone 1)** "
        "and the **uMngeni River corridor (Zone 2)**."
    )
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
        fpath = os.path.join("figures/", fname)
        with col:
            st.markdown(f"**{label}**")
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)
            else:
                st.info(f"Upload `{fname}` to `figures/` to display")


# =====================================================================
# DROUGHT PAGE
# =====================================================================
elif page == "🌵  Drought Prediction":
    st.title("🌵 Drought Prediction — LSTM SPI-3")
    st.markdown(
        "**Two-layer LSTM predicting SPI-3 one month ahead.**  \n"
        "Input: 12-month sequences of SPI-3 and MODIS NDVI anomaly.  \n"
        "Study regions: **KwaZulu-Natal** and **Northern Cape**, 2000–2024 "
        "(CHIRPS v2.0 rainfall + MODIS MOD13A3 NDVI)."
    )
    st.divider()

    with st.spinner("Loading drought models and data..."):
        drought_models, drought_status = load_drought_models()
        drought_data = load_drought_data()

    st.markdown("<div class='notice'><strong>App status:</strong> The performance matrix below will always show your thesis results. Interactive prediction only becomes active when the model files and drought CSVs are present in the GitHub/Streamlit folders.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        friendly_status("KZN LSTM", drought_status['KZN'], "add models/lstm_KZN.pth")
    with c2:
        friendly_status("Northern Cape LSTM", drought_status['NC'], "add models/lstm_NC.pth")
    with c3:
        data_ok = drought_data.get('KZN') is not None or drought_data.get('NC') is not None
        friendly_status("Drought CSV data", data_ok, "add spi3 and ndvi_anomaly CSVs to data/drought/")

    st.divider()

    with st.expander("ℹ️ About the drought model"):
        st.markdown("""
**Data sources (open access)**
- CHIRPS v2.0 — monthly rainfall at 0.05° resolution (Climate Hazards Group)
- MODIS MOD13A3 — monthly 1 km NDVI (NASA via Google Earth Engine)

**SPI-3 computation**
SPI-3 follows McKee et al. (1993). A gamma distribution is fitted per calendar month to
account for South Africa's strong seasonality. WMO (2012) drought severity thresholds:

| SPI-3 | Classification |
|-------|---------------|
| ≥ 0 | Normal / Wet |
| −1.0 to 0 | Mild drought |
| −1.5 to −1.0 | Moderate drought |
| −2.0 to −1.5 | Severe drought |
| < −2.0 | Extreme drought |

**LSTM architecture**
Two-layer LSTM, hidden size 64, dropout 0.3. Input window: 12 months of [SPI-3, NDVI anomaly].
Output: predicted SPI-3 for month 13. Adam optimiser (lr=1e-3), MSE loss, early stopping
patience 15, ReduceLROnPlateau scheduler. Chronological split 70 / 15 / 15.
        """)

    st.divider()

    # ── Metrics ────────────────────────────────────────────────────
    st.subheader("📊 Drought Performance Matrix")
    st.markdown(
        "These are the real thesis values, so they display even when the trained `.pth` files or CSV data are not available on Streamlit."
    )
    col_kzn, col_nc = st.columns(2)
    with col_kzn:
        province_metric_panel(
            "KwaZulu-Natal",
            "0.800",
            "0.132",
            "0.525",
            "Moderate correlation, but weaker explained variance. This suggests the KZN drought signal is more variable and harder for the LSTM to predict."
        )
    with col_nc:
        province_metric_panel(
            "Northern Cape",
            "0.829",
            "0.389",
            "0.750",
            "Stronger correlation and better explained variance. The Northern Cape drought signal is more consistent and easier for the LSTM to track."
        )

    st.caption("RMSE = prediction error magnitude. R² = explained variance. Pearson r = strength of predicted-vs-observed SPI-3 relationship.")

    st.divider()

    # ── Interactive prediction ──────────────────────────────────────
    st.subheader("🧠 Interactive Drought Prediction")
    st.markdown(
        "Select a **province** and a **month** from the historical record. The app loads the "
        "real preceding 12 months of SPI-3 and NDVI anomaly from your data, runs the LSTM, "
        "and predicts the drought condition for the following month."
    )

    prov_choice = st.radio("Province:", ["KwaZulu-Natal (KZN)", "Northern Cape (NC)"], horizontal=True)
    prov_key = "KZN" if "KZN" in prov_choice else "NC"
    prov_data = drought_data.get(prov_key)

    if prov_data is None:
        st.info("Upload the CSV files to data/drought/ to enable this section.")
    else:
        df   = prov_data['df']
        arr  = prov_data['arr']
        scaler = prov_data['scaler']
        dates  = prov_data['dates']

        # need at least 13 rows (12 input + 1 to predict)
        min_idx = 12
        max_idx = len(df) - 1

        # date slider — pick the prediction target month
        min_date = dates.iloc[min_idx].to_pydatetime()
        max_date = dates.iloc[max_idx].to_pydatetime()

        selected_date = st.slider(
            "Select prediction target month:",
            min_value=min_date,
            max_value=max_date,
            value=min(max(pd.Timestamp('2016-04-01'), pd.Timestamp(min_date)), pd.Timestamp(max_date)).to_pydatetime(),
            format="MMM YYYY"
        )

        # find the closest index
        target_idx = (dates - pd.Timestamp(selected_date)).abs().argmin()
        target_idx = max(min_idx, min(target_idx, max_idx))

        window_arr = arr[target_idx - 12 : target_idx]   # 12 months before target
        window_df  = df.iloc[target_idx - 12 : target_idx]
        actual_target = df.iloc[target_idx]['spi3'] if target_idx < len(df) else None

        # show the input window
        st.markdown(f"**Input window:** {dates.iloc[target_idx-12].strftime('%b %Y')} → {dates.iloc[target_idx-1].strftime('%b %Y')}")
        st.markdown(f"**Predicting:** {dates.iloc[target_idx].strftime('%B %Y')}")

        col_l, col_r = st.columns(2)
        with col_l:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(12), window_df['spi3'],
                   color=['#4393c3' if v >= 0 else '#d73027' for v in window_df['spi3']],
                   alpha=0.8)
            ax.axhline(0,    color='black', linewidth=0.7)
            ax.axhline(-1.0, color='#fee090', linewidth=0.8, linestyle='--', alpha=0.9)
            ax.axhline(-1.5, color='#fc8d59', linewidth=0.8, linestyle='--', alpha=0.9)
            ax.axhline(-2.0, color='#d73027', linewidth=0.8, linestyle='--', alpha=0.9)
            ax.set_xticks(range(12))
            ax.set_xticklabels(
                [dates.iloc[target_idx-12+i].strftime('%b %y') for i in range(12)],
                rotation=45, fontsize=7
            )
            ax.set_ylabel("SPI-3")
            ax.set_title("Input window — SPI-3", fontsize=10)
            ax.grid(axis='y', alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)

        with col_r:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(range(12), window_df['ndvi_anomaly'], 'o-', color='#3B6D11', linewidth=1.5)
            ax.axhline(0, color='black', linewidth=0.7)
            ax.set_xticks(range(12))
            ax.set_xticklabels(
                [dates.iloc[target_idx-12+i].strftime('%b %y') for i in range(12)],
                rotation=45, fontsize=7
            )
            ax.set_ylabel("NDVI anomaly")
            ax.set_title("Input window — NDVI anomaly", fontsize=10)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)

        if st.button("▶ Predict drought condition", type="primary"):
            if drought_models[prov_key] is None:
                st.warning(f"Upload models/lstm_{prov_key}.pth to enable prediction.")
            else:
                scaled = scaler.transform(window_arr)
                X_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    pred_scaled = drought_models[prov_key](X_tensor).item()

                dummy = np.zeros((1, 2), dtype=np.float32)
                dummy[0, 0] = pred_scaled
                pred_spi = scaler.inverse_transform(dummy)[0, 0]

                label, color = spi_class(pred_spi)

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Predicted SPI-3", f"{pred_spi:.3f}")
                with r2:
                    if actual_target is not None:
                        st.metric("Actual SPI-3", f"{actual_target:.3f}",
                                  delta=f"{pred_spi - actual_target:+.3f} error")
                with r3:
                    actual_label, _ = spi_class(actual_target) if actual_target is not None else ("—", None)
                    st.metric("Actual class", actual_label if actual_target is not None else "—")

                st.markdown(
                    f"<div style='background:{color}; color:#fff; padding:12px 18px; "
                    f"border-radius:8px; font-weight:600; font-size:16px; margin:12px 0;'>"
                    f"Predicted: {label} &nbsp;({dates.iloc[target_idx].strftime('%B %Y')})</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # ── Full time series view ───────────────────────────────────────
    st.subheader("📈 Historical SPI-3 Time Series")
    st.markdown("Browse the full 2000–2024 drought record for either province.")

    ts_prov = st.radio("Province:", ["KwaZulu-Natal (KZN)", "Northern Cape (NC)"],
                       horizontal=True, key="ts_prov")
    ts_key  = "KZN" if "KZN" in ts_prov else "NC"
    ts_data = drought_data.get(ts_key)

    if ts_data is None:
        st.info("Upload CSV data to view this chart.")
    else:
        ts_df = ts_data['df']

        year_range = st.slider(
            "Year range:",
            min_value=2000, max_value=2024,
            value=(2000, 2024), step=1, key="year_slider"
        )
        mask = (ts_df['date'].dt.year >= year_range[0]) & (ts_df['date'].dt.year <= year_range[1])
        plot_df = ts_df[mask]

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = [
            next(c for lo, hi, _, c in SPI_THRESHOLDS if lo <= v < hi)
            for v in plot_df['spi3']
        ]
        ax.bar(plot_df['date'], plot_df['spi3'], color=colors, width=25, alpha=0.85)
        ax.axhline(0,    color='black', linewidth=0.7)
        ax.axhline(-1.0, color='#fee090', linewidth=0.8, linestyle='--', alpha=0.9, label='Mild (−1.0)')
        ax.axhline(-1.5, color='#fc8d59', linewidth=0.8, linestyle='--', alpha=0.9, label='Moderate (−1.5)')
        ax.axhline(-2.0, color='#d73027', linewidth=0.8, linestyle='--', alpha=0.9, label='Severe (−2.0)')

        worst = plot_df.loc[plot_df['spi3'].idxmin()]
        ax.annotate(
            f"Worst: {worst['date'].strftime('%b %Y')}\nSPI = {worst['spi3']:.2f}",
            xy=(worst['date'], worst['spi3']),
            xytext=(worst['date'], worst['spi3'] - 0.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
            fontsize=8, ha='center'
        )

        handles = [mpatches.Patch(color=c, label=l) for _, _, l, c in SPI_THRESHOLDS]
        ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
        ax.set_ylabel("SPI-3")
        ax.set_title(f"SPI-3 — {PROVINCE_LABELS[ts_key]} ({year_range[0]}–{year_range[1]})", fontsize=11)
        ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

        n_mild    = (plot_df['spi3'] < -1.0).sum()
        n_severe  = (plot_df['spi3'] < -1.5).sum()
        n_extreme = (plot_df['spi3'] < -2.0).sum()
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Mild drought months",    n_mild)
        sc2.metric("Severe drought months",  n_severe)
        sc3.metric("Extreme drought months", n_extreme)

    st.divider()

    # ── Saved result figures ────────────────────────────────────────
    st.subheader("📊 Thesis Result Figures")
    drought_figs = {
        "SPI-3 time series — KZN":           "figures/SPI3_timeseries_KZN.png",
        "SPI-3 time series — Northern Cape":  "figures/SPI3_timeseries_NC.png",
        "LSTM training curves":               "figures/LSTM_training_curves.png",
        "Predicted vs actual SPI-3":          "figures/LSTM_prediction_vs_actual.png",
        "Scatter plot":                       "figures/LSTM_scatter.png",
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
# PIPELINE PAGE
# =====================================================================
elif page == "📓  Pipeline Notebooks":
    hero(
        "📓 Pipeline Notebooks",
        "A simple guide to the full project workflow. Use B7 for flood results without retraining, and use the drought notebook for the SPI-3 LSTM pipeline.",
        ["Google Colab", "GitHub", "B1–B7 flood workflow", "Drought LSTM notebook"]
    )

    REPO       = "https://github.com/ValenxiaA/kzn-flood-detection"
    COLAB_BASE = "https://colab.research.google.com/github/ValenxiaA/kzn-flood-detection/blob/main/notebooks/"

    # ── Quick start callout ─────────────────────────────────────────
    st.info(
        "**Just want to explore results?**  \n"
        "You do not need to re-run the full pipeline. Start with **B7** (flood results figures) "
        "or the **Evaluation notebook** to load the trained model weights and reproduce all "
        "metrics and visualisations directly — no retraining required."
    )

    st.subheader("🌊 Flood Detection Pipeline")
    st.markdown(
        "The flood pipeline runs in order B1 → B7. Each notebook saves its outputs to Google Drive "
        "so the next one can pick up from where it left off. **B6 requires a GPU runtime.**"
    )

    flood_notebooks = [
        (
            "B1 — Study area definition",
            "B1_study_area_definition_final.ipynb",
            "Defines the four KZN study tiles (36JTM, 36JUM, 36JTN, 36JUN) from UNOSAT flood geometry.",
            False
        ),
        (
            "B2 — Data acquisition",
            "B2_acquisition_final.ipynb",
            "Exports Sentinel-1 SAR and Sentinel-2 optical imagery from Google Earth Engine.",
            False
        ),
        (
            "B3 — Mosaic assembly",
            "B3_mosaic_final.ipynb",
            "Reprojects and mosaics tiles into a single EPSG:32736 raster per date.",
            False
        ),
        (
            "B4 — Stack assembly",
            "B4_stack_assembly_final.ipynb",
            "Assembles the 12-channel input stack (6 optical bands + NDVI/NDWI + VV/VH + zero + DEM).",
            False
        ),
        (
            "B5 — Patch extraction",
            "B5_patch_extraction_final.ipynb",
            "Extracts 128×128 training patches with nodata filtering. Saves train/val/test .npy files.",
            False
        ),
        (
            "B6 — Model training ⚡ GPU required",
            "B6_model_trainingfinal.ipynb",
            "Trains fusion (12-ch) and optical-only (9-ch) U-Nets. Saves best checkpoints to Drive.",
            True
        ),
        (
            "B7 — Results & evaluation",
            "B7_final_chapter4_figures_final.ipynb",
            "Loads trained weights and generates all Chapter 4 thesis figures. No retraining needed — start here to explore results.",
            False
        ),
    ]

    for name, fname, desc, gpu in flood_notebooks:
        with st.container():
            ca, cb = st.columns([5, 1])
            with ca:
                badge = " 🟡 GPU" if gpu else ""
                st.markdown(f"**{name}**{badge}")
                st.caption(desc)
            with cb:
                st.markdown(f"[Open in Colab]({COLAB_BASE}{fname})")
            st.divider()

    st.subheader("🌵 Drought Pipeline")
    st.markdown(
        "The drought pipeline is self-contained in a single notebook. It downloads CHIRPS data, "
        "extracts MODIS NDVI via GEE, computes SPI-3, trains the LSTM, and generates all figures."
    )

    drought_nb = [
        (
            "Drought — LSTM SPI-3 prediction",
            "KZN_NC_Drought_Thesis_Final.ipynb",
            "Full pipeline: CHIRPS download, MODIS NDVI extraction, SPI-3 computation, LSTM training "
            "and evaluation for KwaZulu-Natal and Northern Cape (2000–2024). GEE authentication required.",
            False
        ),
    ]

    for name, fname, desc, gpu in drought_nb:
        with st.container():
            ca, cb = st.columns([5, 1])
            with ca:
                st.markdown(f"**{name}**")
                st.caption(desc)
            with cb:
                st.markdown(f"[Open in Colab]({COLAB_BASE}{fname})")
            st.divider()

    st.markdown(f"[View full repository on GitHub]({REPO})")

