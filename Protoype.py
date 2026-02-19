import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Mock "track catalog"
# -----------------------------
TRACKS = [
    # id, title, bpm, energy(0-1), valence(0-1), acoustic(0-1), tags
    ("t01", "Warm Piano — Slow Waves", 60, 0.15, 0.55, 0.85, ["calm", "piano"]),
    ("t02", "Soft Strings — Evening", 68, 0.20, 0.60, 0.70, ["calm", "strings"]),
    ("t03", "Ambient — Breath Focus", 72, 0.18, 0.50, 0.90, ["calm", "ambient"]),
    ("t04", "Lo-fi — Gentle Pulse", 80, 0.35, 0.65, 0.55, ["steady", "lofi"]),
    ("t05", "Acoustic — Easy Listening", 90, 0.40, 0.70, 0.75, ["steady", "acoustic"]),
    ("t06", "Pop — Light & Bright", 105, 0.60, 0.80, 0.25, ["uplift", "pop"]),
    ("t07", "Indie — Upbeat Walk", 120, 0.70, 0.75, 0.30, ["uplift", "indie"]),
    ("t08", "Electronic — Focus Mode", 128, 0.75, 0.55, 0.10, ["focus", "electronic"]),
    ("t09", "Ambient — Deep Night", 55, 0.10, 0.45, 0.95, ["calm", "ambient"]),
    ("t10", "Guitar — Soft Sunrise", 75, 0.25, 0.70, 0.80, ["calm", "acoustic"]),
]

CATALOG = pd.DataFrame(TRACKS, columns=["id", "title", "bpm", "energy", "valence", "acoustic", "tags"])


# -----------------------------
# Biomarker model + scoring
# -----------------------------
@dataclass
class Biomarkers:
    hr: float          # bpm
    hrv_rmssd: float   # ms
    eda: float         # µS (skin conductance level proxy)
    sbp: float         # systolic BP mmHg
    rr: float          # respiration rate /min


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def normalize_biomarkers(b: Biomarkers) -> Dict[str, float]:
    """
    Very rough normalizations to 0..1 (higher = more arousal / stress),
    tuned for a prototype. Replace with population baselines / per-patient baselines later.
    """
    # HR: 55 (low) .. 120 (high)
    hr_n = clamp01((b.hr - 55) / (120 - 55))

    # HRV (RMSSD): 15 (low) .. 80 (high) -> invert (low HRV means higher stress)
    hrv_n = 1.0 - clamp01((b.hrv_rmssd - 15) / (80 - 15))

    # EDA: 0.5 (low) .. 10 (high)
    eda_n = clamp01((b.eda - 0.5) / (10 - 0.5))

    # SBP: 95 (low) .. 160 (high)
    sbp_n = clamp01((b.sbp - 95) / (160 - 95))

    # RR: 10 (low) .. 26 (high)
    rr_n = clamp01((b.rr - 10) / (26 - 10))

    return {"hr": hr_n, "hrv": hrv_n, "eda": eda_n, "sbp": sbp_n, "rr": rr_n}


def arousal_index(norm: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Weighted sum -> 0..1. This is your "AI score" MVP.
    """
    s = 0.0
    wsum = 0.0
    for k, w in weights.items():
        s += w * norm[k]
        wsum += w
    return float(s / wsum) if wsum > 0 else 0.0


def target_profile(score: float) -> Dict[str, float]:
    """
    Map arousal score to desired music features.
    Higher score => calmer music: lower bpm/energy, higher acoustic, moderate valence.
    Lower score => can be a bit more upbeat.
    """
    score = clamp01(score)

    # Desired ranges (rough)
    # bpm target: 60..120 (inverse with score)
    bpm_t = 120 - 60 * score
    # energy target: 0.15..0.75 (inverse with score)
    energy_t = 0.75 - 0.60 * score
    # acoustic target: 0.20..0.95 (increase with score)
    acoustic_t = 0.20 + 0.75 * score
    # valence: keep somewhat positive but not overly stimulating
    valence_t = 0.60 - 0.10 * score  # 0.50..0.60

    return {"bpm": bpm_t, "energy": energy_t, "acoustic": acoustic_t, "valence": valence_t}


def recommend_playlist(score: float, n: int = 6) -> pd.DataFrame:
    prof = target_profile(score)

    # Distance metric: scaled differences
    df = CATALOG.copy()
    df["dist"] = (
        ((df["bpm"] - prof["bpm"]) / 30.0) ** 2 +
        ((df["energy"] - prof["energy"]) / 0.35) ** 2 +
        ((df["acoustic"] - prof["acoustic"]) / 0.40) ** 2 +
        ((df["valence"] - prof["valence"]) / 0.30) ** 2
    )

    df = df.sort_values("dist", ascending=True).head(n)
    return df[["id", "title", "bpm", "energy", "valence", "acoustic", "tags", "dist"]]


# -----------------------------
# Simulated biomarker stream (for demo)
# -----------------------------
def simulate_biomarkers(prev: Biomarkers, drift: float = 0.15) -> Biomarkers:
    rng = np.random.default_rng()

    # Gentle random walk + occasional spikes
    spike = rng.random() < 0.08
    spike_mag = rng.uniform(0.8, 1.3) if spike else 1.0

    hr = np.clip(prev.hr + rng.normal(0, 2.0) * drift * spike_mag, 50, 140)
    hrv = np.clip(prev.hrv_rmssd + rng.normal(0, 3.0) * drift / spike_mag, 5, 120)
    eda = np.clip(prev.eda + rng.normal(0, 0.3) * drift * spike_mag, 0.1, 15)
    sbp = np.clip(prev.sbp + rng.normal(0, 3.0) * drift * spike_mag, 85, 180)
    rr = np.clip(prev.rr + rng.normal(0, 0.8) * drift * spike_mag, 8, 30)

    return Biomarkers(hr=float(hr), hrv_rmssd=float(hrv), eda=float(eda), sbp=float(sbp), rr=float(rr))


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Adaptive Music Therapy Prototype", layout="wide")
st.title("🎧 Adaptive Music Prototype (Biomarkers → AI score → Playlist)")

with st.sidebar:
    st.header("Input mode")
    mode = st.radio("Select mode", ["Manual", "Simulated stream"], index=0)

    st.header("AI weights (arousal index)")
    weights = {
        "hr": st.slider("Weight HR", 0.0, 5.0, 1.5, 0.1),
        "hrv": st.slider("Weight HRV (inverse)", 0.0, 5.0, 2.0, 0.1),
        "eda": st.slider("Weight EDA", 0.0, 5.0, 2.0, 0.1),
        "sbp": st.slider("Weight SBP", 0.0, 5.0, 1.0, 0.1),
        "rr": st.slider("Weight RR", 0.0, 5.0, 1.0, 0.1),
    }

    st.header("Playlist settings")
    n_tracks = st.slider("Tracks in playlist", 3, 10, 6, 1)

    st.header("Closed-loop demo")
    refresh_s = st.slider("Update interval (sec)", 1, 10, 3, 1)
    run_loop = st.toggle("Run closed-loop updates", value=False)

# Initialize session state
if "b" not in st.session_state:
    st.session_state.b = Biomarkers(hr=78, hrv_rmssd=35, eda=2.0, sbp=125, rr=14)
if "log" not in st.session_state:
    st.session_state.log = []  # list of dict rows


def log_step(b: Biomarkers, score: float):
    st.session_state.log.append({
        "timestamp": pd.Timestamp.now(),
        "hr": b.hr,
        "hrv_rmssd": b.hrv_rmssd,
        "eda": b.eda,
        "sbp": b.sbp,
        "rr": b.rr,
        "arousal_score": score,
    })


def compute_and_render(b: Biomarkers):
    norm = normalize_biomarkers(b)
    score = arousal_index(norm, weights)
    pl = recommend_playlist(score, n=n_tracks)

    col1, col2, col3 = st.columns([1.1, 1.0, 1.2], gap="large")

    with col1:
        st.subheader("Biomarkers")
        st.metric("HR (bpm)", f"{b.hr:.0f}")
        st.metric("HRV RMSSD (ms)", f"{b.hrv_rmssd:.0f}")
        st.metric("EDA (µS)", f"{b.eda:.2f}")
        st.metric("SBP (mmHg)", f"{b.sbp:.0f}")
        st.metric("RR (/min)", f"{b.rr:.0f}")

        st.subheader("Normalized (0..1, higher = more arousal)")
        st.write({k: round(v, 3) for k, v in norm.items()})

    with col2:
        st.subheader("AI state")
        st.metric("Arousal / pain-proxy score", f"{score:.3f}")

        prof = target_profile(score)
        st.subheader("Target music profile")
        st.write({
            "target_bpm": round(prof["bpm"], 1),
            "target_energy": round(prof["energy"], 3),
            "target_acoustic": round(prof["acoustic"], 3),
            "target_valence": round(prof["valence"], 3),
        })

        # simple interpretation
        if score >= 0.75:
            st.info("State: High arousal → recommend calm, slow, acoustic, predictable tracks.")
        elif score >= 0.45:
            st.info("State: Moderate arousal → recommend steady, low-mid energy tracks.")
        else:
            st.info("State: Low arousal → can allow more upbeat/variety.")

    with col3:
        st.subheader("Recommended playlist")
        st.dataframe(pl, use_container_width=True, hide_index=True)

    log_step(b, score)


# Manual input UI
if mode == "Manual":
    st.session_state.b = Biomarkers(
        hr=st.slider("HR (bpm)", 45, 140, int(st.session_state.b.hr)),
        hrv_rmssd=st.slider("HRV RMSSD (ms)", 5, 120, int(st.session_state.b.hrv_rmssd)),
        eda=st.slider("EDA (µS)", 0.1, 15.0, float(st.session_state.b.eda), 0.1),
        sbp=st.slider("SBP (mmHg)", 85, 180, int(st.session_state.b.sbp)),
        rr=st.slider("RR (/min)", 8, 30, int(st.session_state.b.rr)),
    )

    compute_and_render(st.session_state.b)

else:
    # Simulated mode
    st.caption("Simulated biomarker stream: random walk + occasional spikes (for demo).")
    compute_and_render(st.session_state.b)

    if run_loop:
        # Update loop: simulate new biomarkers and rerun
        time.sleep(refresh_s)
        st.session_state.b = simulate_biomarkers(st.session_state.b)
        st.rerun()

# Log + export
st.divider()
st.subheader("Session log")

if len(st.session_state.log) == 0:
    st.write("No log yet.")
else:
    log_df = pd.DataFrame(st.session_state.log).sort_values("timestamp", ascending=False)
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download log as CSV", data=csv, file_name="biomarker_ai_log.csv", mime="text/csv")

    if st.button("Clear log"):
        st.session_state.log = []
        st.rerun()
