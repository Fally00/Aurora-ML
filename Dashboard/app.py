"""
Dashboard/app.py — SIT-Py Intelligence Dashboard (v2)
======================================================
Upgraded from basic metrics display to a full intelligence dashboard with:
  - Plotly gauge cards (CPU / RAM / Disk) with threshold color bands
  - Time-series chart with Prediction vs Actual overlay
  - Anomaly timeline bar chart (color-coded Normal/Alert)
  - Workload distribution donut chart
  - Model confidence trend line
  - Feature importance explainability panel
  - Retrain trigger button with live data row count
  - Session statistics with drift indicators
"""
from __future__ import annotations

import os
import sys
import time
import platform
from collections import deque
from datetime import datetime

import psutil
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import (
    MODELS_DIR, LOG_PATH, LIVE_CSV,
    ANOMALY_Z_THRESHOLD, RETRAIN_ROW_THRESHOLD,
)
from Core.schema              import SystemSnapshot
from Core.feature_engineering import FeatureEngineer
from Core.predictor           import ResourcePredictor
from Core.anomaly             import AnomalyDetector
from Core.classifier          import WorkloadClassifier
from Data.collect             import take_snapshot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SIT-Py | Intelligence Dashboard",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: #080b14; color: #e2e8f0; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1623 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: border-color 0.3s;
}
[data-testid="metric-container"]:hover { border-color: #3b82f6; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.5px;
}
.badge-normal   { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }
.badge-alert    { background: #2e1010; color: #f87171; border: 1px solid #991b1b; }
.badge-critical { background: #2e1010; color: #f87171; border: 1px solid #991b1b; }
.badge-high     { background: #2e1f0a; color: #fb923c; border: 1px solid #92400e; }
.badge-medium   { background: #0a1e2e; color: #60a5fa; border: 1px solid #1d4ed8; }
.badge-low      { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }

/* Section headers */
.section-title {
    color: #475569;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2d4a;
}

/* Stat bridge info */
.bridge-info {
    background: #0a1929;
    border: 1px solid #1565c0;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: #90caf9;
    margin-top: 6px;
}

/* Streamlit chrome hide */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    predictor  = ResourcePredictor(MODELS_DIR)
    detector   = AnomalyDetector(MODELS_DIR)
    classifier = WorkloadClassifier(MODELS_DIR)

    pred_ok = True
    try:
        predictor.load()
    except FileNotFoundError:
        pred_ok = False

    if_ok = detector.load()   # returns False → stat bridge

    cls_ok = True
    try:
        classifier.load()
    except FileNotFoundError:
        cls_ok = False

    return predictor, detector, classifier, pred_ok, if_ok, cls_ok


# ── Session state init ─────────────────────────────────────────────────────────

def _init_session():
    if "eng" not in st.session_state:
        st.session_state.eng = FeatureEngineer(window_size=15)
    if "prev_net" not in st.session_state:
        st.session_state.prev_net = psutil.net_io_counters()
    if "last_t" not in st.session_state:
        st.session_state.last_t = time.time()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "cycle" not in st.session_state:
        st.session_state.cycle = 0

_init_session()


# ── Pipeline cycle ─────────────────────────────────────────────────────────────

def run_cycle(predictor, detector, classifier, pred_ok, if_ok, cls_ok):
    now     = time.time()
    elapsed = now - st.session_state.last_t
    st.session_state.last_t = now
    st.session_state.cycle += 1

    snap, st.session_state.prev_net = take_snapshot(
        st.session_state.prev_net, elapsed
    )
    snap = st.session_state.eng.enrich(snap)

    # Anomaly
    if if_ok:
        anomaly = detector.predict(snap.to_anomaly_input())
    else:
        anomaly = st.session_state.eng.statistical_anomaly(
            snap, z_threshold=ANOMALY_Z_THRESHOLD
        )

    # Workload
    if cls_ok:
        workload = classifier.predict(snap.to_classifier_input())
    else:
        workload = {"workload": "N/A", "confidence": 0.0, "probabilities": {}}

    # Prediction
    prediction = {}
    if pred_ok and st.session_state.eng.is_ready():
        prediction = predictor.predict(snap.to_predictor_input())

    # Append to history (ring buffer: 120 points)
    record = {
        "timestamp":    snap.timestamp,
        "cpu":          snap.cpu_pct,
        "ram":          snap.ram_pct,
        "disk":         snap.disk_pct,
        "bandwidth":    snap.bandwidth_kbps,
        "process_count":snap.process_count,
        "pred_cpu":     prediction.get("cpu_pct"),
        "pred_ram":     prediction.get("ram_pct"),
        "pred_disk":    prediction.get("disk_pct"),
        "anomaly":      anomaly["label"],
        "is_anomaly":   anomaly["is_anomaly"],
        "a_score":      anomaly["anomaly_score"],
        "a_source":     anomaly.get("source", ""),
        "workload":     workload["workload"],
        "wl_conf":      workload["confidence"],
    }
    st.session_state.history.append(record)
    if len(st.session_state.history) > 120:
        st.session_state.history.pop(0)

    return snap, prediction, anomaly, workload


# ── Chart helpers ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,35,0.8)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(gridcolor="#1e2d4a", showline=False),
    yaxis=dict(gridcolor="#1e2d4a", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
)

def _gauge_chart(title: str, value: float, thresholds=(50, 75, 90)) -> go.Figure:
    """Creates a Plotly gauge with green/yellow/red bands."""
    lo, mid, hi = thresholds
    color = "#4ade80" if value < lo else ("#fb923c" if value < hi else "#f87171")

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        number= {"suffix": "%", "font": {"size": 22, "color": color}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#0f1623",
            "bordercolor": "#1e2d4a",
            "steps": [
                {"range": [0, lo],   "color": "#0d2e1a"},
                {"range": [lo, hi],  "color": "#2e1f0a"},
                {"range": [hi, 100], "color": "#2e1010"},
            ],
            "threshold": {
                "line": {"color": "#f87171", "width": 2},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    layout = PLOTLY_LAYOUT.copy()
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["height"] = 160
    layout["margin"] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(go.Layout(**layout))
    return fig


def _timeseries_chart(df: pd.DataFrame) -> go.Figure:
    """CPU & RAM actual + predicted overlay, last N cycles."""
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cpu"],
        name="CPU (actual)", line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ram"],
        name="RAM (actual)", line=dict(color="#4ade80", width=2),
    ))

    # Predicted overlays
    pred_cpu = df["pred_cpu"].dropna()
    if not pred_cpu.empty:
        fig.add_trace(go.Scatter(
            x=df.loc[pred_cpu.index, "timestamp"], y=pred_cpu,
            name="CPU (predicted)", line=dict(color="#93c5fd", width=1.5, dash="dot"),
        ))
    pred_ram = df["pred_ram"].dropna()
    if not pred_ram.empty:
        fig.add_trace(go.Scatter(
            x=df.loc[pred_ram.index, "timestamp"], y=pred_ram,
            name="RAM (predicted)", line=dict(color="#86efac", width=1.5, dash="dot"),
        ))

    layout = PLOTLY_LAYOUT.copy()
    yaxis = dict(layout.get("yaxis", {}))
    yaxis.update({"title": "Usage (%)", "range": [0, 100]})
    layout["yaxis"] = yaxis
    layout["title"] = "CPU & RAM — Actual vs Predicted"
    layout["height"] = 280
    fig.update_layout(go.Layout(**layout))
    return fig


def _anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """Bar chart of anomaly scores, colored by label."""
    colors = ["#f87171" if v else "#4ade80" for v in df["is_anomaly"]]
    fig = go.Figure(go.Bar("""
Dashboard/app.py — SIT-Py Intelligence Dashboard (v2)
======================================================
Upgraded from basic metrics display to a full intelligence dashboard with:
  - Plotly gauge cards (CPU / RAM / Disk) with threshold color bands
  - Time-series chart with Prediction vs Actual overlay
  - Anomaly timeline bar chart (color-coded Normal/Alert)
  - Workload distribution donut chart
  - Model confidence trend line
  - Feature importance explainability panel
  - Retrain trigger button with live data row count
  - Session statistics with drift indicators
"""
from __future__ import annotations

import os
import sys
import time
import platform
from collections import deque
from datetime import datetime

import psutil
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import (
    MODELS_DIR, LOG_PATH, LIVE_CSV,
    ANOMALY_Z_THRESHOLD, RETRAIN_ROW_THRESHOLD,
)
from Core.schema              import SystemSnapshot
from Core.feature_engineering import FeatureEngineer
from Core.predictor           import ResourcePredictor
from Core.anomaly             import AnomalyDetector
from Core.classifier          import WorkloadClassifier
from Data.collect             import take_snapshot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SIT-Py | Intelligence Dashboard",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: #080b14; color: #e2e8f0; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1623 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: border-color 0.3s;
}
[data-testid="metric-container"]:hover { border-color: #3b82f6; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.5px;
}
.badge-normal   { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }
.badge-alert    { background: #2e1010; color: #f87171; border: 1px solid #991b1b; }
.badge-critical { background: #2e1010; color: #f87171; border: 1px solid #991b1b; }
.badge-high     { background: #2e1f0a; color: #fb923c; border: 1px solid #92400e; }
.badge-medium   { background: #0a1e2e; color: #60a5fa; border: 1px solid #1d4ed8; }
.badge-low      { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }

/* Section headers */
.section-title {
    color: #475569;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2d4a;
}

/* Stat bridge info */
.bridge-info {
    background: #0a1929;
    border: 1px solid #1565c0;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: #90caf9;
    margin-top: 6px;
}

/* Streamlit chrome hide */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    predictor  = ResourcePredictor(MODELS_DIR)
    detector   = AnomalyDetector(MODELS_DIR)
    classifier = WorkloadClassifier(MODELS_DIR)

    pred_ok = True
    try:
        predictor.load()
    except FileNotFoundError:
        pred_ok = False

    if_ok = detector.load()   # returns False → stat bridge

    cls_ok = True
    try:
        classifier.load()
    except FileNotFoundError:
        cls_ok = False

    return predictor, detector, classifier, pred_ok, if_ok, cls_ok


# ── Session state init ─────────────────────────────────────────────────────────

def _init_session():
    if "eng" not in st.session_state:
        st.session_state.eng = FeatureEngineer(window_size=15)
    if "prev_net" not in st.session_state:
        st.session_state.prev_net = psutil.net_io_counters()
    if "last_t" not in st.session_state:
        st.session_state.last_t = time.time()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "cycle" not in st.session_state:
        st.session_state.cycle = 0

_init_session()


# ── Pipeline cycle ─────────────────────────────────────────────────────────────

def run_cycle(predictor, detector, classifier, pred_ok, if_ok, cls_ok):
    now     = time.time()
    elapsed = now - st.session_state.last_t
    st.session_state.last_t = now
    st.session_state.cycle += 1

    snap, st.session_state.prev_net = take_snapshot(
        st.session_state.prev_net, elapsed
    )
    snap = st.session_state.eng.enrich(snap)

    # Anomaly
    if if_ok:
        anomaly = detector.predict(snap.to_anomaly_input())
    else:
        anomaly = st.session_state.eng.statistical_anomaly(
            snap, z_threshold=ANOMALY_Z_THRESHOLD
        )

    # Workload
    if cls_ok:
        workload = classifier.predict(snap.to_classifier_input())
    else:
        workload = {"workload": "N/A", "confidence": 0.0, "probabilities": {}}

    # Prediction
    prediction = {}
    if pred_ok and st.session_state.eng.is_ready():
        prediction = predictor.predict(snap.to_predictor_input())

    # Append to history (ring buffer: 120 points)
    record = {
        "timestamp":    snap.timestamp,
        "cpu":          snap.cpu_pct,
        "ram":          snap.ram_pct,
        "disk":         snap.disk_pct,
        "bandwidth":    snap.bandwidth_kbps,
        "process_count":snap.process_count,
        "pred_cpu":     prediction.get("cpu_pct"),
        "pred_ram":     prediction.get("ram_pct"),
        "pred_disk":    prediction.get("disk_pct"),
        "anomaly":      anomaly["label"],
        "is_anomaly":   anomaly["is_anomaly"],
        "a_score":      anomaly["anomaly_score"],
        "a_source":     anomaly.get("source", ""),
        "workload":     workload["workload"],
        "wl_conf":      workload["confidence"],
    }
    st.session_state.history.append(record)
    if len(st.session_state.history) > 120:
        st.session_state.history.pop(0)

    return snap, prediction, anomaly, workload


# ── Chart helpers ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,35,0.8)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(gridcolor="#1e2d4a", showline=False),
    yaxis=dict(gridcolor="#1e2d4a", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
)

def _gauge_chart(title: str, value: float, thresholds=(50, 75, 90)) -> go.Figure:
    """Creates a Plotly gauge with green/yellow/red bands."""
    lo, mid, hi = thresholds
    color = "#4ade80" if value < lo else ("#fb923c" if value < hi else "#f87171")

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        number= {"suffix": "%", "font": {"size": 22, "color": color}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#0f1623",
            "bordercolor": "#1e2d4a",
            "steps": [
                {"range": [0, lo],   "color": "#0d2e1a"},
                {"range": [lo, hi],  "color": "#2e1f0a"},
                {"range": [hi, 100], "color": "#2e1010"},
            ],
            "threshold": {
                "line": {"color": "#f87171", "width": 2},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    layout = PLOTLY_LAYOUT.copy()
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["height"] = 160
    layout["margin"] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(**layout)
    return fig


def _timeseries_chart(df: pd.DataFrame) -> go.Figure:
    """CPU & RAM actual + predicted overlay, last N cycles."""
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cpu"],
        name="CPU (actual)", line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ram"],
        name="RAM (actual)", line=dict(color="#4ade80", width=2),
    ))

    # Predicted overlays
    pred_cpu = df["pred_cpu"].dropna()
    if not pred_cpu.empty:
        fig.add_trace(go.Scatter(
            x=df.loc[pred_cpu.index, "timestamp"], y=pred_cpu,
            name="CPU (predicted)", line=dict(color="#93c5fd", width=1.5, dash="dot"),
        ))
    pred_ram = df["pred_ram"].dropna()
    if not pred_ram.empty:
        fig.add_trace(go.Scatter(
            x=df.loc[pred_ram.index, "timestamp"], y=pred_ram,
            name="RAM (predicted)", line=dict(color="#86efac", width=1.5, dash="dot"),
        ))

    layout = PLOTLY_LAYOUT.copy()
    yaxis = dict(layout.get("yaxis", {}))
    yaxis.update({"title": "Usage (%)", "range": [0, 100]})
    layout["yaxis"] = yaxis
    layout["title"] = "CPU & RAM — Actual vs Predicted"
    layout["height"] = 280
    fig.update_layout(**layout)
    return fig


def _anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """Bar chart of anomaly scores, colored by label."""
    colors = ["#f87171" if v else "#4ade80" for v in df["is_anomaly"]]
    fig = go.Figure(go.Bar(
        x=df["timestamp"],
        y=df["a_score"].abs(),
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        name="Anomaly score",
    ))
    layout = PLOTLY_LAYOUT.copy()
    yaxis = dict(layout.get("yaxis", {}))
    yaxis.update({"title": "Anomaly Score (abs)"})
    layout["yaxis"] = yaxis
    layout["title"] = "Anomaly Timeline (red = Alert)"
    layout["height"] = 220
    fig.update_layout(**layout)
    return fig


def _workload_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of workload distribution over session."""
    counts = df["workload"].value_counts().reset_index()
    counts.columns = ["workload", "count"]
    color_map = {
        "Critical": "#f87171", "High": "#fb923c",
        "Medium": "#60a5fa", "Low": "#4ade80", "N/A": "#475569",
    }
    fig = go.Figure(go.Pie(
        labels=counts["workload"],
        values=counts["count"],
        hole=0.55,
        marker_colors=[color_map.get(w, "#94a3b8") for w in counts["workload"]],
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    layout = PLOTLY_LAYOUT.copy()
    layout["title"] = "Workload Distribution"
    layout["height"] = 260
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["margin"] = dict(l=0, r=0, t=30, b=0)
    layout["showlegend"] = False
    layout["font"] = dict(color="#94a3b8")
    fig.update_layout(**layout)
    return fig


def _confidence_trend(df: pd.DataFrame) -> go.Figure:
    """Classifier confidence over time — dropping confidence = model drift."""
    fig = go.Figure(go.Scatter(
        x=df["timestamp"], y=df["wl_conf"],
        fill="tozeroy", fillcolor="rgba(147,197,253,0.08)",
        line=dict(color="#93c5fd", width=2),
        name="Confidence",
    ))
    fig.add_hline(y=0.5, line=dict(color="#fb923c", dash="dash", width=1),
                  annotation_text="50% baseline", annotation_position="bottom right")
    layout = PLOTLY_LAYOUT.copy()
    yaxis = dict(layout.get("yaxis", {}))
    yaxis.update({"title": "Confidence", "range": [0, 1]})
    layout["yaxis"] = yaxis
    layout["title"] = "Classifier Confidence Trend"
    layout["height"] = 220
    fig.update_layout(**layout)
    return fig


# ── Badge helpers ──────────────────────────────────────────────────────────────

def _badge(cls: str, text: str) -> str:
    return f'<span class="badge badge-{cls.lower()}">{text}</span>'


# ── Main render ────────────────────────────────────────────────────────────────

def render(snap: SystemSnapshot, prediction: dict, anomaly: dict,
           workload: dict, pred_ok: bool, if_ok: bool, cls_ok: bool,
           predictor):
    df_hist = pd.DataFrame(st.session_state.history)

    # ── Header ──────────────────────────────────────────────────────────────────
    col_title, col_stats = st.columns([3, 1])
    with col_title:
        st.markdown("## ⚡ SIT-Py — Intelligence Dashboard")
        st.caption(
            f"Cycle #{st.session_state.cycle} · {snap.timestamp} · "
            f"{platform.system()} · "
            f"{'🟢 IF model' if if_ok else '🟡 Stat bridge'}"
        )
    with col_stats:
        if not df_hist.empty:
            n_alerts = int(df_hist["is_anomaly"].sum())
            alert_pct = 100 * n_alerts / max(len(df_hist), 1)
            st.metric("Session anomaly rate", f"{alert_pct:.1f}%",
                      delta=None if n_alerts == 0 else f"{n_alerts} alerts")

    st.divider()

    # ── Row 1: Gauge Cards ───────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Live Metrics</p>', unsafe_allow_html=True)
    g1, g2, g3, g4, g5 = st.columns(5)

    with g1:
        st.plotly_chart(_gauge_chart("CPU Usage", snap.cpu_pct), use_container_width=True)
    with g2:
        st.plotly_chart(_gauge_chart("RAM Usage", snap.ram_pct), use_container_width=True)
    with g3:
        st.plotly_chart(_gauge_chart("Disk Usage", snap.disk_pct, (60, 80, 95)),
                        use_container_width=True)
    with g4:
        st.metric("Bandwidth", f"{snap.bandwidth_kbps:.0f} kbps",
                  delta=f"↑{snap.bw_spike:.1f}× avg" if snap.bw_spike > 2 else None)
        st.metric("Packet Rate", f"{snap.packet_rate_pps:.0f} pps")
    with g5:
        st.metric("Processes", snap.process_count)
        st.metric("Δ CPU", f"{snap.cpu_delta:+.1f}%")
        st.metric("CPU σ", f"{snap.cpu_rolling_std:.2f}")

    st.divider()

    # ── Row 2: ML Status ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">ML Intelligence</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("**🚨 Anomaly Detector**")
        a_label = anomaly["label"]
        st.markdown(_badge(a_label, a_label), unsafe_allow_html=True)
        st.caption(f"Score: {anomaly['anomaly_score']:.4f}  "
                   f"| Source: {anomaly.get('source', 'if')}")
        if not if_ok:
            st.markdown(
                '<div class="bridge-info">⚠ Statistical bridge active — '
                'run <code>Core/anomaly.py</code> after collecting live data '
                f'({RETRAIN_ROW_THRESHOLD} rows needed)</div>',
                unsafe_allow_html=True
            )
        z_scores = anomaly.get("z_scores", {})
        if z_scores:
            for metric, z in z_scores.items():
                st.progress(min(abs(z) / 5.0, 1.0), text=f"{metric} z={z:+.2f}")

    with s2:
        st.markdown("**⚡ Workload Classifier**")
        wl = workload["workload"]
        st.markdown(_badge(wl, wl), unsafe_allow_html=True)
        st.caption(f"Confidence: {workload['confidence']*100:.1f}%")
        proba = workload.get("probabilities", {})
        if proba:
            for label, prob in sorted(proba.items(), key=lambda x: -x[1]):
                color = {"Critical":"#f87171","High":"#fb923c",
                         "Medium":"#60a5fa","Low":"#4ade80"}.get(label, "#94a3b8")
                st.markdown(f"<span style='color:{color};font-size:12px'>"
                            f"{label}: {prob*100:.1f}%</span>",
                            unsafe_allow_html=True)
                st.progress(prob)
        if not cls_ok:
            st.warning("Classifier not loaded. Run `Core/classifier.py`")

    with s3:
        st.markdown("**📈 Resource Predictor**")
        if prediction:
            err_cpu = abs(snap.cpu_pct - prediction.get("cpu_pct", snap.cpu_pct))
            err_ram = abs(snap.ram_pct - prediction.get("ram_pct", snap.ram_pct))
            c_a, c_b = st.columns(2)
            with c_a:
                st.metric("Next CPU",  f"{prediction.get('cpu_pct', 0):.1f}%",
                          delta=f"Δ {err_cpu:.1f}%" if err_cpu > 0 else None,
                          delta_color="off")
                st.metric("Next RAM",  f"{prediction.get('ram_pct', 0):.1f}%",
                          delta=f"Δ {err_ram:.1f}%" if err_ram > 0 else None,
                          delta_color="off")
            with c_b:
                st.metric("Next Disk", f"{prediction.get('disk_pct', 0):.1f}%")
        else:
            warmup = st.session_state.eng.history_len()
            st.info(f"Warming up... ({warmup}/15 cycles)")
            st.progress(warmup / 15)
        if not pred_ok:
            st.warning("Predictor not loaded. Run `Core/predictor.py`")

        # Feature importances mini-panel
        if pred_ok and predictor and predictor.is_ready:
            with st.expander("🔍 Feature Importances", expanded=False):
                imps = predictor.feature_importances()
                for feat, val in list(imps.items())[:6]:
                    st.text(f"{feat:22s} {val:.4f}")
                    st.progress(float(val))

    st.divider()

    # ── Row 3: Charts ─────────────────────────────────────────────────────────────
    if len(df_hist) >= 3:
        st.markdown('<p class="section-title">Telemetry & Analysis</p>',
                    unsafe_allow_html=True)

        ch1, ch2 = st.columns([3, 2])
        with ch1:
            st.plotly_chart(_timeseries_chart(df_hist), use_container_width=True)
        with ch2:
            st.plotly_chart(_workload_donut(df_hist), use_container_width=True)

        ch3, ch4 = st.columns(2)
        with ch3:
            st.plotly_chart(_anomaly_timeline(df_hist), use_container_width=True)
        with ch4:
            st.plotly_chart(_confidence_trend(df_hist), use_container_width=True)

        st.divider()

    # ── Row 4: Event Log ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Event Log</p>', unsafe_allow_html=True)
    if df_hist.empty:
        st.caption("No data yet — starting collection...")
    else:
        display_cols = ["timestamp", "cpu", "ram", "disk",
                        "anomaly", "workload", "wl_conf", "a_score"]
        available = [c for c in display_cols if c in df_hist.columns]
        log_df = df_hist[available].copy().iloc[::-1]
        log_df.columns = [c.replace("_", " ").title() for c in log_df.columns]
        if "Wl Conf" in log_df.columns:
            log_df["Wl Conf"] = log_df["Wl Conf"].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
            )
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    # ── Retrain CTA ───────────────────────────────────────────────────────────────
    st.divider()
    live_rows = 0
    if os.path.exists(LIVE_CSV):
        try:
            live_rows = len(pd.read_csv(LIVE_CSV))
        except Exception:
            live_rows = 0

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        st.markdown(f"**Live data collected:** {live_rows:,} rows  "
                    f"(need {RETRAIN_ROW_THRESHOLD:,} to retrain)")
        st.progress(min(live_rows / RETRAIN_ROW_THRESHOLD, 1.0))
    with rc2:
        retrain_ready = live_rows >= RETRAIN_ROW_THRESHOLD
        if st.button("🔄 Trigger Retrain", disabled=not retrain_ready,
                     help="Runs pipeline/retrain.py with collected live data"):
            with st.spinner("Retraining all models..."):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "pipeline/retrain.py", "--force"],
                    cwd=ROOT_DIR, capture_output=True, text=True
                )
                if result.returncode == 0:
                    st.success("Retrain complete! Restart the dashboard to load new models.")
                else:
                    st.error(f"Retrain failed:\n{result.stderr}")


# ── Main ──────────────────────────────────────────────────────────────────────

predictor, detector, classifier, pred_ok, if_ok, cls_ok = load_models()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    refresh_rate = st.slider("Refresh rate (s)", 1, 10, 2)
    running      = st.checkbox("▶ Live monitoring", value=True)

    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Cycles run",      st.session_state.cycle)
    st.metric("History points",  len(st.session_state.history))

    if st.session_state.history:
        df_s = pd.DataFrame(st.session_state.history)
        n_alerts = int(df_s["is_anomaly"].sum())
        st.metric("Alerts detected", n_alerts)
        if "wl_conf" in df_s.columns:
            avg_conf = df_s["wl_conf"].mean()
            st.metric("Avg confidence", f"{avg_conf*100:.1f}%")

    st.divider()
    st.markdown("### 🤖 Model Status")
    st.markdown(f"{'✅' if pred_ok else '❌'} Predictor (RF)")
    st.markdown(f"{'✅' if if_ok else '🟡'} Anomaly ({'IF' if if_ok else 'Stat bridge'})")
    st.markdown(f"{'✅' if cls_ok else '❌'} Classifier (RF)")

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.session_state.cycle   = 0
        st.session_state.eng.reset()
        st.rerun()

if running:
    snap, prediction, anomaly, workload = run_cycle(
        predictor, detector, classifier, pred_ok, if_ok, cls_ok
    )
    render(snap, prediction, anomaly, workload,
           pred_ok, if_ok, cls_ok, predictor)
    time.sleep(refresh_rate)
    st.rerun()
else:
    if st.session_state.history:
        last = st.session_state.history[-1]
        snap_dummy = SystemSnapshot(
            timestamp        = last["timestamp"],
            cpu_pct          = last["cpu"],
            ram_pct          = last["ram"],
            disk_pct         = last["disk"],
            bandwidth_kbps   = last["bandwidth"],
            process_count    = int(last.get("process_count", 0)),
            cpu_delta        = 0.0,
            cpu_rolling_std  = 0.0,
            bw_spike         = 1.0,
        )
        render(
            snap_dummy,
            {"cpu_pct": last.get("pred_cpu"), "ram_pct": last.get("pred_ram"),
             "disk_pct": last.get("pred_disk")},
            {"label": last["anomaly"], "anomaly_score": last["a_score"],
             "is_anomaly": last["is_anomaly"],
             "source": last.get("a_source", "")},
            {"workload": last["workload"], "confidence": last["wl_conf"],
             "probabilities": {}},
            pred_ok, if_ok, cls_ok, predictor,
        )
    else:
        st.info("▶ Press **Live monitoring** in the sidebar to start.")
        x=df["timestamp"],
        y=df["a_score"].abs(),
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        name="Anomaly score",
    ))
    layout = PLOTLY_LAYOUT.copy()
    yaxis = dict(layout.get("yaxis", {}))
    yaxis.update({"title": "Anomaly Score (abs)"})
    layout["yaxis"] = yaxis
    layout["title"] = "Anomaly Timeline (red = Alert)"
    layout["height"] = 220
    fig.update_layout(go.Layout(**layout))
    return fig


def _workload_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of workload distribution over session."""
    counts = df["workload"].value_counts().reset_index()
    counts.columns = ["workload", "count"]
    color_map = {
        "Critical": "#f87171", "High": "#fb923c",
        "Medium": "#60a5fa", "Low": "#4ade80", "N/A": "#475569",
    }
    fig = go.Figure(go.Pie(
        labels=counts["workload"],
        values=counts["count"],
        hole=0.55,
        marker_colors=[color_map.get(w, "#94a3b8") for w in counts["workload"]],
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    layout = PLOTLY_LAYOUT.copy()
    layout["title"] = "Workload Distribution"
    layout["height"] = 260
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["margin"] = dict(l=0, r=0, t=30, b=0)
    layout["showlegend"] = False
    layout["font"] = dict(color="#94a3b8")
    fig.update_layout(go.Layout(**layout))
    return fig


def _confidence_trend(df: pd.DataFrame) -> go.Figure:
    """Classifier confidence over time — dropping confidence = model drift."""
    fig = go.Figure(go.Scatter(
        x=df["timestamp"], y=df["wl_conf"],
        fill="tozeroy", fillcolor="rgba(147,197,253,0.08)",
        line=dict(color="#93c5fd", width=2),
        name="Confidence",
    ))
    fig.add_hline(y=0.5, line=dict(color="#fb923c", dash="dash", width=1),
                  annotation_text="50% baseline", annotation_position="bottom right")
    layout = PLOTLY_LAYOUT.copy()
    yaxis = dict(layout.get("yaxis", {}))
    yaxis.update({"title": "Confidence", "range": [0, 1]})
    layout["yaxis"] = yaxis
    layout["title"] = "Classifier Confidence Trend"
    layout["height"] = 220
    fig.update_layout(go.Layout(**layout))
    return fig


# ── Badge helpers ──────────────────────────────────────────────────────────────

def _badge(cls: str, text: str) -> str:
    return f'<span class="badge badge-{cls.lower()}">{text}</span>'


# ── Main render ────────────────────────────────────────────────────────────────

def render(snap: SystemSnapshot, prediction: dict, anomaly: dict,
           workload: dict, pred_ok: bool, if_ok: bool, cls_ok: bool,
           predictor):
    df_hist = pd.DataFrame(st.session_state.history)

    # ── Header ──────────────────────────────────────────────────────────────────
    col_title, col_stats = st.columns([3, 1])
    with col_title:
        st.markdown("## ⚡ SIT-Py — Intelligence Dashboard")
        st.caption(
            f"Cycle #{st.session_state.cycle} · {snap.timestamp} · "
            f"{platform.system()} · "
            f"{'🟢 IF model' if if_ok else '🟡 Stat bridge'}"
        )
    with col_stats:
        if not df_hist.empty:
            n_alerts = int(df_hist["is_anomaly"].sum())
            alert_pct = 100 * n_alerts / max(len(df_hist), 1)
            st.metric("Session anomaly rate", f"{alert_pct:.1f}%",
                      delta=None if n_alerts == 0 else f"{n_alerts} alerts")

    st.divider()

    # ── Row 1: Gauge Cards ───────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Live Metrics</p>', unsafe_allow_html=True)
    g1, g2, g3, g4, g5 = st.columns(5)

    with g1:
        st.plotly_chart(_gauge_chart("CPU Usage", snap.cpu_pct), use_container_width=True)
    with g2:
        st.plotly_chart(_gauge_chart("RAM Usage", snap.ram_pct), use_container_width=True)
    with g3:
        st.plotly_chart(_gauge_chart("Disk Usage", snap.disk_pct, (60, 80, 95)),
                        use_container_width=True)
    with g4:
        st.metric("Bandwidth", f"{snap.bandwidth_kbps:.0f} kbps",
                  delta=f"↑{snap.bw_spike:.1f}× avg" if snap.bw_spike > 2 else None)
        st.metric("Packet Rate", f"{snap.packet_rate_pps:.0f} pps")
    with g5:
        st.metric("Processes", snap.process_count)
        st.metric("Δ CPU", f"{snap.cpu_delta:+.1f}%")
        st.metric("CPU σ", f"{snap.cpu_rolling_std:.2f}")

    st.divider()

    # ── Row 2: ML Status ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">ML Intelligence</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("**🚨 Anomaly Detector**")
        a_label = anomaly["label"]
        st.markdown(_badge(a_label, a_label), unsafe_allow_html=True)
        st.caption(f"Score: {anomaly['anomaly_score']:.4f}  "
                   f"| Source: {anomaly.get('source', 'if')}")
        if not if_ok:
            st.markdown(
                '<div class="bridge-info">⚠ Statistical bridge active — '
                'run <code>Core/anomaly.py</code> after collecting live data '
                f'({RETRAIN_ROW_THRESHOLD} rows needed)</div>',
                unsafe_allow_html=True
            )
        z_scores = anomaly.get("z_scores", {})
        if z_scores:
            for metric, z in z_scores.items():
                st.progress(min(abs(z) / 5.0, 1.0), text=f"{metric} z={z:+.2f}")

    with s2:
        st.markdown("**⚡ Workload Classifier**")
        wl = workload["workload"]
        st.markdown(_badge(wl, wl), unsafe_allow_html=True)
        st.caption(f"Confidence: {workload['confidence']*100:.1f}%")
        proba = workload.get("probabilities", {})
        if proba:
            for label, prob in sorted(proba.items(), key=lambda x: -x[1]):
                color = {"Critical":"#f87171","High":"#fb923c",
                         "Medium":"#60a5fa","Low":"#4ade80"}.get(label, "#94a3b8")
                st.markdown(f"<span style='color:{color};font-size:12px'>"
                            f"{label}: {prob*100:.1f}%</span>",
                            unsafe_allow_html=True)
                st.progress(prob)
        if not cls_ok:
            st.warning("Classifier not loaded. Run `Core/classifier.py`")

    with s3:
        st.markdown("**📈 Resource Predictor**")
        if prediction:
            err_cpu = abs(snap.cpu_pct - prediction.get("cpu_pct", snap.cpu_pct))
            err_ram = abs(snap.ram_pct - prediction.get("ram_pct", snap.ram_pct))
            c_a, c_b = st.columns(2)
            with c_a:
                st.metric("Next CPU",  f"{prediction.get('cpu_pct', 0):.1f}%",
                          delta=f"Δ {err_cpu:.1f}%" if err_cpu > 0 else None,
                          delta_color="off")
                st.metric("Next RAM",  f"{prediction.get('ram_pct', 0):.1f}%",
                          delta=f"Δ {err_ram:.1f}%" if err_ram > 0 else None,
                          delta_color="off")
            with c_b:
                st.metric("Next Disk", f"{prediction.get('disk_pct', 0):.1f}%")
        else:
            warmup = st.session_state.eng.history_len()
            st.info(f"Warming up... ({warmup}/15 cycles)")
            st.progress(warmup / 15)
        if not pred_ok:
            st.warning("Predictor not loaded. Run `Core/predictor.py`")

        # Feature importances mini-panel
        if pred_ok and predictor and predictor.is_ready:
            with st.expander("🔍 Feature Importances", expanded=False):
                imps = predictor.feature_importances()
                for feat, val in list(imps.items())[:6]:
                    st.text(f"{feat:22s} {val:.4f}")
                    st.progress(float(val))

    st.divider()

    # ── Row 3: Charts ─────────────────────────────────────────────────────────────
    if len(df_hist) >= 3:
        st.markdown('<p class="section-title">Telemetry & Analysis</p>',
                    unsafe_allow_html=True)

        ch1, ch2 = st.columns([3, 2])
        with ch1:
            st.plotly_chart(_timeseries_chart(df_hist), use_container_width=True)
        with ch2:
            st.plotly_chart(_workload_donut(df_hist), use_container_width=True)

        ch3, ch4 = st.columns(2)
        with ch3:
            st.plotly_chart(_anomaly_timeline(df_hist), use_container_width=True)
        with ch4:
            st.plotly_chart(_confidence_trend(df_hist), use_container_width=True)

        st.divider()

    # ── Row 4: Event Log ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Event Log</p>', unsafe_allow_html=True)
    if df_hist.empty:
        st.caption("No data yet — starting collection...")
    else:
        display_cols = ["timestamp", "cpu", "ram", "disk",
                        "anomaly", "workload", "wl_conf", "a_score"]
        available = [c for c in display_cols if c in df_hist.columns]
        log_df = df_hist[available].copy().iloc[::-1]
        log_df.columns = [c.replace("_", " ").title() for c in log_df.columns]
        if "Wl Conf" in log_df.columns:
            log_df["Wl Conf"] = log_df["Wl Conf"].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
            )
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    # ── Retrain CTA ───────────────────────────────────────────────────────────────
    st.divider()
    live_rows = 0
    if os.path.exists(LIVE_CSV):
        try:
            live_rows = len(pd.read_csv(LIVE_CSV))
        except Exception:
            live_rows = 0

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        st.markdown(f"**Live data collected:** {live_rows:,} rows  "
                    f"(need {RETRAIN_ROW_THRESHOLD:,} to retrain)")
        st.progress(min(live_rows / RETRAIN_ROW_THRESHOLD, 1.0))
    with rc2:
        retrain_ready = live_rows >= RETRAIN_ROW_THRESHOLD
        if st.button("🔄 Trigger Retrain", disabled=not retrain_ready,
                     help="Runs pipeline/retrain.py with collected live data"):
            with st.spinner("Retraining all models..."):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "pipeline/retrain.py", "--force"],
                    cwd=ROOT_DIR, capture_output=True, text=True
                )
                if result.returncode == 0:
                    st.success("Retrain complete! Restart the dashboard to load new models.")
                else:
                    st.error(f"Retrain failed:\n{result.stderr}")


# ── Main ──────────────────────────────────────────────────────────────────────

predictor, detector, classifier, pred_ok, if_ok, cls_ok = load_models()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    refresh_rate = st.slider("Refresh rate (s)", 1, 10, 2)
    running      = st.checkbox("▶ Live monitoring", value=True)

    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Cycles run",      st.session_state.cycle)
    st.metric("History points",  len(st.session_state.history))

    if st.session_state.history:
        df_s = pd.DataFrame(st.session_state.history)
        n_alerts = int(df_s["is_anomaly"].sum())
        st.metric("Alerts detected", n_alerts)
        if "wl_conf" in df_s.columns:
            avg_conf = df_s["wl_conf"].mean()
            st.metric("Avg confidence", f"{avg_conf*100:.1f}%")

    st.divider()
    st.markdown("### 🤖 Model Status")
    st.markdown(f"{'✅' if pred_ok else '❌'} Predictor (RF)")
    st.markdown(f"{'✅' if if_ok else '🟡'} Anomaly ({'IF' if if_ok else 'Stat bridge'})")
    st.markdown(f"{'✅' if cls_ok else '❌'} Classifier (RF)")

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.session_state.cycle   = 0
        st.session_state.eng.reset()
        st.rerun()

if running:
    snap, prediction, anomaly, workload = run_cycle(
        predictor, detector, classifier, pred_ok, if_ok, cls_ok
    )
    render(snap, prediction, anomaly, workload,
           pred_ok, if_ok, cls_ok, predictor)
    time.sleep(refresh_rate)
    st.rerun()
else:
    if st.session_state.history:
        last = st.session_state.history[-1]
        snap_dummy = SystemSnapshot(
            timestamp        = last["timestamp"],
            cpu_pct          = last["cpu"],
            ram_pct          = last["ram"],
            disk_pct         = last["disk"],
            bandwidth_kbps   = last["bandwidth"],
            process_count    = int(last.get("process_count", 0)),
            cpu_delta        = 0.0,
            cpu_rolling_std  = 0.0,
            bw_spike         = 1.0,
        )
        render(
            snap_dummy,
            {"cpu_pct": last.get("pred_cpu"), "ram_pct": last.get("pred_ram"),
             "disk_pct": last.get("pred_disk")},
            {"label": last["anomaly"], "anomaly_score": last["a_score"],
             "is_anomaly": last["is_anomaly"],
             "source": last.get("a_source", "")},
            {"workload": last["workload"], "confidence": last["wl_conf"],
             "probabilities": {}},
            pred_ok, if_ok, cls_ok, predictor,
        )
    else:
        st.info("▶ Press **Live monitoring** in the sidebar to start.")