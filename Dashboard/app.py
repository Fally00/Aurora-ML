import os
import sys
import time
import platform
import psutil
import pandas as pd
import streamlit as st
from collections import deque
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from Core.predictor  import ResourcePredictor, WINDOW_SIZE
from Core.anomly      import AnomalyDetector
from Core.classifier import WorkloadClassifier
from Data.collect    import take_snapshot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SIT-Py | Resource Monitor",
    page_icon  = "🌟",
    layout     = "wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* dark background */
    .stApp { background-color: #0e1117; }

    /* metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1d27;
        border: 1px solid #2d3040;
        border-radius: 10px;
        padding: 16px;
    }

    /* status badge */
    .badge-normal {
        background-color: #1a3a2a;
        color: #4caf88;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
    }
    .badge-attack {
        background-color: #3a1a1a;
        color: #f44336;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
    }
    .badge-critical { background-color: #3a1a1a; color: #f44336; padding: 4px 14px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .badge-high     { background-color: #3a2d1a; color: #ff9800; padding: 4px 14px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .badge-medium   { background-color: #1a2a3a; color: #42a5f5; padding: 4px 14px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .badge-low      { background-color: #1a3a2a; color: #4caf50; padding: 4px 14px; border-radius: 20px; font-weight: 700; display: inline-block; }

    /* section headers */
    .section-title {
        color: #7c8db5;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* log table */
    .log-row-attack { color: #f44336; }
    .log-row-normal { color: #4caf88; }

    /* hide streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = os.path.join(ROOT_DIR, "models")
    predictor  = ResourcePredictor(models_dir)
    detector   = AnomalyDetector(models_dir)
    classifier = WorkloadClassifier(models_dir)
    predictor.load()
    detector.load()
    classifier.load()
    return predictor, detector, classifier

# ── Session state init ─────────────────────────────────────────────────────────
if "window_buffer" not in st.session_state:
    st.session_state.window_buffer = deque(maxlen=WINDOW_SIZE)

if "prev_net" not in st.session_state:
    st.session_state.prev_net = psutil.net_io_counters()

if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts for charts

if "cycle" not in st.session_state:
    st.session_state.cycle = 0

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_anomaly_snap(snap):
    return {
        "CPU_Usage": snap.get("CPUUsage", 0.0),
        "Memory_Usage": snap.get("RAMUsage", 0.0),
        "Bandwidth": snap.get("Bandwidth", 0.0),
        "Packet_Rate": snap.get("Packet_Rate", 0.0),
        "Failed_Logins": 0,
        "Malware_Alerts": 0,
        "Intrusion_Alerts": 0,
        "Task_Priority": snap.get("Task_Priority", "Medium"),
        "Traffic_Type": "HTTP",
    }

def build_classifier_snap(snap):
    return {
        "CPU_Usage": snap.get("CPUUsage", 0.0),
        "Memory_Usage": snap.get("RAMUsage", 0.0),
        "Bandwidth": snap.get("Bandwidth", 0.0),
        "Packet_Rate": snap.get("Packet_Rate", 0.0),
        "Failed_Logins": 0,
        "Malware_Alerts": 0,
        "Intrusion_Alerts": 0,
        "Traffic_Type": "HTTP",
    }

def build_predictor_window(buf):
    return [
        {
            "CPUUsage": s.get("CPUUsage", 0.0),
            "RAMUsage": s.get("RAMUsage", 0.0),
            "Temperature": s.get("Temperature", 0.0),
            "DiskUsage": s.get("DiskUsage", 0.0),
            "FanSpeed": max(s.get("FanSpeed", 0), 0),
        }
        for s in buf
    ]

def workload_badge(w):
    cls = f"badge-{w.lower()}"
    return f'<span class="{cls}">{w}</span>'

def anomaly_badge(label):
    cls = "badge-attack" if label == "Attack" else "badge-normal"
    return f'<span class="{cls}">{label}</span>'

# ── Run one pipeline cycle ─────────────────────────────────────────────────────
def run_cycle(predictor, detector, classifier):
    snap, st.session_state.prev_net = take_snapshot(st.session_state.prev_net)
    st.session_state.cycle += 1

    anomaly    = detector.predict(build_anomaly_snap(snap))
    workload   = classifier.predict(build_classifier_snap(snap))

    st.session_state.window_buffer.append(snap)
    if len(st.session_state.window_buffer) == WINDOW_SIZE:
        prediction = predictor.predict(build_predictor_window(st.session_state.window_buffer))
    else:
        prediction = {}

    # save to history (keep last 60 points)
    record = {
        "timestamp" : snap.get("timestamp", ""),
        "CPU"       : snap.get("CPUUsage", 0.0),
        "RAM"       : snap.get("RAMUsage", 0.0),
        "Disk"      : snap.get("DiskUsage", 0.0),
        "Bandwidth" : snap.get("Bandwidth", 0.0),
        "pred_CPU"  : prediction.get("CPUUsage"),
        "pred_RAM"  : prediction.get("RAMUsage"),
        "pred_Disk" : prediction.get("DiskUsage"),
        "anomaly"   : anomaly["label"],
        "workload"  : workload["workload"],
        "wl_conf"   : workload["confidence"],
        "a_score"   : anomaly["anomaly_score"],
    }
    st.session_state.history.append(record)
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

    return snap, prediction, anomaly, workload

# ── Dashboard Layout ───────────────────────────────────────────────────────────
def render(snap, prediction, anomaly, workload):
    df_hist = pd.DataFrame(st.session_state.history)

    # ── Header ──
    st.markdown("##  SIT-Py — Intelligent Resource Monitor")
    st.caption(f"Cycle #{st.session_state.cycle} · {snap.get('timestamp', '')} · Platform: {platform.system()}")
    st.divider()

    # ── Row 1: Live Metrics ──
    st.markdown('<p class="section-title">Live Metrics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CPU Usage",    f"{snap.get('CPUUsage', 0):.1f}%")
    c2.metric("RAM Usage",    f"{snap.get('RAMUsage', 0):.1f}%")
    c3.metric("Disk Usage",   f"{snap.get('DiskUsage', 0):.1f}%")
    c4.metric("Bandwidth",    f"{snap.get('Bandwidth', 0):.1f} kbps")
    c5.metric("Packet Rate",  f"{snap.get('Packet_Rate', 0):.1f} pps")

    st.divider()

    # ── Row 2: Status Cards ──
    st.markdown('<p class="section-title">ML Status</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("**🚨 Anomaly Detector**")
        st.markdown(anomaly_badge(anomaly["label"]), unsafe_allow_html=True)
        st.caption(f"Score: {anomaly['anomaly_score']}")

    with s2:
        st.markdown("**⚡ Workload Classifier**")
        st.markdown(workload_badge(workload["workload"]), unsafe_allow_html=True)
        st.caption(f"Confidence: {workload['confidence']*100:.1f}%")

        # probability mini-breakdown
        proba = workload.get("probabilities", {})
        if proba:
            for label, prob in sorted(proba.items(), key=lambda x: -x[1]):
                # show label then a small progress bar (avoid deprecated `text` param)
                st.write(f"{label}: {prob*100:.1f}%")
                st.progress(prob)

    with s3:
        st.markdown("**📈 Resource Predictor**")
        if prediction:
            st.markdown(f"Next CPU &nbsp;&nbsp;→ **{prediction.get('CPUUsage', 'N/A')}%**", unsafe_allow_html=True)
            st.markdown(f"Next RAM &nbsp;&nbsp;→ **{prediction.get('RAMUsage', 'N/A')}%**", unsafe_allow_html=True)
            st.markdown(f"Next Disk → **{prediction.get('DiskUsage', 'N/A')}%**", unsafe_allow_html=True)
        else:
            st.info(f"Warming up... ({len(st.session_state.window_buffer)}/{WINDOW_SIZE} cycles)")

    st.divider()

    # ── Row 3: Charts ──
    if len(df_hist) >= 2:
        st.markdown('<p class="section-title">History (last 60 cycles)</p>', unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("**CPU & RAM over time**")
            chart_data = df_hist[["timestamp", "CPU", "RAM"]].set_index("timestamp")
            st.line_chart(chart_data, color=["#42a5f5", "#4caf88"])

        with ch2:
            st.markdown("**Bandwidth over time**")
            bw_data = df_hist[["timestamp", "Bandwidth"]].set_index("timestamp")
            st.line_chart(bw_data, color=["#ff9800"])

        st.divider()

    # ── Row 4: Event Log ──
    st.markdown('<p class="section-title">Event Log</p>', unsafe_allow_html=True)
    if df_hist.empty:
        st.caption("No data yet...")
    else:
        log_display = df_hist[["timestamp", "CPU", "RAM", "Disk", "anomaly", "workload", "wl_conf"]].copy()
        log_display.columns = ["Time", "CPU%", "RAM%", "Disk%", "Anomaly", "Workload", "Confidence"]
        log_display["Confidence"] = log_display["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        log_display = log_display.iloc[::-1]   # newest first
        st.dataframe(log_display, width="stretch", hide_index=True)

# ── Main ───────────────────────────────────────────────────────────────────────
predictor, detector, classifier = load_models()

# sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    refresh_rate = st.slider("Refresh rate (seconds)", 1, 10, 2)
    # `st.toggle` may not exist in all Streamlit versions — use checkbox for compatibility
    running      = st.checkbox("Live monitoring", value=True)
    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Cycles run",    st.session_state.cycle)
    st.metric("Events logged", len(st.session_state.history))
    attacks = sum(1 for r in st.session_state.history if r["anomaly"] == "Attack")
    st.metric("Anomalies detected", attacks)

if running:
    snap, prediction, anomaly, workload = run_cycle(predictor, detector, classifier)
    render(snap, prediction, anomaly, workload)
    time.sleep(refresh_rate)
    st.rerun()
else:
    if st.session_state.history:
        last = st.session_state.history[-1]
        snap_dummy = {
            "CPUUsage": last["CPU"], "RAMUsage": last["RAM"],
            "DiskUsage": last["Disk"], "Bandwidth": last["Bandwidth"],
            "Packet_Rate": 0, "Task_Priority": "Medium",
            "timestamp": last["timestamp"]
        }
        render(
            snap_dummy,
            {"CPUUsage": last["pred_CPU"], "RAMUsage": last["pred_RAM"], "DiskUsage": last["pred_Disk"]},
            {"label": last["anomaly"], "anomaly_score": last["a_score"], "is_anomaly": last["anomaly"] == "Attack"},
            {"workload": last["workload"], "confidence": last["wl_conf"], "probabilities": {}}
        )
    else:
        st.info("Press the toggle in the sidebar to start monitoring.")