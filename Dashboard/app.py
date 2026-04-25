"""
Streamlit dashboard for SIT-Py.

This version is aligned with the updated per-agent datasets, runtime schema,
and retraining pipeline.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import ANOMALY_Z_THRESHOLD, LIVE_CSV, MODELS_DIR, RETRAIN_ROW_THRESHOLD, WINDOW_SIZE
from Core.anomaly import AnomalyDetector
from Core.classifier import WorkloadClassifier
from Core.feature_engineering import FeatureEngineer
from Core.predictor import ResourcePredictor
from Core.schema import SystemSnapshot
from Data.collect import take_snapshot


st.set_page_config(
    page_title="SIT-Py Dashboard",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(180deg, #07111f 0%, #0d1726 100%);
    color: #d8e2f1;
}
.card {
    border: 1px solid #1e3552;
    border-radius: 14px;
    padding: 16px 18px;
    background: rgba(10, 20, 34, 0.85);
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
}
.badge-normal { background: #10291b; color: #67e8a1; }
.badge-alert { background: #341012; color: #fda4af; }
.badge-low { background: #10291b; color: #67e8a1; }
.badge-medium { background: #11263d; color: #93c5fd; }
.badge-high { background: #3b250d; color: #fdba74; }
.badge-critical { background: #341012; color: #fda4af; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    predictor = ResourcePredictor(MODELS_DIR)
    detector = AnomalyDetector(MODELS_DIR)
    classifier = WorkloadClassifier(MODELS_DIR)

    pred_ok = True
    try:
        predictor.load()
    except FileNotFoundError:
        pred_ok = False

    anomaly_ok = detector.load()

    cls_ok = True
    try:
        classifier.load()
    except FileNotFoundError:
        cls_ok = False

    return predictor, detector, classifier, pred_ok, anomaly_ok, cls_ok


def _init_session() -> None:
    if "engineer" not in st.session_state:
        st.session_state.engineer = FeatureEngineer(window_size=WINDOW_SIZE)
    if "prev_net" not in st.session_state:
        st.session_state.prev_net = psutil.net_io_counters()
    if "last_t" not in st.session_state:
        st.session_state.last_t = time.time()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "cycle" not in st.session_state:
        st.session_state.cycle = 0


def _badge(kind: str, text: str) -> str:
    return f'<span class="badge badge-{kind.lower()}">{text}</span>'


def _history_frame() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history)


def _safe_csv_len(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return len(pd.read_csv(path, engine="python", on_bad_lines="skip"))


def run_cycle(predictor, detector, classifier, pred_ok: bool, anomaly_ok: bool, cls_ok: bool):
    now = time.time()
    elapsed = now - st.session_state.last_t
    st.session_state.last_t = now
    st.session_state.cycle += 1

    snap, st.session_state.prev_net = take_snapshot(st.session_state.prev_net, elapsed)
    snap = st.session_state.engineer.enrich(snap)

    if anomaly_ok:
        anomaly = detector.predict(snap.to_anomaly_input())
    else:
        anomaly = st.session_state.engineer.statistical_anomaly(
            snap,
            z_threshold=ANOMALY_Z_THRESHOLD,
        )

    if cls_ok:
        workload = classifier.predict(snap.to_classifier_input())
    else:
        workload = {"workload": "N/A", "confidence": 0.0, "probabilities": {}}

    prediction = {}
    if pred_ok and st.session_state.engineer.is_ready():
        prediction = predictor.predict(snap.to_predictor_input())

    st.session_state.history.append(
        {
            "timestamp": snap.timestamp,
            "cpu": snap.cpu_pct,
            "ram": snap.ram_pct,
            "disk": snap.disk_pct,
            "bandwidth": snap.bandwidth_kbps,
            "packet_rate": snap.packet_rate_pps,
            "process_count": snap.process_count,
            "pred_cpu": prediction.get("cpu_pct"),
            "pred_ram": prediction.get("ram_pct"),
            "pred_disk": prediction.get("disk_pct"),
            "anomaly": anomaly["label"],
            "anomaly_score": anomaly["anomaly_score"],
            "is_anomaly": anomaly["is_anomaly"],
            "anomaly_source": anomaly.get("source", ""),
            "workload": workload["workload"],
            "wl_conf": workload["confidence"],
        }
    )
    st.session_state.history = st.session_state.history[-120:]
    return snap, prediction, anomaly, workload


def _telemetry_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["cpu"],
            name="CPU",
            line=dict(color="#93c5fd", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["ram"],
            name="RAM",
            line=dict(color="#86efac", width=2),
        )
    )

    if "pred_cpu" in df.columns and df["pred_cpu"].notna().any():
        subset = df[df["pred_cpu"].notna()]
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["pred_cpu"],
                name="Pred CPU",
                line=dict(color="#bfdbfe", width=1.5, dash="dot"),
            )
        )

    if "pred_ram" in df.columns and df["pred_ram"].notna().any():
        subset = df[df["pred_ram"].notna()]
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["pred_ram"],
                name="Pred RAM",
                line=dict(color="#bbf7d0", width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        title="CPU and RAM telemetry",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,16,28,0.75)",
        font=dict(color="#c7d2e3"),
        xaxis=dict(gridcolor="#1e3552"),
        yaxis=dict(gridcolor="#1e3552", range=[0, 100]),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _anomaly_chart(df: pd.DataFrame) -> go.Figure:
    colors = ["#fda4af" if value else "#67e8a1" for value in df["is_anomaly"]]
    fig = go.Figure(
        go.Bar(
            x=df["timestamp"],
            y=df["anomaly_score"],
            marker_color=colors,
            name="Anomaly score",
        )
    )
    fig.update_layout(
        title="Anomaly score timeline",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,16,28,0.75)",
        font=dict(color="#c7d2e3"),
        xaxis=dict(gridcolor="#1e3552"),
        yaxis=dict(gridcolor="#1e3552"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render(
    snap: SystemSnapshot,
    prediction: dict,
    anomaly: dict,
    workload: dict,
    pred_ok: bool,
    anomaly_ok: bool,
    cls_ok: bool,
) -> None:
    df = _history_frame()

    st.title("SIT-Py Dashboard")
    st.caption(
        f"Cycle #{st.session_state.cycle} | {snap.timestamp} | "
        f"{platform.system()} | "
        f"{'Supervised anomaly model' if anomaly_ok else 'Statistical bridge'}"
    )

    top_cols = st.columns(5)
    top_cols[0].metric("CPU", f"{snap.cpu_pct:.1f}%")
    top_cols[1].metric("RAM", f"{snap.ram_pct:.1f}%")
    top_cols[2].metric("Disk", f"{snap.disk_pct:.1f}%")
    top_cols[3].metric("Bandwidth", f"{snap.bandwidth_kbps:.1f} kbps")
    top_cols[4].metric("Processes", f"{snap.process_count}")

    ml_left, ml_mid, ml_right = st.columns(3)
    with ml_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Anomaly**")
        st.markdown(_badge(anomaly["label"], anomaly["label"]), unsafe_allow_html=True)
        st.caption(
            f"Score: {anomaly['anomaly_score']:.4f} | Source: {anomaly.get('source', '')}"
        )
        if not anomaly_ok:
            st.info("Bridge mode is active until a trained anomaly model is available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with ml_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Workload**")
        st.markdown(_badge(workload["workload"], workload["workload"]), unsafe_allow_html=True)
        st.caption(f"Confidence: {workload['confidence'] * 100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with ml_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Prediction**")
        if prediction:
            st.write(
                f"CPU {prediction.get('cpu_pct', 0):.1f}% | "
                f"RAM {prediction.get('ram_pct', 0):.1f}% | "
                f"Disk {prediction.get('disk_pct', 0):.1f}%"
            )
        else:
            st.write(f"Warming up ({st.session_state.engineer.history_len()}/{WINDOW_SIZE})")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    if len(df) >= 2:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(_telemetry_chart(df), use_container_width=True)
        with chart_right:
            st.plotly_chart(_anomaly_chart(df), use_container_width=True)

        summary_left, summary_right = st.columns(2)
        with summary_left:
            workload_counts = df["workload"].value_counts().rename_axis("workload").reset_index(name="count")
            st.subheader("Workload distribution")
            st.dataframe(workload_counts, use_container_width=True, hide_index=True)
        with summary_right:
            st.subheader("Recent events")
            event_cols = [
                "timestamp",
                "cpu",
                "ram",
                "disk",
                "anomaly",
                "workload",
                "wl_conf",
            ]
            event_df = df[event_cols].copy().iloc[::-1]
            event_df["wl_conf"] = event_df["wl_conf"].apply(lambda value: f"{value * 100:.1f}%")
            st.dataframe(event_df, use_container_width=True, hide_index=True)

    st.divider()
    live_rows = _safe_csv_len(LIVE_CSV)
    progress = min(live_rows / RETRAIN_ROW_THRESHOLD, 1.0) if RETRAIN_ROW_THRESHOLD else 0.0
    retrain_left, retrain_right = st.columns([3, 1])
    with retrain_left:
        st.write(
            f"Live rows collected: {live_rows:,} / {RETRAIN_ROW_THRESHOLD:,} needed for retraining"
        )
        st.progress(progress)
    with retrain_right:
        if st.button("Trigger Retrain"):
            with st.spinner("Retraining all models..."):
                result = subprocess.run(
                    [sys.executable, os.path.join("pipeline", "retrain.py"), "--force"],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                load_models.clear()
                st.success("Retrain complete. Reloading models on next rerun.")
            else:
                st.error(result.stderr or result.stdout or "Retrain failed.")

    if not pred_ok:
        st.warning("Predictor model not loaded. Train with `python Core/predictor.py`.")
    if not cls_ok:
        st.warning("Classifier model not loaded. Train with `python Core/classifier.py`.")
    if not anomaly_ok:
        st.warning("Anomaly model not loaded. Train with `python Core/anomaly.py`.")


_init_session()
predictor, detector, classifier, pred_ok, anomaly_ok, cls_ok = load_models()

with st.sidebar:
    st.header("Controls")
    refresh_rate = st.slider("Refresh rate (seconds)", 1, 10, 2)
    running = st.checkbox("Live monitoring", value=True)

    st.divider()
    st.subheader("Session")
    st.metric("Cycles", st.session_state.cycle)
    st.metric("History points", len(st.session_state.history))
    if st.session_state.history:
        df = _history_frame()
        st.metric("Alerts", int(df["is_anomaly"].sum()))
        st.metric("Avg confidence", f"{df['wl_conf'].mean() * 100:.1f}%")

    st.divider()
    st.subheader("Models")
    st.write(f"Predictor: {'Ready' if pred_ok else 'Missing'}")
    st.write(f"Anomaly: {'Ready' if anomaly_ok else 'Bridge mode'}")
    st.write(f"Classifier: {'Ready' if cls_ok else 'Missing'}")

    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.cycle = 0
        st.session_state.engineer.reset()
        st.rerun()


if running:
    snapshot, prediction, anomaly, workload = run_cycle(
        predictor,
        detector,
        classifier,
        pred_ok,
        anomaly_ok,
        cls_ok,
    )
    render(snapshot, prediction, anomaly, workload, pred_ok, anomaly_ok, cls_ok)
    time.sleep(refresh_rate)
    st.rerun()
else:
    if st.session_state.history:
        last = st.session_state.history[-1]
        snapshot = SystemSnapshot(
            timestamp=last["timestamp"],
            cpu_pct=float(last["cpu"]),
            ram_pct=float(last["ram"]),
            disk_pct=float(last["disk"]),
            bandwidth_kbps=float(last["bandwidth"]),
            packet_rate_pps=float(last["packet_rate"]),
            process_count=int(last["process_count"]),
        )
        render(
            snapshot,
            {
                "cpu_pct": last.get("pred_cpu"),
                "ram_pct": last.get("pred_ram"),
                "disk_pct": last.get("pred_disk"),
            },
            {
                "label": last["anomaly"],
                "anomaly_score": float(last["anomaly_score"]),
                "is_anomaly": bool(last["is_anomaly"]),
                "source": last.get("anomaly_source", ""),
            },
            {
                "workload": last["workload"],
                "confidence": float(last["wl_conf"]),
                "probabilities": {},
            },
            pred_ok,
            anomaly_ok,
            cls_ok,
        )
    else:
        st.info("Enable live monitoring in the sidebar to start collecting data.")
