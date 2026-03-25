"""
EdgeHAR Real-Time Monitoring Dashboard.

A Streamlit-based dashboard for real-time human activity recognition
monitoring. Supports two modes:
    - Live Simulation: Auto-generates sensor data and runs ONNX inference
    - Upload CSV: Upload sensor data files for batch analysis

Features:
    - Real-time activity classification with confidence display
    - Live sensor signal visualization (6 channels)
    - Prediction history and confusion matrix
    - Serial port connection for ESP32 (live mode)

Usage:
    streamlit run dashboard/app.py
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    "project_root": Path(__file__).resolve().parent.parent,
    "onnx_model_path": Path(__file__).resolve().parent.parent / "models" / "har_model.onnx",
    "class_names": [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
        "SITTING", "STANDING", "LAYING",
    ],
    "class_emojis": {
        "WALKING": "🚶",
        "WALKING_UPSTAIRS": "🪜⬆️",
        "WALKING_DOWNSTAIRS": "🪜⬇️",
        "SITTING": "🪑",
        "STANDING": "🧍",
        "LAYING": "🛌",
    },
    "num_channels": 6,
    "sequence_length": 128,
    "channel_names": [
        "Accel X", "Accel Y", "Accel Z",
        "Gyro X", "Gyro Y", "Gyro Z",
    ],
    "simulation_interval": 0.5,  # seconds between predictions in sim mode
}

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeHAR Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session State Initialization ────────────────────────────────────────────
def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "predictions": [],           # List of {timestamp, activity, confidence, probs}
        "sensor_buffer": np.zeros((CONFIG["num_channels"], CONFIG["sequence_length"])),
        "is_connected": False,
        "serial_port": None,
        "ground_truth": [],          # Simulated ground truth for confusion matrix
        "model_loaded": False,
        "ort_session": None,
        "simulation_running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_onnx_model() -> Optional[object]:
    """Load the ONNX model for inference.

    Returns:
        ONNX Runtime InferenceSession, or None if model not found.
    """
    if st.session_state.ort_session is not None:
        return st.session_state.ort_session

    model_path = CONFIG["onnx_model_path"]
    if not model_path.exists():
        return None

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        st.session_state.ort_session = session
        st.session_state.model_loaded = True
        return session
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return None


def run_inference(session: object, sensor_data: np.ndarray) -> tuple:
    """Run inference on sensor data using ONNX Runtime.

    Args:
        session: ONNX Runtime InferenceSession.
        sensor_data: Numpy array of shape (6, 128).

    Returns:
        Tuple of (predicted_class_idx, confidence, all_probabilities).
    """
    # Prepare input: add batch dimension
    input_data = sensor_data.astype(np.float32).reshape(1, CONFIG["num_channels"], CONFIG["sequence_length"])

    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_data})

    # Softmax to get probabilities
    logits = output[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    return predicted_class, confidence, probs


def generate_simulated_data(activity_idx: int = None) -> tuple:
    """Generate synthetic sensor data for simulation mode.

    Args:
        activity_idx: Optional forced activity index. If None, random.

    Returns:
        Tuple of (sensor_data, ground_truth_label).
    """
    if activity_idx is None:
        activity_idx = np.random.randint(0, 6)

    t = np.linspace(0, 1.024, CONFIG["sequence_length"])  # ~1 second of data
    noise_scale = 0.02
    data = np.zeros((CONFIG["num_channels"], CONFIG["sequence_length"]))

    if activity_idx == 0:  # WALKING
        data[0] = 0.15 * np.sin(2 * np.pi * 2 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[1] = 0.10 * np.sin(2 * np.pi * 2 * t + np.pi / 3) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[2] = 0.98 + 0.20 * np.sin(2 * np.pi * 4 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[3] = 0.05 * np.sin(2 * np.pi * 2 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[4] = 0.08 * np.cos(2 * np.pi * 2 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[5] = 0.03 * np.sin(2 * np.pi * 1 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
    elif activity_idx == 1:  # WALKING_UPSTAIRS
        data[0] = 0.20 * np.sin(2 * np.pi * 2.5 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[1] = 0.12 * np.sin(2 * np.pi * 2.5 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[2] = 0.95 + 0.35 * np.sin(2 * np.pi * 5 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[3] = 0.10 * np.sin(2 * np.pi * 2.5 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[4] = 0.15 * np.cos(2 * np.pi * 2.5 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[5] = 0.04 * np.sin(2 * np.pi * 1.25 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
    elif activity_idx == 2:  # WALKING_DOWNSTAIRS
        data[0] = 0.18 * np.sin(2 * np.pi * 1.8 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[1] = 0.14 * np.sin(2 * np.pi * 1.8 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[2] = 1.02 - 0.30 * np.abs(np.sin(2 * np.pi * 3.6 * t)) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[3] = 0.07 * np.sin(2 * np.pi * 1.8 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[4] = -0.12 * np.cos(2 * np.pi * 1.8 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[5] = 0.05 * np.sin(2 * np.pi * 0.9 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
    elif activity_idx == 3:  # SITTING
        data[0] = 0.02 * np.sin(2 * np.pi * 0.1 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[1] = 0.01 * np.sin(2 * np.pi * 0.15 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[2] = 0.98 + np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.5
        data[3] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[4] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[5] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.2
    elif activity_idx == 4:  # STANDING
        data[0] = 0.03 * np.sin(2 * np.pi * 0.3 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[1] = 0.02 * np.sin(2 * np.pi * 0.25 * t) + np.random.randn(CONFIG["sequence_length"]) * noise_scale
        data[2] = 0.99 + np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.5
        data[3] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[4] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[5] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.2
    elif activity_idx == 5:  # LAYING
        data[0] = 0.97 + np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[1] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[2] = 0.05 + np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.3
        data[3] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.2
        data[4] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.2
        data[5] = np.random.randn(CONFIG["sequence_length"]) * noise_scale * 0.1

    return data.astype(np.float32), activity_idx


def list_serial_ports() -> list:
    """List available serial ports on the system.

    Returns:
        List of serial port names (e.g., ['COM3', 'COM4'] or ['/dev/ttyUSB0']).
    """
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    except ImportError:
        return []


def render_sidebar() -> dict:
    """Render the sidebar and return user settings.

    Returns:
        Dictionary of sidebar settings.
    """
    with st.sidebar:
        st.title("🏃 EdgeHAR")
        st.caption("Edge Human Activity Recognition")
        st.divider()

        # Mode selector
        mode = st.radio(
            "📡 Mode",
            ["Live Simulation", "Upload CSV"],
            help="Select data input mode",
        )

        st.divider()

        # Serial port settings (for live mode)
        serial_port = None
        baud_rate = 115200
        if mode == "Live Simulation":
            st.subheader("🔌 Serial Connection")
            ports = list_serial_ports()
            if ports:
                serial_port = st.selectbox("Port", ports)
            else:
                st.info("No serial ports found. Using simulation mode.")

            baud_rate = st.number_input("Baud Rate", value=115200, step=9600)

            if serial_port:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔗 Connect", use_container_width=True):
                        st.session_state.is_connected = True
                with col2:
                    if st.button("🔌 Disconnect", use_container_width=True):
                        st.session_state.is_connected = False

        st.divider()

        # Confidence threshold
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.50,
            max_value=0.99,
            value=0.70,
            step=0.01,
            help="Minimum confidence to display a prediction",
        )

        # Show raw signals toggle
        show_signals = st.toggle("📊 Show Raw Signals", value=True)

        st.divider()
        st.caption("Built with ❤️ using PyTorch & Streamlit")

    return {
        "mode": mode,
        "serial_port": serial_port,
        "baud_rate": baud_rate,
        "confidence_threshold": confidence_threshold,
        "show_signals": show_signals,
    }


def render_activity_display(
    activity: str,
    confidence: float,
    probs: np.ndarray,
    threshold: float,
) -> None:
    """Render the current activity display with emoji and confidence bar.

    Args:
        activity: Predicted activity name.
        confidence: Prediction confidence (0-1).
        probs: All class probabilities.
        threshold: Minimum confidence threshold.
    """
    emoji = CONFIG["class_emojis"].get(activity, "❓")

    if confidence >= threshold:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px; color: white;">
                <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                <h2 style="margin: 10px 0;">{activity}</h2>
                <h3 style="margin: 0;">Confidence: {confidence:.1%}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        border-radius: 15px; color: white;">
                <h1 style="font-size: 4rem; margin: 0;">❓</h1>
                <h2 style="margin: 10px 0;">Uncertain</h2>
                <h3 style="margin: 0;">Below threshold ({threshold:.0%})</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Probability bars for all classes
    st.markdown("##### Class Probabilities")
    for i, (name, prob) in enumerate(zip(CONFIG["class_names"], probs)):
        emoji_small = CONFIG["class_emojis"].get(name, "")
        st.progress(float(prob), text=f"{emoji_small} {name}: {prob:.1%}")


def render_prediction_chart(predictions: list) -> None:
    """Render real-time prediction chart (last 50 predictions).

    Args:
        predictions: List of prediction dictionaries.
    """
    if not predictions:
        st.info("Waiting for predictions...")
        return

    recent = predictions[-50:]
    df = pd.DataFrame([
        {"Time": p["timestamp"], "Activity": p["activity_idx"]}
        for p in recent
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df["Activity"],
        mode="lines+markers",
        marker=dict(size=6, color=df["Activity"], colorscale="Viridis"),
        line=dict(width=2),
        hovertext=[CONFIG["class_names"][idx] for idx in df["Activity"]],
    ))

    fig.update_layout(
        title="Prediction Timeline (Last 50)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(6)),
            ticktext=[f"{CONFIG['class_emojis'].get(n, '')} {n}" for n in CONFIG["class_names"]],
        ),
        xaxis_title="Sample #",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(predictions: list, ground_truths: list) -> None:
    """Render a live confusion matrix from recent predictions.

    Args:
        predictions: List of prediction dictionaries.
        ground_truths: List of ground truth label indices.
    """
    if not predictions or not ground_truths:
        st.info("Collecting data for confusion matrix...")
        return

    # Use last 100 samples
    n = min(100, len(predictions), len(ground_truths))
    recent_preds = [p["activity_idx"] for p in predictions[-n:]]
    recent_truth = ground_truths[-n:]

    # Build confusion matrix
    cm = np.zeros((6, 6), dtype=int)
    for true, pred in zip(recent_truth, recent_preds):
        cm[true][pred] += 1

    # Render as heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=CONFIG["class_names"],
        y=CONFIG["class_names"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig.update_layout(
        title="Live Confusion Matrix (Last 100)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_signal_plot(sensor_data: np.ndarray) -> None:
    """Render raw sensor signal plot (6 channels, last 128 timesteps).

    Args:
        sensor_data: Array of shape (6, 128).
    """
    fig = go.Figure()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#C9B1FF"]

    for i, (name, color) in enumerate(zip(CONFIG["channel_names"], colors)):
        fig.add_trace(go.Scatter(
            y=sensor_data[i],
            mode="lines",
            name=name,
            line=dict(color=color, width=1.5),
        ))

    fig.update_layout(
        title="Raw Sensor Signals (128 timesteps)",
        xaxis_title="Timestep",
        yaxis_title="Value",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_table(predictions: list) -> None:
    """Render prediction history table.

    Args:
        predictions: List of prediction dictionaries.
    """
    if not predictions:
        st.info("No predictions yet.")
        return

    recent = predictions[-20:][::-1]  # Last 20, newest first
    df = pd.DataFrame([
        {
            "⏰ Time": p["timestamp"],
            "🏷️ Activity": f"{CONFIG['class_emojis'].get(p['activity'], '')} {p['activity']}",
            "📊 Confidence": f"{p['confidence']:.1%}",
        }
        for p in recent
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─── Main App ────────────────────────────────────────────────────────────────
def main() -> None:
    """Main Streamlit application."""
    init_session_state()

    # Load model
    session = load_onnx_model()

    # Sidebar
    settings = render_sidebar()

    # ─── Header ──────────────────────────────────────────────────────────
    st.title("🏃 EdgeHAR — Real-Time Activity Dashboard")
    st.caption("CNN-LSTM Human Activity Recognition from IMU Sensor Data")

    # Check if model is available
    if session is None:
        st.error(
            "⚠️ **ONNX model not found!**\n\n"
            "Please train and export the model first:\n"
            "```bash\n"
            "python data/download_data.py   # Download dataset\n"
            "python src/train.py            # Train model\n"
            "python src/export.py           # Export to ONNX\n"
            "```"
        )
        st.stop()

    # ─── Simulation Mode ─────────────────────────────────────────────────
    if settings["mode"] == "Live Simulation":
        # Generate simulated data
        sensor_data, gt_label = generate_simulated_data()
        st.session_state.sensor_buffer = sensor_data

        # Run inference
        pred_idx, confidence, probs = run_inference(session, sensor_data)
        pred_name = CONFIG["class_names"][pred_idx]

        # Store prediction
        prediction = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "activity": pred_name,
            "activity_idx": pred_idx,
            "confidence": confidence,
            "probs": probs.tolist(),
        }
        st.session_state.predictions.append(prediction)
        st.session_state.ground_truth.append(gt_label)

        # ─── Main Layout ────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 1.5, 1])

        with col1:
            render_activity_display(
                pred_name, confidence, probs, settings["confidence_threshold"]
            )

        with col2:
            render_prediction_chart(st.session_state.predictions)

        with col3:
            render_confusion_matrix(
                st.session_state.predictions,
                st.session_state.ground_truth,
            )

        # ─── Bottom Row ─────────────────────────────────────────────────
        st.divider()
        bottom_col1, bottom_col2 = st.columns([1.5, 1])

        with bottom_col1:
            if settings["show_signals"]:
                render_signal_plot(sensor_data)

        with bottom_col2:
            render_prediction_table(st.session_state.predictions)

        # Auto-rerun for live simulation
        time.sleep(CONFIG["simulation_interval"])
        st.rerun()

    # ─── Upload CSV Mode ─────────────────────────────────────────────────
    elif settings["mode"] == "Upload CSV":
        st.subheader("📂 Upload Sensor Data")
        uploaded_file = st.file_uploader(
            "Upload a CSV with 6 columns (ax, ay, az, gx, gy, gz) and 128 rows",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if df.shape[1] < 6:
                    st.error("CSV must have at least 6 columns (ax, ay, az, gx, gy, gz)")
                    st.stop()

                # Take first 6 columns and 128 rows
                data = df.iloc[:CONFIG["sequence_length"], :CONFIG["num_channels"]].values.T
                if data.shape != (6, 128):
                    st.warning(
                        f"Expected shape (6, 128), got ({data.shape[0]}, {data.shape[1]}). "
                        f"Padding/truncating to fit."
                    )
                    padded = np.zeros((6, 128), dtype=np.float32)
                    rows = min(data.shape[0], 6)
                    cols = min(data.shape[1], 128)
                    padded[:rows, :cols] = data[:rows, :cols]
                    data = padded

                sensor_data = data.astype(np.float32)

                # Run inference
                pred_idx, confidence, probs = run_inference(session, sensor_data)
                pred_name = CONFIG["class_names"][pred_idx]

                # Display results
                col1, col2 = st.columns([1, 2])
                with col1:
                    render_activity_display(
                        pred_name, confidence, probs, settings["confidence_threshold"]
                    )
                with col2:
                    render_signal_plot(sensor_data)

            except Exception as e:
                st.error(f"Error processing CSV: {e}")


if __name__ == "__main__":
    main()
