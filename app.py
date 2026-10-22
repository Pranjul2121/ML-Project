
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

DATA_FILE = Path("data/ai4i2020.csv")
MODEL_FILE = Path("models/model.joblib")
METRICS_FILE = Path("models/metrics.json")

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ AI Predictive Maintenance System")
st.caption(
    "Machine failure classification using machine learning "
    "| Educational prototype"
)

if not MODEL_FILE.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_FILE)

with open(METRICS_FILE, "r") as file:
    saved_metrics = json.load(file)

if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric(
        "Recorded failures",
        f"{int(df['Machine failure'].sum()):,}",
    )
    c3.metric(
        "Selected model",
        saved_metrics["selected_model"],
    )

    with st.expander("Explore training dataset", expanded=False):
        st.write("Failure distribution")
        st.bar_chart(df["Machine failure"].value_counts())
        st.write("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.header("Machine Sensor Input")

st.write(
    "Enter readings for a machine. The model will estimate "
    "its probability of belonging to the failure class."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine type",
            ["L", "M", "H"],
            help="Product quality category in the AI4I dataset.",
        )

        air_temp = st.number_input(
            "Air temperature (K)",
            min_value=290.0,
            max_value=310.0,
            value=300.0,
            step=0.5,
        )

        process_temp = st.number_input(
            "Process temperature (K)",
            min_value=300.0,
            max_value=320.0,
            value=310.0,
            step=0.5,
        )

    with col2:
        speed = st.number_input(
            "Rotational speed (rpm)",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=50,
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
        )

        tool_wear = st.number_input(
            "Tool wear (minutes)",
            min_value=0,
            max_value=300,
            value=100,
            step=5,
        )

    submitted = st.form_submit_button(
        "Predict Machine Failure",
        type="primary",
        width="stretch"
    )

if submitted:
    input_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": speed,
    "Torque": torque,
    "Tool wear": tool_wear,
}])


    probability = float(
        model.predict_proba(input_data)[0][1]
    )
    prediction = int(model.predict(input_data)[0])

    st.divider()
    st.subheader("Prediction Results")

    r1, r2 = st.columns(2)

    r1.metric(
        "Model-estimated failure probability",
        f"{probability * 100:.2f}%",
    )

    r2.metric(
        "Predicted class",
        "Failure" if prediction == 1 else "No failure",
    )

    st.progress(probability)

    if prediction == 1:
        st.error(
            "The model predicts the failure class. "
            "Inspect the machine and verify readings."
        )
    else:
        st.success(
            "The model predicts the non-failure class. "
            "Continue normal monitoring."
        )

    st.caption(
        "This is a model score derived from a synthetic dataset, "
        "not a validated real-world failure forecast. "
        "The score may not be calibrated. Do not use it alone "
        "for safety-critical maintenance decisions."
    )

st.divider()
st.header("Model Evaluation")

for result in saved_metrics["metrics"]:
    st.subheader(result["model"])
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{result['accuracy']:.3f}")
    b.metric("Precision", f"{result['precision']:.3f}")
    c.metric("Recall", f"{result['recall']:.3f}")
    d.metric("F1-score", f"{result['f1']:.3f}")
    e.metric("ROC-AUC", f"{result['roc_auc']:.3f}")
