
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DATA_FILE = Path("data/ai4i2020.csv")
MODEL_FILE = Path("models/model.joblib")
METRICS_FILE = Path("models/metrics.json")

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #10233f, #176b75);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; margin-bottom: 0.4rem; }
    .hero p { color: #e1f5f5; margin-bottom: 0; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 14px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>⚙️ Predictive Maintenance AI</h1>
    <p>Machine Failure Prediction using Machine Learning</p>
    <p>Sensor analytics • Model evaluation • Failure classification</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load model and metrics ----------
if not MODEL_FILE.exists() or not METRICS_FILE.exists():
    st.error(
        "Trained model or metrics file is missing. "
        "Run `python train_model.py` first."
    )
    st.stop()

model = joblib.load(MODEL_FILE)

with open(METRICS_FILE, "r", encoding="utf-8") as file:
    saved_metrics = json.load(file)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Dashboard", "Predict Failure", "Model Performance", "Dataset Explorer"],
)

st.sidebar.divider()
st.sidebar.caption("College ML Project")
st.sidebar.caption("Dataset: AI4I 2020 Predictive Maintenance")

# ---------- Load dataset ----------
df = pd.read_csv(DATA_FILE) if DATA_FILE.exists() else None

# ---------- Dashboard ----------
if page == "Dashboard":
    st.subheader("Project Overview")
    st.write(
        "This application uses machine sensor readings and a trained "
        "classification model to estimate whether a record belongs to "
        "the machine-failure class."
    )

    if df is not None:
        total_records = len(df)
        failures = int(df["Machine failure"].sum())
        non_failures = total_records - failures
        failure_rate = failures / total_records * 100 if total_records else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_records:,}")
        c2.metric("Failure Records", f"{failures:,}")
        c3.metric("Non-Failure Records", f"{non_failures:,}")
        c4.metric("Recorded Failure Rate", f"{failure_rate:.2f}%")

        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("#### Failure Distribution")
            counts = df["Machine failure"].value_counts().reindex(
                [0, 1], fill_value=0
            )
            counts.index = ["No Failure", "Failure"]
            st.bar_chart(counts)

        with right:
            st.markdown("#### Sensor Overview")
            sensor = st.selectbox(
                "Select sensor",
                [
                    "Air temperature",
                    "Process temperature",
                    "Rotational speed",
                    "Torque",
                    "Tool wear",
                ],
            )
            st.line_chart(df[sensor].head(100))

    st.info(
        "This is a dataset-based educational prototype. "
        "It is not connected to a physical machine or live sensors."
    )

# ---------- Predict Failure ----------
elif page == "Predict Failure":
    st.subheader("Machine Sensor Input")
    st.write("Enter readings to generate a model prediction.")

    with st.form("prediction_form"):
        st.markdown("#### Machine Information")
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

        st.markdown("#### Temperature Readings")
        c1, c2 = st.columns(2)
        with c1:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=290.0,
                max_value=310.0,
                value=300.0,
                step=0.5,
            )
        with c2:
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=300.0,
                max_value=320.0,
                value=310.0,
                step=0.5,
            )

        st.markdown("#### Operating Conditions")
        c3, c4, c5 = st.columns(3)
        with c3:
            speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=1000,
                max_value=3000,
                value=1500,
                step=50,
            )
        with c4:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=1.0,
            )
        with c5:
            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=300,
                value=100,
                step=5,
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            type="primary",
            width="stretch",
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

        try:
            probability = float(model.predict_proba(input_data)[0][1])
            prediction = int(model.predict(input_data)[0])

            st.divider()
            st.subheader("Prediction Result")

            c1, c2 = st.columns(2)
            c1.metric(
                "Model-Estimated Failure Score",
                f"{probability * 100:.2f}%",
            )
            c2.metric(
                "Predicted Class",
                "Failure" if prediction == 1 else "No Failure",
            )

            # These are illustrative score bands, not validated
            # real-world maintenance thresholds.
            if probability < 0.30:
                risk = "LOW"
                st.success(f"LOW SCORE BAND — {probability * 100:.1f}%")
            elif probability < 0.60:
                risk = "MEDIUM"
                st.warning(f"MEDIUM SCORE BAND — {probability * 100:.1f}%")
            else:
                risk = "HIGH"
                st.error(f"HIGH SCORE BAND — {probability * 100:.1f}%")

            st.progress(probability)

            result = input_data.copy()
            result["Predicted Class"] = (
                "Failure" if prediction == 1 else "No Failure"
            )
            result["Model Score (%)"] = round(probability * 100, 2)

            st.markdown("#### Submitted Sensor Readings")
            st.dataframe(
    result.astype({
        "Type": "string",
        "Predicted Class": "string",
    }),
    hide_index=True,
    width="stretch"
)

            st.download_button(
                "Download Prediction Result (CSV)",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="machine_prediction.csv",
                mime="text/csv",
            )

            st.caption(
                "The score comes from a model trained on a synthetic dataset. "
                "The LOW/MEDIUM/HIGH bands are illustrative only, not "
                "validated real-world risk levels. The model does not "
                "forecast time until failure. Verify results before taking "
                "any maintenance action."
            )

        except (ValueError, KeyError) as error:
            st.error(f"Prediction input error: {error}")
            st.write(
                "Expected model features:",
                list(getattr(model, "feature_names_in_", [])),
            )

# ---------- Model Performance ----------
elif page == "Model Performance":
    st.subheader("Model Evaluation")
    st.write("Compare models using the saved test-set evaluation metrics.")

    results = saved_metrics.get("metrics", [])
    if not results:
        st.warning("No model metrics found. Run train_model.py again.")
    else:
        metrics_df = pd.DataFrame(results)
        st.dataframe(metrics_df, hide_index=True, width="stretch")

        chart_df = metrics_df.set_index("model")[
            ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ]
        st.markdown("#### Model Comparison")
        st.bar_chart(chart_df)

        selected = saved_metrics.get("selected_model", "Not available")
        st.success(f"Selected model based on test F1-score: {selected}")

        st.caption(
            "Metrics are from the held-out test split. Model selection "
            "using this same test split can make the reported performance "
            "optimistic; use a validation set for model selection and "
            "reserve a final untouched test set for a stronger evaluation."
        )

# ---------- Dataset Explorer ----------
elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")

    if df is None:
        st.warning("Dataset not found. Run `python download_data.py`.")
    else:
        st.write("Dataset shape:", df.shape)
        st.markdown("#### Preview")
        st.dataframe(
    df.head(100).astype(str),
    hide_index=True,
    width="stretch"
)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(
    df.describe(include="all").fillna("").astype(str),
    hide_index=True,
    width="stretch"
)

        st.markdown("#### Missing Values")
        missing = df.isnull().sum().rename("Missing Values")
        st.dataframe(missing)

        st.download_button(
            "Download Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ai4i2020.csv",
            mime="text/csv",
        )
