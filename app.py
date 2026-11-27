"""Streamlit web interface for the machine failure prediction system.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.predict import load_bundle, predict_dataframe, predict_record, risk_level  # noqa: E402

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.6rem; max-width: 1250px;}
    .app-title {font-size: 2.05rem; font-weight: 800; color: #17324d; margin-bottom: 0.1rem;}
    .app-sub {color: #5c6b7a; font-size: 0.95rem; margin-bottom: 1.2rem;}
    .result-card {
        border-radius: 14px; padding: 1.15rem 1.35rem; margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(16,42,67,0.10);
    }
    .result-label {font-size: 0.85rem; letter-spacing: 0.09em; text-transform: uppercase;
                   color: #4a5a68; font-weight: 600;}
    .result-verdict {font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 0.15rem 0 0.35rem;}
    .result-detail {font-size: 0.95rem; color: #33475b;}
    .result-prob {font-size: 1.55rem; font-weight: 750; color: #17324d;}
    .metric-note {color: #6b7a89; font-size: 0.8rem;}
    .section-title {font-size: 1.15rem; font-weight: 700; color: #17324d;
                    margin: 0.4rem 0 0.5rem; border-bottom: 2px solid #e6edf3; padding-bottom: 0.3rem;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    .disclaimer {color: #6b7a89; font-size: 0.78rem; line-height: 1.4;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model...")
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data
def get_model_card() -> dict | None:
    if config.MODEL_CARD_PATH.exists():
        return json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    return None


def slider_bounds(bundle: dict, column: str, step: float) -> tuple[float, float]:
    low, high = bundle["input_ranges"][column]
    pad = (high - low) * 0.25
    low = max(0.0, low - pad)
    high = high + pad
    if step >= 1:
        return int(round(low)), int(round(high))
    return round(low, 1), round(high, 1)


def init_state(bundle: dict) -> None:
    defaults = bundle["input_defaults"]
    for key, value in defaults.items():
        st.session_state.setdefault(f"in_{key}", value)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("last_submit", None)


def apply_preset(bundle: dict, preset: dict) -> None:
    for key, value in preset.items():
        st.session_state[f"in_{key}"] = value
    st.session_state["result"] = None
    st.rerun()


def result_card(result: dict) -> None:
    if result["prediction"] == 1:
        colour, background, verdict = "#c9402b", "#fdeeea", "FAILURE PREDICTED"
    else:
        colour, background, verdict = "#1f7a8c", "#e9f4f6", "NO FAILURE PREDICTED"

    probability = result["probability"]
    probability_text = (
        "<0.01%" if probability < 0.0001
        else f"{probability:.2%}" if probability < 0.01
        else f"{probability:.1%}"
    )
    st.markdown(
        f"""
        <div class="result-card" style="border-left: 9px solid {colour}; background: {background};">
            <div class="result-label">Prediction for the entered machine</div>
            <div class="result-verdict" style="color:{colour};">{verdict}</div>
            <div class="result-detail">Machine failure: <b>{result['label']}</b>
                &nbsp;|&nbsp; Risk level: <b>{result['risk_level']}</b></div>
            <div class="result-prob" style="margin-top:0.55rem;">{probability_text}
                <span style="font-size:0.9rem;font-weight:500;color:#4a5a68;">
                estimated failure probability</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Decision threshold {result['threshold']:.2f} - probabilities at or above it "
        "are classified as a failure."
    )


def main() -> None:
    bundle = get_bundle()
    card = get_model_card()
    init_state(bundle)

    st.markdown('<div class="app-title">Machine Failure Prediction System</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Predictive maintenance for industrial machines - '
        'AI4I 2020 dataset, six sensor inputs, one risk decision.</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar: trained model summary
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Model in use")
        st.markdown(f"**{bundle['model_name']}**")
        st.caption(
            f"5-fold CV F1: **{bundle.get('cv_f1_mean', float('nan')):.3f}**  \n"
            f"Trained: {bundle.get('trained_at', 'unknown')}"
        )
        st.markdown("---")
        st.markdown("#### Held-out test performance")
        metrics = bundle.get("test_metrics", {})
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        col_b.metric("F1 (failure)", f"{metrics.get('f1', 0):.3f}")
        col_a.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        col_b.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        col_a.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        col_b.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
        st.caption(
            f"Confusion matrix: {metrics.get('tn', 0)} TN, {metrics.get('fp', 0)} FP, "
            f"{metrics.get('fn', 0)} FN, {metrics.get('tp', 0)} TP"
        )

        with st.expander("Hyperparameters and features"):
            params = bundle.get("best_params", {})
            st.dataframe(
                pd.DataFrame({"setting": list(params),
                              "value": [str(v) for v in params.values()]}),
                hide_index=True,
            )
            st.markdown("**Input features**")
            st.code(", ".join(bundle["features"]), language=None)
            if bundle.get("engineered_features"):
                st.markdown("**Derived automatically**")
                st.code(", ".join(bundle["engineered_features"]), language=None)

        st.markdown("---")
        st.markdown(
            '<div class="disclaimer">Educational decision support only. '
            "AI4I is a synthetic dataset; do not use this model for real "
            "maintenance decisions without validation on plant data.</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">1. Choose a scenario or enter measurements</div>',
                unsafe_allow_html=True)
    presets = {
        "Healthy machine": {
            "Type": "L", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1550, "Torque [Nm]": 42.8, "Tool wear [min]": 12,
        },
        "Worn tool": {
            "Type": "L", "Air temperature [K]": 300.4, "Process temperature [K]": 310.5,
            "Rotational speed [rpm]": 1450, "Torque [Nm]": 55.0, "Tool wear [min]": 215,
        },
        "Power overload": {
            "Type": "L", "Air temperature [K]": 301.2, "Process temperature [K]": 311.8,
            "Rotational speed [rpm]": 1300, "Torque [Nm]": 70.0, "Tool wear [min]": 200,
        },
    }
    preset_columns = st.columns(len(presets))
    for column, (label, preset) in zip(preset_columns, presets.items()):
        if column.button(label, width="stretch"):
            apply_preset(bundle, preset)

    # ------------------------------------------------------------------
    # Input form
    # ------------------------------------------------------------------
    with st.form("machine_form", border=True):
        left, right = st.columns(2)
        with left:
            machine_type = st.selectbox(
                "Machine quality type", config.TYPE_CATEGORIES,
                key="in_Type", help="L = low, M = medium, H = high quality variant.",
            )
            air_low, air_high = slider_bounds(bundle, "Air temperature [K]", 0.1)
            air = st.slider("Air temperature [K]", air_low, air_high, step=0.1,
                            key="in_Air temperature [K]")
            process_low, process_high = slider_bounds(bundle, "Process temperature [K]", 0.1)
            process = st.slider("Process temperature [K]", process_low, process_high,
                                step=0.1, key="in_Process temperature [K]")
        with right:
            rpm_low, rpm_high = slider_bounds(bundle, "Rotational speed [rpm]", 1)
            rpm = st.slider("Rotational speed [rpm]", rpm_low, rpm_high, step=1,
                            key="in_Rotational speed [rpm]")
            torque_low, torque_high = slider_bounds(bundle, "Torque [Nm]", 0.1)
            torque = st.slider("Torque [Nm]", torque_low, torque_high, step=0.1,
                               key="in_Torque [Nm]")
            wear_low, wear_high = slider_bounds(bundle, "Tool wear [min]", 1)
            wear = st.slider("Tool wear [min]", wear_low, wear_high, step=1,
                             key="in_Tool wear [min]")

        st.caption(
            "Sliders cover the observed training range. Mechanical power and the "
            "temperature difference are derived by the saved pipeline - no extra input needed."
        )
        submitted = st.form_submit_button(
            "Predict failure risk", type="primary", width="stretch"
        )

    if submitted:
        if process < air:
            st.error("Process temperature must be at least the air temperature.")
        else:
            record = {
                "Type": machine_type,
                "Air temperature [K]": float(air),
                "Process temperature [K]": float(process),
                "Rotational speed [rpm]": float(rpm),
                "Torque [Nm]": float(torque),
                "Tool wear [min]": float(wear),
            }
            st.session_state["result"] = predict_record(bundle, record)
            st.session_state["last_submit"] = pd.Timestamp.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    result = st.session_state.get("result")
    st.markdown('<div class="section-title">2. Prediction</div>', unsafe_allow_html=True)

    if result is None:
        st.info("Set the measurements and press **Predict failure risk** to score the machine.")
    else:
        left, right = st.columns([1.15, 1])
        with left:
            result_card(result)
            if result["warnings"]:
                for warning in result["warnings"]:
                    st.warning(warning)
            st.caption(f"Scored at {st.session_state.get('last_submit', '')} "
                       f"with {result['model_name']}.")

        with right:
            st.markdown("#### Inputs reviewed")
            record = result["record"]
            table = pd.DataFrame(
                {
                    "Measurement": [
                        "Type", "Air temperature", "Process temperature",
                        "Rotational speed", "Torque", "Tool wear",
                        "Mechanical power", "Temperature difference",
                    ],
                    "Value": [
                        record["Type"],
                        f"{record['Air temperature [K]']:.1f} K",
                        f"{record['Process temperature [K]']:.1f} K",
                        f"{record['Rotational speed [rpm]']:.0f} rpm",
                        f"{record['Torque [Nm]']:.1f} Nm",
                        f"{record['Tool wear [min]']:.0f} min",
                        f"{result['engineered']['Mechanical power [W]']:.0f} W",
                        f"{result['engineered']['Temperature difference [K]']:.1f} K",
                    ],
                }
            )
            st.dataframe(table, hide_index=True, width="stretch")

            probability = result["probability"]
            st.markdown("#### Risk gauge")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(
                f"LOW below 30% | MODERATE 30-60% | HIGH at or above 60%. "
                f"This machine: **{risk_level(probability)}**."
            )

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">3. Batch scoring (optional)</div>',
                unsafe_allow_html=True)
    with st.expander("Upload a CSV of machines and download predictions"):
        st.caption(
            "Required columns: Type, Air temperature [K], Process temperature [K], "
            "Rotational speed [rpm], Torque [Nm], Tool wear [min]."
        )
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                scored = predict_dataframe(bundle, frame)
                st.dataframe(scored, width="stretch")
                st.download_button(
                    "Download predictions",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="machine_failure_predictions.csv",
                    mime="text/csv",
                )
                st.caption(f"{int(scored['prediction'].sum())} of {len(scored)} "
                           "machines flagged as likely to fail.")
            except ValueError as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
