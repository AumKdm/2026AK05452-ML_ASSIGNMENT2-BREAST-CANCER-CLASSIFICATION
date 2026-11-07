"""
Breast Cancer Classifier – Model Evaluation App
ML Assignment 2 | 2026AK05452

Loads the 5 pre-trained models from model/ and evaluates them on an uploaded
test CSV. Nothing is retrained here: the app only calls predict / predict_proba.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_FILE = BASE_DIR / "test_data.csv"

PINK = "#C2185B"
TEAL = "#00897B"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_HELP = {
    "Accuracy": "Share of all test rows predicted correctly.",
    "AUC": "Area under the ROC curve: how well the model ranks malignant above benign (1.0 = perfect).",
    "Precision": "Of the tumours predicted Malignant, the share that really are Malignant.",
    "Recall": "Of the truly Malignant tumours, the share the model caught.",
    "F1": "Harmonic mean of Precision and Recall.",
    "MCC": "Matthews Correlation Coefficient: balanced score from -1 to 1, robust to class imbalance.",
}

st.set_page_config(
    page_title="Breast Cancer Classifier | ML Assignment 2",
    page_icon="🎗️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
      .hero {{
        background: linear-gradient(120deg, {PINK} 0%, #8E24AA 100%);
        padding: 1.4rem 1.8rem; border-radius: 14px; color: white; margin-bottom: 1.2rem;
      }}
      .hero h1 {{ color: white; font-size: 1.9rem; margin: 0 0 .3rem 0; }}
      .hero p  {{ color: #FCE4EC; margin: 0; font-size: 1rem; }}
      div[data-testid="stMetric"] {{
        background: rgba(194, 24, 91, 0.06);
        border: 1px solid rgba(194, 24, 91, 0.25);
        border-radius: 12px; padding: .7rem .9rem;
      }}
      .pill {{
        display: inline-block; padding: .15rem .65rem; border-radius: 999px;
        background: rgba(0, 137, 123, .12); color: {TEAL}; font-weight: 600; font-size: .85rem;
      }}
    </style>
    <div class="hero">
      <h1>🎗️ Breast Cancer Diagnosis – Model Evaluation Studio</h1>
      <p>Upload the test CSV, pick one of five pre-trained classifiers, and see how it performs
      on Malignant vs Benign tumours. No model is retrained in this app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved models…")
def load_artifacts():
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    return meta, models


@st.cache_data
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


try:
    META, MODELS = load_artifacts()
except Exception as exc:  # pragma: no cover - shown in the UI
    st.error(f"Could not load the saved models from `model/`: {exc}")
    st.stop()

FEATURES = META["feature_names"]
TARGET = META["target"]
LABELS = [META["class_labels"]["0"], META["class_labels"]["1"]]  # ['Benign', 'Malignant']


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame):
    """Validate the uploaded frame and return (X, y) or raise ValueError."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required feature column(s) are missing: "
            + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        )
    if TARGET not in df.columns:
        raise ValueError(f"The target column `{TARGET}` is missing. The test CSV must include it.")

    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):  # also accept the original UCI letters
        y = y.astype(str).str.strip().str.upper().map({"M": 1, "B": 0, "MALIGNANT": 1, "BENIGN": 0})
    if y.isna().any() or not set(pd.unique(y)).issubset({0, 1}):
        raise ValueError(f"`{TARGET}` must contain only 0/1 (or B/M) values.")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad_rows = X.isna().any(axis=1)
    if bad_rows.all():
        raise ValueError("No usable rows: every row has a missing or non-numeric feature value.")
    return X[~bad_rows], y[~bad_rows].astype(int), int(bad_rows.sum())


def score(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    both_classes = y.nunique() == 2
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if both_classes else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_prob


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False, ax=ax,
                xticklabels=LABELS, yticklabels=LABELS, annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig, cm


def plot_roc(y, y_prob, auc_value):
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr, tpr, color=PINK, lw=2.2, label=f"AUC = {auc_value:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PINK)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (Recall)")
    ax.set_title("ROC curve", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Controls on the main page: upload + model choice
# --------------------------------------------------------------------------- #
ctrl_upload, ctrl_model = st.columns([3, 2], gap="large")

with ctrl_upload:
    st.markdown("#### 1️⃣ Upload test data")
    uploaded = st.file_uploader(
        "Test CSV (30 features + `diagnosis`)", type=["csv"],
        help="Upload test data only. It must contain all 30 feature columns and the target column.",
    )
    if SAMPLE_TEST_FILE.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=SAMPLE_TEST_FILE.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )

with ctrl_model:
    st.markdown("#### 2️⃣ Choose a model")
    model_name = st.selectbox("Classifier", list(MODELS.keys()),
                              index=list(MODELS.keys()).index("Support Vector Machine"))
    st.caption(
        f"Models trained on {META.get('n_rows_train', '455')} rows of the "
        f"[UCI Breast Cancer Wisconsin (Diagnostic)]({META.get('dataset_url', '')}) dataset. "
        f"scikit-learn {META.get('sklearn_version', '')}."
    )

st.divider()

if uploaded is None:
    st.info("☝️ Upload your **test CSV** above to begin. "
            "No file handy? Download the sample `test_data.csv` above and upload it.")
    with st.expander("Expected CSV format"):
        st.write(f"**{len(FEATURES)} feature columns** + target column **`{TARGET}`** "
                 "(1 = Malignant, 0 = Benign).")
        st.code(", ".join(FEATURES + [TARGET]), language="text")
    st.stop()

# --------------------------------------------------------------------------- #
# Validate data
# --------------------------------------------------------------------------- #
try:
    raw = read_csv(uploaded.getvalue())
    X, y, dropped = prepare(raw)
except Exception as exc:
    st.error(f"❌ {exc}")
    st.stop()

if dropped:
    st.warning(f"{dropped} row(s) had missing or non-numeric feature values and were skipped.")

n_mal, n_ben = int((y == 1).sum()), int((y == 0).sum())
st.markdown(
    f"<span class='pill'>📄 {uploaded.name}</span> &nbsp; "
    f"<span class='pill'>{len(X)} rows</span> &nbsp; "
    f"<span class='pill'>{raw.shape[1]} columns</span> &nbsp; "
    f"<span class='pill'>Malignant {n_mal} · Benign {n_ben}</span>",
    unsafe_allow_html=True,
)
st.write("")

tab_model, tab_compare, tab_data = st.tabs(
    ["🔍 Selected model", "🏆 Compare all 5 models", "🗂️ Uploaded data"]
)

# --------------------------------------------------------------------------- #
# Tab 1: selected model
# --------------------------------------------------------------------------- #
with tab_model:
    metrics, y_pred, y_prob = score(MODELS[model_name], X, y)
    st.subheader(f"{model_name}")

    cols = st.columns(6)
    for col, name in zip(cols, METRIC_NAMES):
        val = metrics[name]
        col.metric(name, "n/a" if np.isnan(val) else f"{val:.4f}", help=METRIC_HELP[name])

    left, right = st.columns(2)
    with left:
        fig_cm, cm = plot_confusion(y, y_pred, f"Confusion matrix – {model_name}")
        st.pyplot(fig_cm)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"✅ {tp} malignant caught · ⚠️ {fn} malignant missed · "
                   f"{fp} false alarm(s) · {tn} benign correctly cleared")
    with right:
        if np.isnan(metrics["AUC"]):
            st.info("ROC curve needs both classes in the uploaded data.")
        else:
            st.pyplot(plot_roc(y, y_prob, metrics["AUC"]))

    st.markdown("#### Classification report")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=LABELS,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

    with st.expander("See individual predictions"):
        preds = X.copy()
        preds["actual"] = y.map({0: LABELS[0], 1: LABELS[1]}).values
        preds["predicted"] = pd.Series(y_pred, index=X.index).map({0: LABELS[0], 1: LABELS[1]})
        preds["P(Malignant)"] = np.round(y_prob, 4)
        preds["correct"] = preds["actual"] == preds["predicted"]
        st.dataframe(preds[["actual", "predicted", "P(Malignant)", "correct"] + FEATURES[:5]], height=320)
        st.download_button("⬇️ Download predictions", preds.to_csv(index=False).encode(),
                           file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                           mime="text/csv")

# --------------------------------------------------------------------------- #
# Tab 2: all models on the uploaded data
# --------------------------------------------------------------------------- #
with tab_compare:
    st.subheader("All five models on the uploaded test data")
    rows = {name: score(m, X, y)[0] for name, m in MODELS.items()}
    comp = pd.DataFrame(rows).T[METRIC_NAMES]
    comp.index.name = "ML Model Name"

    best = comp["MCC"].idxmax()
    st.success(f"🏆 Best model on this data by MCC: **{best}** "
               f"(MCC = {comp.loc[best, 'MCC']:.4f}, Accuracy = {comp.loc[best, 'Accuracy']:.4f})")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#F8BBD0"))

    long = comp.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="RdPu_r", ax=ax)
    ax.set_ylim(max(0, np.nanmin(comp.values) - 0.05), 1.0)
    ax.set_xlabel("")
    ax.legend(loc="lower right", ncol=5, fontsize=8, title="")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Confusion matrices")
    cm_cols = st.columns(5)
    for col, (name, m) in zip(cm_cols, MODELS.items()):
        fig_small, _ = plot_confusion(y, m.predict(X), name)
        col.pyplot(fig_small)

# --------------------------------------------------------------------------- #
# Tab 3: data preview
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Uploaded data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows used", len(X))
    c2.metric("Feature columns", len(FEATURES))
    c3.metric("Malignant share", f"{n_mal / len(y):.1%}")
    st.caption(f"Showing all {len(raw)} rows and {raw.shape[1]} columns. Scroll inside the table to see more.")
    st.dataframe(raw, height=420)

st.divider()
st.caption("ML Assignment 2 · 2026AK05452 · Logistic Regression, Decision Tree, KNN, "
           "Gaussian Naive Bayes, SVM · built with Streamlit")
