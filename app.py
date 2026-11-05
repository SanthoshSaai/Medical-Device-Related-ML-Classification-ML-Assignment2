"""Streamlit dashboard for the saved Cardiotocography classifiers.

The app evaluates uploaded, already-labelled test CSV rows. It never retrains models.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
TARGET = "NSP"
CLASS_NAMES = {1: "Normal", 2: "Suspect", 3: "Pathological"}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="MedGuard CTG | Fetal State Classification", page_icon="🩺", layout="wide")
st.title("MedGuard CTG")
st.caption("Cardiotocography-based fetal-state classification · Five ML classifiers")
st.info(
    "Academic machine-learning demonstration using UCI Cardiotocography data. "
    "It is not a clinical decision-support tool and must not be used to diagnose or guide patient care."
)


@st.cache_resource
def load_bundles():
    loaded = {}
    for display_name, file_name in MODEL_FILES.items():
        path = MODEL_DIR / file_name
        if path.exists():
            loaded[display_name] = joblib.load(path)
    return loaded


def compute_auc(y_true, probabilities, classes):
    """Compute macro one-vs-rest AUC for classes represented in the uploaded test set."""
    y_array = np.asarray(y_true)
    classes = np.asarray(classes)
    col_for_label = {label: column for column, label in enumerate(classes)}
    values = []
    for label in np.unique(y_array):
        binary_truth = (y_array == label).astype(int)
        if binary_truth.min() == binary_truth.max():
            continue
        values.append(
            roc_auc_score(binary_truth, np.asarray(probabilities)[:, col_for_label[label]])
        )
    return float(np.mean(values)) if values else float("nan")


def score_model(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    classes = model.named_steps["classifier"].classes_
    return {
        "Accuracy": accuracy_score(y, prediction),
        "AUC (macro OvR)": compute_auc(y, probability, classes),
        "Precision (macro)": precision_score(y, prediction, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, prediction, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y, prediction, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }, prediction


bundles = load_bundles()
if len(bundles) != len(MODEL_FILES):
    st.error(
        "Trained model files are missing. Run `python train.py` first, then commit the generated "
        "model files to GitHub before deploying this app."
    )
    st.stop()

st.sidebar.header("Evaluation setup")
uploaded = st.sidebar.file_uploader(
    "Upload a labelled test CSV",
    type=["csv"],
    help="The CSV must contain the 21 predictor columns and the target column NSP (1, 2, or 3).",
)

if uploaded is not None:
    try:
        test_data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"The uploaded file could not be read as CSV: {exc}")
        st.stop()
else:
    if not TEST_DATA_PATH.exists():
        st.warning("Upload a labelled test CSV, or run train.py to generate test_data.csv.")
        st.stop()
    test_data = pd.read_csv(TEST_DATA_PATH)
    st.sidebar.caption("Using the generated held-out test_data.csv.")

# All models share the exact training feature list.
reference_bundle = next(iter(bundles.values()))
feature_names = reference_bundle["feature_names"]
missing_columns = [name for name in feature_names if name not in test_data.columns]
if TARGET not in test_data.columns:
    st.error("The test CSV must include the target column 'NSP' so evaluation metrics can be calculated.")
    st.stop()
if missing_columns:
    st.error(f"The test CSV is missing required feature columns: {', '.join(missing_columns)}")
    st.stop()
if test_data.empty:
    st.error("The uploaded test CSV has no data rows.")
    st.stop()

try:
    X_test = test_data[feature_names].apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(test_data[TARGET], errors="raise").astype(int)
except Exception as exc:
    st.error(f"Feature or target columns contain invalid values: {exc}")
    st.stop()

valid_labels = set(CLASS_NAMES)
unknown_labels = sorted(set(y_test.unique()) - valid_labels)
if unknown_labels:
    st.error(f"Unexpected NSP labels {unknown_labels}. Expected 1=Normal, 2=Suspect, 3=Pathological.")
    st.stop()

st.subheader("Dataset preview")
left, middle, right = st.columns(3)
left.metric("Test records", f"{len(test_data):,}")
middle.metric("Input features", f"{len(feature_names)}")
right.metric("Classes present", f"{y_test.nunique()} / 3")
st.dataframe(test_data.head(8), use_container_width=True)

# Evaluate every saved classifier on the same uploaded test data so the comparison
# table updates when the user uploads a different compatible test CSV.
all_metrics = {}
all_predictions = {}
for name, bundle in bundles.items():
    metric_values, predictions = score_model(bundle["model"], X_test, y_test)
    all_metrics[name] = metric_values
    all_predictions[name] = predictions

comparison = pd.DataFrame.from_dict(all_metrics, orient="index")
comparison.index.name = "ML Model Name"
st.subheader("Five-model performance comparison")
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

selected_model = st.selectbox("Select a classifier for detailed evaluation", list(bundles.keys()))
selected_bundle = bundles[selected_model]
selected_model_object = selected_bundle["model"]
selected_prediction = all_predictions[selected_model]
selected_probability = selected_model_object.predict_proba(X_test)
selected_classes = selected_model_object.named_steps["classifier"].classes_

st.subheader(f"Detailed metrics — {selected_model}")
metric_columns = st.columns(3)
for idx, (metric_name, metric_value) in enumerate(all_metrics[selected_model].items()):
    shown = "N/A" if not np.isfinite(metric_value) else f"{metric_value:.4f}"
    metric_columns[idx % 3].metric(metric_name, shown)

st.subheader("Confusion matrix")
labels = [1, 2, 3]
cm = confusion_matrix(y_test, selected_prediction, labels=labels)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
image = ax.imshow(cm, interpolation="nearest")
ax.set_title("Actual vs predicted fetal-state class")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_xticks(range(3), [f"{label} — {CLASS_NAMES[label]}" for label in labels], rotation=15, ha="right")
ax.set_yticks(range(3), [f"{label} — {CLASS_NAMES[label]}" for label in labels])
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
for row in range(cm.shape[0]):
    for column in range(cm.shape[1]):
        ax.text(column, row, str(cm[row, column]), ha="center", va="center")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification report")
report = classification_report(
    y_test,
    selected_prediction,
    labels=labels,
    target_names=[f"{label} — {CLASS_NAMES[label]}" for label in labels],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Prediction sample")
prediction_preview = pd.DataFrame({
    "Actual NSP": y_test.to_numpy(),
    "Actual label": [CLASS_NAMES[int(value)] for value in y_test],
    "Predicted NSP": selected_prediction,
    "Predicted label": [CLASS_NAMES[int(value)] for value in selected_prediction],
})
st.dataframe(prediction_preview.head(25), use_container_width=True)
st.download_button(
    "Download predictions CSV",
    data=prediction_preview.to_csv(index=False).encode("utf-8"),
    file_name="ctg_predictions.csv",
    mime="text/csv",
)

with st.expander("About the dataset and intended use"):
    st.write(
        "The UCI Cardiotocography dataset contains 2,126 cardiotocograms and 21 measured input features. "
        "The target NSP represents fetal state: Normal (1), Suspect (2), or Pathological (3). "
        "The additional UCI target CLASS is intentionally excluded from the predictors to avoid label leakage."
    )
    st.markdown("Source: https://archive.ics.uci.edu/dataset/193/cardiotocography")
