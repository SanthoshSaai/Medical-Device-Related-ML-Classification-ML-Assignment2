"""Train and evaluate five classifiers on UCI Cardiotocography (CTG) data.

Target: NSP (1=Normal, 2=Suspect, 3=Pathological).
The second UCI target, CLASS, is deliberately excluded to prevent target leakage.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
DATASET_ID = 193
TARGET = "NSP"
TARGET_LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological",
}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_ctg_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Fetch the official UCI dataset and return 21 predictors and NSP target."""
    print("Fetching Cardiotocography dataset (UCI ID 193)...")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("ucimlrepo is missing. Install dependencies with: python -m pip install -r requirements.txt") from exc
    dataset = fetch_ucirepo(id=DATASET_ID)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    if not isinstance(targets, pd.DataFrame):
        targets = pd.DataFrame(targets)

    if TARGET not in targets.columns:
        raise ValueError(
            f"Expected target column '{TARGET}' from UCI, found: {list(targets.columns)}"
        )

    # Defensive leakage guard: CLASS is a separate expert-assigned target and must
    # never be used to predict NSP.
    forbidden = {"CLASS", "NSP"}
    features = features.drop(columns=[c for c in forbidden if c in features.columns], errors="ignore")

    # UCI describes all 21 CTG measurements as input features. Convert to numeric;
    # any unexpected malformed cells become NaN and are imputed inside each pipeline.
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    raw_target = targets[TARGET]
    if pd.api.types.is_numeric_dtype(raw_target):
        target = pd.to_numeric(raw_target, errors="coerce")
    else:
        label_map = {
            "N": 1, "NORMAL": 1,
            "S": 2, "SUSPECT": 2,
            "P": 3, "PATHOLOGICAL": 3,
        }
        normalized = raw_target.astype(str).str.strip().str.upper()
        target = normalized.map(label_map)
        numeric_fallback = pd.to_numeric(raw_target, errors="coerce")
        target = target.fillna(numeric_fallback)

    valid_rows = target.notna()
    features = features.loc[valid_rows].reset_index(drop=True)
    target = target.loc[valid_rows].astype(int).reset_index(drop=True)

    if len(features) < 500:
        raise ValueError(f"Dataset has only {len(features)} usable rows; at least 500 are required.")
    if features.shape[1] < 12:
        raise ValueError(f"Dataset has only {features.shape[1]} usable input features; at least 12 are required.")

    observed_classes = sorted(target.unique().tolist())
    if observed_classes != [1, 2, 3]:
        raise ValueError(f"Unexpected NSP classes {observed_classes}; expected [1, 2, 3].")

    metadata = {
        "dataset_name": "Cardiotocography",
        "uci_dataset_id": DATASET_ID,
        "dataset_url": "https://archive.ics.uci.edu/dataset/193/cardiotocography",
        "citation": "Campos, D. & Bernardes, J. (2000). Cardiotocography. UCI Machine Learning Repository. https://doi.org/10.24432/C51S4N",
        "license": "CC BY 4.0",
        "usable_rows": int(len(features)),
        "input_feature_count": int(features.shape[1]),
        "input_features": list(features.columns),
        "target_column": TARGET,
        "target_labels": TARGET_LABELS,
        "target_distribution": {str(int(k)): int(v) for k, v in target.value_counts().sort_index().items()},
        "excluded_target_columns": ["CLASS"],
    }
    return features, target, metadata


def make_models() -> dict[str, Pipeline]:
    """Create the five required classifiers with preprocessing inside pipelines."""
    def scaled(classifier):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

    def unscaled(classifier):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])

    return {
        "Logistic Regression": scaled(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Decision Tree": unscaled(
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "K-Nearest Neighbors": scaled(
            KNeighborsClassifier(n_neighbors=9, weights="distance")
        ),
        "Gaussian Naive Bayes": unscaled(GaussianNB()),
        "Support Vector Machine": scaled(
            SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        ),
    }


def multiclass_auc(y_true, probabilities, classes) -> float:
    """Macro one-vs-rest AUC; only classes represented in y_true are estimable."""
    y_array = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    class_to_column = {label: index for index, label in enumerate(classes)}
    auc_values = []
    for label in np.unique(y_array):
        binary_truth = (y_array == label).astype(int)
        if binary_truth.min() == binary_truth.max():
            continue
        auc_values.append(roc_auc_score(binary_truth, probabilities[:, class_to_column[label]]))
    return float(np.mean(auc_values)) if auc_values else float("nan")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    fitted_classifier = model.named_steps["classifier"]
    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "AUC": multiclass_auc(y_test, probabilities, fitted_classifier.classes_),
        "Precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, predictions)),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_ctg_dataset()
    print(f"Dataset shape: {X.shape[0]:,} instances × {X.shape[1]} features")
    print("Target distribution:")
    print(y.value_counts().sort_index().rename(index=TARGET_LABELS).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # The app's upload contract is exactly these 21 feature columns plus NSP.
    test_export = X_test.copy()
    test_export[TARGET] = y_test.to_numpy()
    test_export.to_csv(TEST_DATA_PATH, index=False)

    comparison_rows = []
    model_file_names = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }

    trained_models = make_models()
    for model_name, model in trained_models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        comparison_rows.append({"ML Model Name": model_name, **metrics})

        bundle = {
            "model": model,
            "feature_names": list(X.columns),
            "target_name": TARGET,
            "target_labels": TARGET_LABELS,
            "model_name": model_name,
            "dataset_name": "Cardiotocography",
        }
        joblib.dump(bundle, MODEL_DIR / model_file_names[model_name])
        print("  " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(METRICS_PATH, index=False)

    metadata.update({
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "training_class_distribution": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_distribution": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "model_files": model_file_names,
    })
    with open(RESULTS_DIR / "dataset_metadata.json", "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    print("\nTraining completed.")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Model comparison: {METRICS_PATH}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
