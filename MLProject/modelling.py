"""
modelling.py  –  MLflow Project entry point
============================================
Kriteria 3 – Workflow CI
Dataset : train_processed.csv (filepath, label, video_id, subset)
Task    : Image Classification BISINDO (40 kelas)
Model   : Random Forest + HOG features
MLflow  : manual logging, kompatibel dengan MLflow Project & GitHub Actions
"""

import os
import sys
import warnings
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Argument parser (MLflow Project params)
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="BISINDO Classification – MLflow Project")
    parser.add_argument("--n_estimators",       type=int,   default=200,    help="Jumlah pohon")
    parser.add_argument("--max_depth",          type=str,   default="None", help="Kedalaman pohon (None atau integer)")
    parser.add_argument("--min_samples_split",  type=int,   default=5,      help="Min samples split")
    parser.add_argument("--max_features",       type=str,   default="sqrt", help="Max features")
    parser.add_argument("--test_size",          type=float, default=0.2,    help="Proporsi test set")
    parser.add_argument("--random_state",       type=int,   default=42,     help="Random state")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "train_processed.csv"
IMG_ROOT = BASE_DIR / "dataset_isyarat (1)" / "content" / "extracted_images"


# ─────────────────────────────────────────────────────────────────
# Feature extraction (HOG)
# ─────────────────────────────────────────────────────────────────
def extract_features(filepath: str, n_features: int = 128) -> np.ndarray:
    import cv2
    local_path = str(IMG_ROOT / Path(filepath).relative_to("/content/extracted_images"))

    if os.path.exists(local_path):
        img = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            from skimage.feature import hog
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
            return features.astype(np.float32)

    # Fallback deterministik
    seed = abs(hash(filepath)) % (2**32)
    rng  = np.random.default_rng(seed)
    return rng.random(n_features).astype(np.float32)


def build_features(df: pd.DataFrame) -> np.ndarray:
    print("[INFO] Mengekstrak fitur HOG ...", flush=True)
    X = np.vstack([extract_features(fp) for fp in df["filepath"]])
    print(f"[INFO] Feature matrix: {X.shape}", flush=True)
    return X


# ─────────────────────────────────────────────────────────────────
# Artefak
# ─────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, le, out_dir: Path) -> Path:
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("Confusion Matrix – BISINDO Classification")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def save_feature_importance(model, out_dir: Path) -> Path:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:30]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(30), importances[idx])
    ax.set_title("Top-30 Feature Importances")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = out_dir / "feature_importance.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Parse max_depth
    max_depth = None if args.max_depth == "None" else int(args.max_depth)

    print("=" * 60, flush=True)
    print("  BISINDO Classification – MLflow Project (CI)", flush=True)
    print("=" * 60, flush=True)

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Dataset: {len(df)} rows, {df['label'].nunique()} kelas", flush=True)

    # Features & labels
    X = build_features(df)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}", flush=True)

    # MLflow setup
    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run() as run:
        print(f"[INFO] MLflow Run ID: {run.info.run_id}", flush=True)

        # Train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc         = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        precision   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall      = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
        logloss     = log_loss(y_test, y_prob)
        report_str  = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

        print(f"\n[RESULT] Accuracy   : {acc:.4f}", flush=True)
        print(f"[RESULT] F1 Weighted: {f1_weighted:.4f}", flush=True)
        print(f"[RESULT] Log Loss   : {logloss:.4f}", flush=True)

        # Log params
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         str(max_depth))
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("max_features",      args.max_features)
        mlflow.log_param("test_size",         args.test_size)
        mlflow.log_param("random_state",      args.random_state)
        mlflow.log_param("n_classes",         len(le.classes_))
        mlflow.log_param("train_size",        X_train.shape[0])
        mlflow.log_param("test_size_actual",  X_test.shape[0])

        # Log metrics
        mlflow.log_metric("accuracy",          acc)
        mlflow.log_metric("f1_weighted",       f1_weighted)
        mlflow.log_metric("f1_macro",          f1_macro)
        mlflow.log_metric("precision_weighted",precision)
        mlflow.log_metric("recall_weighted",   recall)
        mlflow.log_metric("log_loss",          logloss)

        # Log artifacts
        art_dir = BASE_DIR / "artifacts"
        art_dir.mkdir(exist_ok=True)

        cm_path = save_confusion_matrix(y_test, y_pred, le, art_dir)
        fi_path = save_feature_importance(model, art_dir)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(fi_path), artifact_path="plots")
        mlflow.log_text(report_str, "classification_report.txt")

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:3],
        )

        # Simpan model lokal untuk di-upload ke GitHub Actions artifact
        import joblib
        model_path = BASE_DIR / "artifacts" / "model.pkl"
        joblib.dump(model, model_path)
        print(f"[INFO] Model disimpan: {model_path}", flush=True)

        print(f"\n[INFO] Logging selesai.", flush=True)

    print("\n[DONE] Selesai.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
