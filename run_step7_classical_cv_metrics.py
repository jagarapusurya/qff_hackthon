"""
Fast 5-fold classical baseline for QFF2026 Q2.
Uses the same 300-row subset as the noisy-kernel run when its saved index file exists.
Fits StandardScaler and PCA inside each fold to avoid preprocessing leakage.
Writes metrics and out-of-fold predictions under results/tables/.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.q2_data import stratified_subset

TRAIN_PATH = ROOT / "data" / "q2" / "hd_data_train.csv"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SUBSET_SIZE = 300
SUBSET_SEED = 42
N_SPLITS = 5
N_COMPONENTS = 6
LABELS = ["A", "B"]
TAG = "classical_rbf_svm_5fold_n300"

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Could not find training data: {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)
    label_col = "label" if "label" in df.columns else next(
        (c for c in df.columns if c.lower() in {"target", "class", "y"}), None
    )
    if label_col is None:
        raise ValueError(f"Could not identify label column. Columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c.startswith("x")]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != label_col and c.lower() not in {"sample_id", "id"}]
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[label_col].astype(str).to_numpy()

    idx_path = TABLES / "subset_idx_zz_ring_k6_r1_n300.npy"
    if idx_path.exists():
        subset_idx = np.asarray(np.load(idx_path), dtype=int)
        if len(subset_idx) != SUBSET_SIZE:
            raise ValueError(f"{idx_path} has {len(subset_idx)} indices, expected {SUBSET_SIZE}.")
        print(f"Using saved Q2 subset indices: {idx_path}", flush=True)
    else:
        # Recreate the SAME balanced subset as run_step6_noisy_test.py (shared definition in src/q2_data.py).
        subset_idx = stratified_subset((y_all == "B").astype(int), SUBSET_SIZE, SUBSET_SEED)
        np.save(idx_path, subset_idx)
        print(f"Saved recreated subset indices: {idx_path}", flush=True)

    X = X_all[subset_idx]
    y = y_all[subset_idx]
    print(f"Rows: {len(y)} | features: {X.shape[1]} | class counts: {dict(pd.Series(y).value_counts())}", flush=True)
    print("Running 5-fold classical RBF-SVM CV (no quantum circuit simulation)...", flush=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.empty(len(y), dtype=object)
    fold_rows = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(N_COMPONENTS, X.shape[1]), random_state=42)),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        oof_pred[va] = pred

        acc = accuracy_score(y[va], pred)
        bal_acc = balanced_accuracy_score(y[va], pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y[va], pred, labels=LABELS, average="binary", pos_label="B", zero_division=0
        )
        fold_rows.append({
            "fold": fold, "n_train": len(tr), "n_validation": len(va),
            "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
            "precision_B": float(p), "recall_B": float(r), "f1_B": float(f1)
        })
        print(
            f"Fold {fold}/{N_SPLITS}: accuracy={acc:.4f}, balanced_accuracy={bal_acc:.4f}, "
            f"precision_B={p:.4f}, recall_B={r:.4f}, f1_B={f1:.4f}",
            flush=True
        )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y, oof_pred, labels=LABELS, average="macro", zero_division=0
    )
    precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(
        y, oof_pred, labels=LABELS, average="binary", pos_label="B", zero_division=0
    )
    summary = {
        "tag": TAG,
        "model": "RBF-SVM",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "preprocessing": "StandardScaler + PCA(6), fitted separately inside each fold",
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "subset_size": int(len(y)),
        "subset_indices_file": str(idx_path.relative_to(ROOT)),
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "positive_class_for_binary_metrics": "B",
        "accuracy": float(accuracy_score(y, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "precision_B": float(precision_b),
        "recall_B": float(recall_b),
        "f1_B": float(f1_b),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "confusion_matrix_labels_A_B": confusion_matrix(y, oof_pred, labels=LABELS).tolist(),
        "note": "Out-of-fold CV metrics on the selected 300 training rows; these are not hidden-test metrics."
    }

    json_path = TABLES / f"{TAG}_metrics.json"
    folds_path = TABLES / f"{TAG}_fold_metrics.csv"
    pred_path = TABLES / f"{TAG}_oof_predictions.csv"
    report_path = TABLES / f"{TAG}_classification_report.txt"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    pd.DataFrame({
        "original_train_row_index": subset_idx,
        "true_label": y,
        "oof_predicted_label": oof_pred
    }).to_csv(pred_path, index=False)
    report_path.write_text(
        classification_report(y, oof_pred, labels=LABELS, zero_division=0),
        encoding="utf-8"
    )

    print("\n=== OVERALL OUT-OF-FOLD METRICS ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nSaved files:", flush=True)
    print(json_path, flush=True)
    print(folds_path, flush=True)
    print(pred_path, flush=True)
    print(report_path, flush=True)

if __name__ == "__main__":
    main()
