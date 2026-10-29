"""Fold-correct 5-fold CV metrics for the noisy QFF2026 Q2 quantum-kernel SVM.

Run from the project root:
    .\.venv\Scripts\python.exe run_step7_noisy_cv_metrics.py

This practical evaluation uses the same fixed, stratified 300-sample training subset
as the final noisy prediction run. For each fold, StandardScaler and PCA are fitted
only on that fold's training samples, and noisy train/validation kernels are freshly
computed using QFF2026-v2. The official test set is never read or used.

Estimated kernel evaluations: 5 * (240*241/2 + 60*240) = 216,600 circuits,
with 4096 shots per circuit. This may take substantial time.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.q2_data import load_q2, to_labels, stratified_subset
from src.qcv import encode
from src.qkernel import feature_map_ops
from src.noisy_kernel import noisy_kernel, SHOTS
from noise_model.benchmark_noise_model import VERSION, SEED as NOISE_SEED

SEED = 42
N_SUB = 300
K_QUBITS = 6
REPS = 1
SCALE = 0.35
TOPOLOGY = "ring"
MAP_NAME = "zz"
C_VALUE = 1.0  # frozen before this CV run; not tuned on these folds
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "zz_ring_k6_r1_n300"


def get_subset(y: np.ndarray) -> np.ndarray:
    path = OUT / f"subset_idx_{TAG}.npy"
    if path.exists():
        subset = np.load(path).astype(int)
        if len(subset) == N_SUB and len(np.unique(subset)) == N_SUB:
            return subset
        raise RuntimeError(f"Subset file {path} has unexpected shape/content")
    subset = stratified_subset(y, N_SUB, SEED)
    np.save(path, subset)
    return subset


def main() -> None:
    X, y, _X_test, _test_ids = load_q2()  # test features are intentionally unused
    subset = get_subset(y)
    Xs, ys = X[subset], y[subset]
    if len(Xs) != N_SUB:
        raise RuntimeError(f"Expected {N_SUB} selected training samples, got {len(Xs)}")

    ops = feature_map_ops(MAP_NAME, K_QUBITS, REPS, SCALE, TOPOLOGY)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = np.full(len(ys), -1, dtype=int)
    fold_rows = []
    started = time.perf_counter()

    print(f"Starting fold-correct noisy quantum-kernel CV")
    print(f"Samples={len(ys)}, class counts={{'A': {int((ys == 0).sum())}, 'B': {int((ys == 1).sum())}}}")
    print(f"Noise model={VERSION}; shots={SHOTS}; C={C_VALUE}; folds=5")
    print(f"Expected circuits: {5 * ((240 * 241 // 2) + (60 * 240)):,}")

    for fold, (tr, va) in enumerate(cv.split(Xs, ys), start=1):
        print(f"\n===== Fold {fold}/5: train={len(tr)}, validation={len(va)} =====", flush=True)
        # Fit preprocessing on this fold's training rows only.
        Ztr, Zva = encode(Xs[tr], [Xs[tr], Xs[va]], K_QUBITS)

        Ktr, train_stats = noisy_kernel(
            ops, K_QUBITS, Ztr, Ztr, symmetric=True, shots=SHOTS,
            seed=NOISE_SEED + fold, verbose=True,
        )
        Kva, val_stats = noisy_kernel(
            ops, K_QUBITS, Zva, Ztr, symmetric=False, shots=SHOTS,
            seed=NOISE_SEED + 100 + fold, verbose=True,
        )

        model = SVC(kernel="precomputed", C=C_VALUE, class_weight=None)
        model.fit(Ktr, ys[tr])
        pred = model.predict(Kva)
        oof_pred[va] = pred
        fold_rows.append({
            "fold": fold,
            "train_n": int(len(tr)),
            "validation_n": int(len(va)),
            "accuracy": float(accuracy_score(ys[va], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(ys[va], pred)),
            "precision_B": float(precision_score(ys[va], pred, pos_label=1, zero_division=0)),
            "recall_B": float(recall_score(ys[va], pred, pos_label=1, zero_division=0)),
            "f1_B": float(f1_score(ys[va], pred, pos_label=1, zero_division=0)),
            "train_kernel_circuits": int(train_stats["n_circuits"]),
            "validation_kernel_circuits": int(val_stats["n_circuits"]),
        })
        print(f"Fold {fold} metrics: {json.dumps(fold_rows[-1], indent=2)}", flush=True)

    if np.any(oof_pred < 0):
        raise RuntimeError("Some samples did not receive an out-of-fold prediction")

    # Label B is the positive class (encoded as 1); also report macro metrics.
    metrics = {
        "evaluation": "5-fold stratified out-of-fold CV on fixed 300-sample training subset",
        "model": "Noisy ZZ-ring fidelity-kernel SVM",
        "feature_map": MAP_NAME,
        "logical_topology": TOPOLOGY,
        "n_qubits": K_QUBITS,
        "reps": REPS,
        "s": SCALE,
        "C": C_VALUE,
        "shots": SHOTS,
        "noise_model_version": VERSION,
        "n_samples": int(len(ys)),
        "class_counts": {"A": int((ys == 0).sum()), "B": int((ys == 1).sum())},
        "positive_class": "B",
        "accuracy": float(accuracy_score(ys, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, oof_pred)),
        "precision_B": float(precision_score(ys, oof_pred, pos_label=1, zero_division=0)),
        "recall_B": float(recall_score(ys, oof_pred, pos_label=1, zero_division=0)),
        "f1_B": float(f1_score(ys, oof_pred, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(ys, oof_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(ys, oof_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(ys, oof_pred, average="macro", zero_division=0)),
        "confusion_matrix_labels_A_B": confusion_matrix(ys, oof_pred, labels=[0, 1]).tolist(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "note": "Fold-specific scaler/PCA; official test features and labels are not used for evaluation. Metrics estimate performance on the selected 300-sample subset, not all 800 training samples or hidden test labels.",
    }

    oof = pd.DataFrame({
        "original_train_row_index": subset,
        "true_label": to_labels(ys),
        "oof_predicted_label": to_labels(oof_pred),
    })
    oof_path = OUT / f"noisy_cv_oof_predictions_{TAG}.csv"
    folds_path = OUT / f"noisy_cv_fold_metrics_{TAG}.csv"
    json_path = OUT / f"noisy_cv_metrics_{TAG}.json"
    report_path = OUT / f"noisy_cv_classification_report_{TAG}.txt"
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2))
    report = classification_report(ys, oof_pred, labels=[0, 1], target_names=["A", "B"], zero_division=0)
    report_path.write_text(report)

    print("\n===== FINAL OUT-OF-FOLD METRICS =====")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report (A/B):")
    print(report)
    print(f"Saved: {json_path}")
    print(f"Saved: {folds_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {report_path}")

if __name__ == "__main__":
    main()
