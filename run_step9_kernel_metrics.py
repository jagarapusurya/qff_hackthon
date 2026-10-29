"""Evaluation metrics for the ZZ-ring k=6 quantum kernel: ideal vs noisy vs classical, on IDENTICAL folds.

Run from the project root (no new noisy circuits; a few seconds):
    python run_step9_kernel_metrics.py

* Ideal kernel: fold-correct 5-fold CV on the same 300-sample subset and the same StratifiedKFold(seed 42)
  as run_step7_noisy_cv_metrics.py, C=1 frozen (same as the noisy run). Gives paired per-fold differences.
* Noisy kernel: read from the saved step-7 outputs (216,600 QFF2026-v2 circuits; not recomputed).
* Classical RBF-SVM: read from the saved step-7 classical outputs (same folds), plus RBF and polynomial SVMs
  tuned by an inner grid search on each training fold only (nested CV, same outer folds).
* 95% bootstrap CIs (2000 resamples, seed 42) of balanced accuracy and of paired model differences.
* Signal loss: saved 300x300 noisy train kernel vs the exact ideal kernel on the same encoded points.
* Context: ideal kernel and classical RBF on all 800 training samples (5-fold, same protocol).
No noise mitigation is applied: the loss is reported as a negative result.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.q2_data import load_q2, stratified_subset
from src.qcv import encode
from src.qkernel import feature_map_ops, batched_states, kernel_from_states
from src.noise_analysis import matrix_stats

SEED, N_SUB, K, REPS, S, TOPO, NAME, C_VALUE = 42, 300, 6, 1, 0.35, "ring", "zz", 1.0
OUT = ROOT / "results" / "tables"
TAG = "zz_ring_k6_r1_n300"


def metrics(y, pred, score=None):
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
             precision_B=precision_score(y, pred, zero_division=0), recall_B=recall_score(y, pred, zero_division=0),
             f1_B=f1_score(y, pred, zero_division=0), f1_macro=f1_score(y, pred, average="macro", zero_division=0),
             confusion_matrix_labels_A_B=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def ideal_cv(X, y, cv, ops):
    pred, score, folds = np.zeros(len(y), int), np.zeros(len(y)), []
    for tr, va in cv.split(X, y):
        ztr, zva = encode(X[tr], [X[tr], X[va]], K)
        Str, Sva = batched_states(ops, ztr, K), batched_states(ops, zva, K)
        m = SVC(kernel="precomputed", C=C_VALUE).fit(kernel_from_states(Str, Str), y[tr])
        Kva = kernel_from_states(Sva, Str)
        pred[va], score[va] = m.predict(Kva), m.decision_function(Kva)
        folds.append(balanced_accuracy_score(y[va], pred[va]))
    return pred, score, np.array(folds)


def rbf_cv(X, y, cv):
    pred, score, folds = np.zeros(len(y), int), np.zeros(len(y)), []
    for tr, va in cv.split(X, y):
        m = make_pipeline(StandardScaler(), PCA(K, random_state=SEED), SVC(kernel="rbf", C=1.0, gamma="scale"))
        m.fit(X[tr], y[tr])
        pred[va], score[va] = m.predict(X[va]), m.decision_function(X[va])
        folds.append(balanced_accuracy_score(y[va], pred[va]))
    return pred, score, np.array(folds)


def tuned_cv(X, y, cv):
    """Classical kernels tuned on each outer TRAINING fold only (inner 5-fold grid search), then scored on the
    untouched outer validation fold. Same grid as run_step2_baselines.py for PCA-6."""
    grids = {"rbf": {"svc__C": [0.3, 1, 3, 10, 30], "svc__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
             "poly": {"svc__C": [0.3, 1, 10], "svc__degree": [2, 3]}}
    out = {}
    for kern, grid in grids.items():
        pred, score, folds, chosen = np.zeros(len(y), int), np.zeros(len(y)), [], []
        for tr, va in cv.split(X, y):
            pipe = make_pipeline(StandardScaler(), PCA(K, random_state=SEED),
                                 SVC(kernel=kern, coef0=1.0 if kern == "poly" else 0.0))
            inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy").fit(X[tr], y[tr])
            pred[va], score[va] = gs.predict(X[va]), gs.decision_function(X[va])
            folds.append(balanced_accuracy_score(y[va], pred[va]))
            chosen.append({k.replace("svc__", ""): v for k, v in gs.best_params_.items()})
        out[kern] = (pred, score, np.array(folds), chosen)
    return out


def bootstrap(y, preds, n_boot=2000, seed=SEED):
    """95% percentile CIs of balanced accuracy per model and of paired differences (same resampled rows)."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    stats = {n: [] for n in names}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        for n in names:
            stats[n].append(balanced_accuracy_score(y[i], preds[n][i]))
    stats = {n: np.array(v) for n, v in stats.items()}
    ci = {n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for n, v in stats.items()}
    diffs = {}
    for a, b in [("noisy", "ideal"), ("noisy", "rbf_tuned"), ("ideal", "rbf_tuned"), ("noisy", "rbf_default"),
                 ("noisy", "poly_tuned"), ("ideal", "poly_tuned")]:
        d = stats[a] - stats[b]
        diffs[f"{a}_minus_{b}"] = dict(mean=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                       p_le_0=float((d <= 0).mean()))
    return ci, diffs


def main():
    X, y, _, _ = load_q2()
    sub = stratified_subset(y, N_SUB, SEED)
    saved = np.load(OUT / f"subset_idx_{TAG}.npy")
    if not np.array_equal(sub, saved):
        raise RuntimeError("Saved 300-sample subset differs from the shared definition; regenerate it first")
    Xs, ys = X[sub], y[sub]
    ops = feature_map_ops(NAME, K, REPS, S, TOPO)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # identical to both step-7 scripts

    # --- 300-subset, identical folds
    p_i, s_i, f_i = ideal_cv(Xs, ys, cv, ops)
    noisy_oof = pd.read_csv(OUT / f"noisy_cv_oof_predictions_{TAG}.csv")
    cls_oof = pd.read_csv(OUT / "classical_rbf_svm_5fold_n300_oof_predictions.csv")
    for d in (noisy_oof, cls_oof):
        if not np.array_equal(d.original_train_row_index.to_numpy(), sub):
            raise RuntimeError("Saved OOF predictions are not on the expected subset/order")
    p_n = (noisy_oof.oof_predicted_label == "B").astype(int).to_numpy()
    p_c = (cls_oof.oof_predicted_label == "B").astype(int).to_numpy()
    f_n = pd.read_csv(OUT / f"noisy_cv_fold_metrics_{TAG}.csv").balanced_accuracy.to_numpy()
    f_c = pd.read_csv(OUT / "classical_rbf_svm_5fold_n300_fold_metrics.csv").balanced_accuracy.to_numpy()

    rows = []
    for name, p, sc, f in [("Quantum ZZ-ring kernel, ideal (statevector)", p_i, s_i, f_i),
                           ("Quantum ZZ-ring kernel, noisy (QFF2026-v2, 4096 shots)", p_n, None, f_n),
                           ("Classical RBF-SVM (PCA-6, C=1, gamma=scale)", p_c, None, f_c)]:
        rows.append(dict(model=name, n=len(ys), **metrics(ys, p, sc),
                         fold_bal_acc_mean=float(f.mean()), fold_bal_acc_std=float(f.std())))
    tuned = tuned_cv(Xs, ys, cv)
    for kern, label in [("rbf", "Classical RBF-SVM, tuned in each training fold (nested CV)"),
                        ("poly", "Classical poly-SVM, tuned in each training fold (nested CV)")]:
        p, sc, f, chosen = tuned[kern]
        rows.append(dict(model=label, n=len(ys), **metrics(ys, p, sc), fold_bal_acc_mean=float(f.mean()),
                         fold_bal_acc_std=float(f.std()), chosen_params=json.dumps(chosen)))
    ci, diffs = bootstrap(ys, {"ideal": p_i, "noisy": p_n, "rbf_default": p_c, "rbf_tuned": tuned["rbf"][0],
                               "poly_tuned": tuned["poly"][0]})
    for r, key in zip(rows, ["ideal", "noisy", "rbf_default", "rbf_tuned", "poly_tuned"]):
        if key:
            r["bal_acc_ci95"] = ci[key]
    d = f_n - f_i
    paired = dict(bootstrap_paired_differences=diffs,noisy_minus_ideal_mean=float(d.mean()), noisy_minus_ideal_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                  per_fold_ideal=f_i.tolist(), per_fold_noisy=f_n.tolist(), per_fold_diff=d.tolist(),
                  noisy_vs_ideal_prediction_agreement=float((p_n == p_i).mean()))

    # --- signal loss in the saved 300x300 noisy training kernel (encoding fitted on all 800 train rows, as used)
    Z = encode(X, [X], K)[0][sub]
    S_ = batched_states(ops, Z, K)
    Kn = np.load(OUT / f"K_noisy_train_{TAG}.npy").astype(float)
    loss = matrix_stats(kernel_from_states(S_, S_), Kn, ys)

    # --- context: all 800 training samples, same protocol
    cv800 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p8, s8, f8 = ideal_cv(X, y, cv800, ops)
    pr8, sr8, fr8 = rbf_cv(X, y, cv800)
    full = [dict(model="Quantum ZZ-ring kernel, ideal, n=800", n=len(y), **metrics(y, p8, s8),
                 fold_bal_acc_mean=float(f8.mean()), fold_bal_acc_std=float(f8.std())),
            dict(model="Classical RBF-SVM (PCA-6, C=1, gamma=scale), n=800", n=len(y), **metrics(y, pr8, sr8),
                 fold_bal_acc_mean=float(fr8.mean()), fold_bal_acc_std=float(fr8.std()))]

    tab = pd.DataFrame(rows + full)
    tab.drop(columns="confusion_matrix_labels_A_B").to_csv(OUT / f"kernel_metrics_{TAG}.csv", index=False)
    out = dict(subset_n300_same_folds=rows, paired_noise_effect=paired, kernel_signal_loss_n300=loss,
               all_800_context=full, C=C_VALUE, cv="StratifiedKFold(5, shuffle=True, random_state=42)",
               mitigation="none (noise loss reported as a negative result)")
    (OUT / f"kernel_metrics_{TAG}.json").write_text(json.dumps(out, indent=2))
    print(tab.drop(columns="confusion_matrix_labels_A_B").round(4).to_string(index=False))
    print(json.dumps(paired, indent=1))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in loss.items()}, indent=1))


if __name__ == "__main__":
    main()
