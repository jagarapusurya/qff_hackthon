"""Ideal-vs-noisy kernel analysis and paired CV on a stratified subset."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from .qkernel import offdiag_stats, target_alignment

SEED = 42
C_WIDE = (0.1, 0.3, 1, 3, 10, 30, 100, 300)


def matrix_stats(K_ideal, K_noisy, y):
    n = K_ideal.shape[0]; m = ~np.eye(n, dtype=bool)
    a, b = K_ideal[m], K_noisy[m]
    slope, intercept = np.polyfit(a, b, 1)
    resid = b - (slope * a + intercept)
    r2 = 1 - resid.var() / b.var()
    rho = spearmanr(a, b).correlation
    return {
        "n": n,
        "diag_ideal_mean": float(np.diag(K_ideal).mean()), "diag_noisy_mean": float(np.diag(K_noisy).mean()),
        "offdiag_mean_ideal": float(a.mean()), "offdiag_mean_noisy": float(b.mean()),
        "offdiag_std_ideal": float(a.std()), "offdiag_std_noisy": float(b.std()),
        "frobenius_rel_error_raw": float(np.linalg.norm(K_noisy - K_ideal) / np.linalg.norm(K_ideal)),
        "mean_abs_error_raw": float(np.abs(K_noisy - K_ideal)[m].mean()),
        "affine_slope": float(slope), "affine_intercept": float(intercept), "affine_R2": float(r2),
        "spearman_rho": float(rho),
        "rel_error_after_affine_fit": float(np.linalg.norm(resid) / np.linalg.norm(b - b.mean())),
        "alignment_ideal": target_alignment(K_ideal, y), "alignment_noisy": target_alignment(K_noisy, y),
    }


def cv_precomputed(K, y, cv, Cs=C_WIDE):
    """Per-fold balanced accuracy for each C, then pick the C with best mean (documented tuning)."""
    per = {C: [] for C in Cs}
    for tr, va in cv.split(K, y):
        for C in Cs:
            p = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr]).predict(K[np.ix_(va, tr)])
            per[C].append(balanced_accuracy_score(y[va], p))
    means = {C: np.mean(v) for C, v in per.items()}
    bc = max(means, key=means.get)
    return bc, np.array(per[bc])


def cv_rbf(Z, y, cv):
    best = None
    for C in (0.3, 1, 3, 10, 30):
        for g in (0.003, 0.01, 0.03, 0.1, "scale"):
            sc = []
            for tr, va in cv.split(Z, y):
                p = SVC(kernel="rbf", C=C, gamma=g).fit(Z[tr], y[tr]).predict(Z[va])
                sc.append(balanced_accuracy_score(y[va], p))
            if best is None or np.mean(sc) > best[0]:
                best = (np.mean(sc), C, g, np.array(sc))
    return best[1:3], best[3]


def paired_summary(name, scores, ref=None):
    r = {"model": name, "cv_mean": scores.mean(), "cv_std": scores.std(), "n_folds": len(scores)}
    if ref is not None:
        d = scores - ref
        r["diff_vs_ref_mean"] = d.mean(); r["diff_vs_ref_se"] = d.std(ddof=1) / np.sqrt(len(d))
    return r


def subset_cv_table(K_ideal, K_noisy, Z_sub, y, n_splits=5, n_repeats=6):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    C_i, s_i = cv_precomputed(K_ideal, y, cv)
    C_n, s_n = cv_precomputed(K_noisy, y, cv)
    (C_r, g_r), s_r = cv_rbf(Z_sub, y, cv)
    rows = [paired_summary("Classical RBF-SVM (same PCA-6 features)", s_r),
            paired_summary("Quantum kernel, ideal simulation", s_i, s_r),
            paired_summary("Quantum kernel, noisy (QFF2026-v2)", s_n, s_r)]
    for r, extra in zip(rows, [f"C={C_r}, gamma={g_r}", f"C={C_i}", f"C={C_n}"]):
        r["hyperparameters"] = extra
    rows.append(dict(model="Noisy minus ideal (paired)", cv_mean=(s_n - s_i).mean(),
                     cv_std=(s_n - s_i).std(), n_folds=len(s_n), diff_vs_ref_mean=(s_n - s_i).mean(),
                     diff_vs_ref_se=(s_n - s_i).std(ddof=1) / np.sqrt(len(s_n)), hyperparameters=""))
    return pd.DataFrame(rows)
