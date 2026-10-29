"""Fold-correct cross-validation of quantum-kernel SVMs (scaler + PCA fitted on each training fold only)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .qkernel import feature_map_ops, batched_states, kernel_from_states, offdiag_stats, target_alignment

SEED = 42
C_GRID = (0.3, 1, 3, 10, 30)


def encode(X_fit, X_list, k):
    """Fit StandardScaler+PCA(k)+unit-variance rescale on X_fit only; transform each array in X_list."""
    sc = StandardScaler().fit(X_fit)
    pca = PCA(k, random_state=SEED).fit(sc.transform(X_fit))
    sd = pca.transform(sc.transform(X_fit)).std(axis=0)
    return [pca.transform(sc.transform(X)) / sd for X in X_list]


def cv_quantum(name, k, reps, s, topology, X, y, cv, Cs=C_GRID):
    ops = feature_map_ops(name, k, reps, s, topology)
    per_C = {C: [] for C in Cs}
    conc, align = [], []
    for tr, va in cv.split(X, y):
        ztr, zva = encode(X[tr], [X[tr], X[va]], k)
        Str, Sva = batched_states(ops, ztr, k), batched_states(ops, zva, k)
        Ktr, Kva = kernel_from_states(Str, Str), kernel_from_states(Sva, Str)
        conc.append(offdiag_stats(Ktr)[1]); align.append(target_alignment(Ktr, y[tr]))
        for C in Cs:
            p = SVC(kernel="precomputed", C=C).fit(Ktr, y[tr]).predict(Kva)
            per_C[C].append(balanced_accuracy_score(y[va], p))
    means = {C: np.mean(v) for C, v in per_C.items()}
    bestC = max(means, key=means.get)
    return dict(map=name, topology=topology if name != "product" else "-", k=k, reps=reps, s=s, best_C=bestC,
                cv_mean=means[bestC], cv_std=float(np.std(per_C[bestC])),
                offdiag_std=float(np.mean(conc)), alignment=float(np.mean(align)))
