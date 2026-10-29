"""Tuned classical kernel baselines (RBF / polynomial / linear SVM) with repeated stratified CV.
Scaler and PCA live inside the sklearn Pipeline, so they are re-fitted on each training fold only."""
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def cv_splitter(n_splits=5, n_repeats=3, seed=SEED):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def make_model(kernel, C, gamma=None, degree=3, k_pca=None):
    steps = [StandardScaler()]
    if k_pca:
        steps.append(PCA(k_pca, random_state=SEED))
    steps.append(SVC(kernel=kernel, C=C, gamma=gamma if gamma is not None else "scale",
                     degree=degree, coef0=1.0 if kernel == "poly" else 0.0, random_state=SEED))
    return make_pipeline(*steps)


def cv_score(model, X, y, cv):
    s = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
    return s.mean(), s.std(), s


def tune_baselines(X, y, k_list=(None, 4, 6, 7, 8, 10), cv=None):
    """Grid search per family. Returns a DataFrame with one row per config."""
    cv = cv or cv_splitter()
    rows = []
    for k in k_list:
        dim = "all 50" if k is None else f"PCA-{k}"
        for C, g in itertools.product([0.3, 1, 3, 10, 30], ["scale", 0.003, 0.01, 0.03, 0.1]):
            gi = None if g == "scale" else g
            m, s, _ = cv_score(make_model("rbf", C, gi, k_pca=k), X, y, cv)
            rows.append(dict(family="RBF-SVM", dim=dim, C=C, gamma=g, degree="", cv_mean=m, cv_std=s))
        for C, d in itertools.product([0.3, 1, 10], [2, 3]):
            m, s, _ = cv_score(make_model("poly", C, "scale", degree=d, k_pca=k), X, y, cv)
            rows.append(dict(family="Poly-SVM", dim=dim, C=C, gamma="scale", degree=d, cv_mean=m, cv_std=s))
        for C in [0.03, 0.3, 3]:
            m, s, _ = cv_score(make_model("linear", C, k_pca=k), X, y, cv)
            rows.append(dict(family="Linear-SVM", dim=dim, C=C, gamma="", degree="", cv_mean=m, cv_std=s))
    return pd.DataFrame(rows)
