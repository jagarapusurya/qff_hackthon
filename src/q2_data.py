"""PS-Q2 data loading. All fitting (scaler, PCA) happens on training data only."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data" / "q2"
LABELS = ("A", "B")


def load_q2():
    """Returns X_train (800,50), y_train (0/1 for A/B), X_test (200,50), test sample_ids."""
    tr = pd.read_csv(DATA / "hd_data_train.csv")
    te = pd.read_csv(DATA / "hd_data_test_features.csv")
    feat = [c for c in tr.columns if c.startswith("x")]
    y = (tr["label"] == "B").astype(int).to_numpy()
    return tr[feat].to_numpy(float), y, te[feat].to_numpy(float), te["sample_id"].to_numpy()


def to_labels(y01):
    return np.where(np.asarray(y01) == 1, "B", "A")


def stratified_subset(y01, n_sub, seed=SEED):
    """Balanced training subset (n_sub/2 per class), sorted. The single definition used by every n_sub run."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(np.asarray(y01) == c), n_sub // 2, replace=False)
                                   for c in (0, 1)]))
