"""Ideal-simulation ZZ-ring quantum-kernel SVM trained on ALL 800 training samples (alternative Q2 predictions).
C is chosen by fold-correct 5x2 CV on the training set only; test features are only transformed/predicted.
Writes predictions_q2_ideal_n800.csv (same format as predictions_q2.csv)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.svm import SVC
from src.q2_data import load_q2, to_labels
from src.baselines import cv_splitter
from src.qcv import cv_quantum, encode
from src.qkernel import feature_map_ops, batched_states, kernel_from_states

K, REPS, S, TOPO, NAME = 6, 1, 0.35, "ring", "zz"
X, y, Xte, ids = load_q2()
cv = cv_quantum(NAME, K, REPS, S, TOPO, X, y, cv_splitter(5, 2))
print("CV (5x2, fold-correct):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cv.items()})
Ztr, Zte = encode(X, [X, Xte], K)
ops = feature_map_ops(NAME, K, REPS, S, TOPO)
Str, Ste = batched_states(ops, Ztr, K), batched_states(ops, Zte, K)
model = SVC(kernel="precomputed", C=cv["best_C"]).fit(kernel_from_states(Str, Str), y)
pred = model.predict(kernel_from_states(Ste, Str))
out = pd.DataFrame({"sample_id": ids, "label": to_labels(pred)})
out.to_csv("predictions_q2_ideal_n800.csv", index=False)
print(out.label.value_counts().to_dict())
json.dump({"cv": cv, "n_train": 800, "counts": out.label.value_counts().to_dict()}, open("results/tables/ideal_n800_final.json", "w"), indent=2, default=float)
