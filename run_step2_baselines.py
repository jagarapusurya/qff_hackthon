import sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.q2_data import load_q2
from src.baselines import tune_baselines, make_model, cv_score, cv_splitter

X, y, Xte, ids = load_q2()
cv = cv_splitter()
t0 = time.perf_counter()
df = tune_baselines(X, y, cv=cv)
print(f"grid done in {time.perf_counter()-t0:.0f}s, {len(df)} configs")
df.to_csv("results/tables/classical_grid.csv", index=False)
best = df.sort_values("cv_mean", ascending=False).groupby(["family", "dim"]).head(1)
print(best.sort_values("cv_mean", ascending=False).head(14).round(4).to_string(index=False))
# organizer-style reference: plain RBF-SVM on standardised 50 features (default C, gamma)
m, s, _ = cv_score(make_model("rbf", 1.0), X, y, cv)
print(f"\nOrganizer-style RBF (C=1, gamma=scale, all 50): {m:.4f} +/- {s:.4f}")
