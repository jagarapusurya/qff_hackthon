import sys, time, itertools
sys.path.insert(0, ".")
import pandas as pd
from src.q2_data import load_q2
from src.baselines import cv_splitter
from src.qcv import cv_quantum

X, y, _, _ = load_q2()
cv = cv_splitter(5, 2)
S_GRID = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
specs = [("product", "ring"), ("zz", "ring"), ("zz", "all"), ("reup", "ring")]
rows, t0 = [], time.perf_counter()
for (name, topo), k, reps, s in itertools.product(specs, [4, 6, 8], [1, 2], S_GRID):
    rows.append(cv_quantum(name, k, reps, s, topo, X, y, cv))
df = pd.DataFrame(rows)
df.to_csv("results/tables/quantum_ablation_grid.csv", index=False)
print(f"{len(df)} configs in {time.perf_counter()-t0:.0f}s")
best = df.sort_values("cv_mean", ascending=False).groupby(["map", "topology"]).head(1)
print(best.sort_values("cv_mean", ascending=False).round(4).to_string(index=False))
print("\nTop 8 overall:")
print(df.sort_values("cv_mean", ascending=False).head(8).round(4).to_string(index=False))
