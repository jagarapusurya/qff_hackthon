"""usage: python run_step6_analysis.py <tag> <k>"""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.q2_data import load_q2
from src.qcv import encode
from src.noise_analysis import matrix_stats, subset_cv_table

tag, k = sys.argv[1], int(sys.argv[2])
X, y, _, _ = load_q2()
sub = np.load(f"results/tables/subset_idx_{tag}.npy")
Ki = np.load(f"results/tables/K_ideal_{tag}.npy"); Kn = np.load(f"results/tables/K_noisy_{tag}.npy")
ys = y[sub]; Zs = encode(X, [X], k)[0][sub]

ms = matrix_stats(Ki, Kn, ys)
json.dump(ms, open(f"results/tables/matrix_stats_{tag}.json", "w"), indent=2)
print(json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in ms.items()}, indent=1))

tab = subset_cv_table(Ki, Kn, Zs, ys)
tab.to_csv(f"results/tables/subset_cv_{tag}.csv", index=False)
print(tab.round(4).to_string(index=False))

# figures
order = np.argsort(ys, kind="stable")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, M, t in [(ax[0], Ki, "Ideal kernel"), (ax[1], Kn, "Noisy kernel (QFF2026-v2)")]:
    im = a.imshow(M[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1); a.set_title(t)
    a.set_xlabel("samples sorted by class (A then B)"); fig.colorbar(im, ax=a, fraction=0.046)
n = len(ys); m = ~np.eye(n, dtype=bool)
ax[2].scatter(Ki[m], Kn[m], s=2, alpha=0.15)
xs = np.linspace(0, 1, 50); ax[2].plot(xs, ms["affine_slope"] * xs + ms["affine_intercept"], "r-",
    label=f"fit: {ms['affine_slope']:.2f}x + {ms['affine_intercept']:.3f}  (R²={ms['affine_R2']:.2f})")
ax[2].set_xlabel("ideal K"); ax[2].set_ylabel("noisy K"); ax[2].set_title("Noisy vs ideal entries"); ax[2].legend()
fig.tight_layout(); fig.savefig(f"results/figures/noise_effect_{tag}.png", dpi=180)
