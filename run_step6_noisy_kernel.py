"""Build ideal + noisy kernels on a stratified training subset for one feature-map config.
usage: python run_step6_noisy_kernel.py <map> <topology> <k> <reps> <s> <tag> [n_sub]"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from src.q2_data import load_q2
from src.qcv import encode
from src.qkernel import feature_map_ops, batched_states, kernel_from_states, fidelity_circuit
from src.noisy_kernel import noisy_kernel, compile_noisy

name, topo, k, reps, s, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
n_sub = int(sys.argv[7]) if len(sys.argv) > 7 else 120
SEED = 42

X, y, _, _ = load_q2()
rng = np.random.default_rng(SEED)
sub = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], n_sub // 2, replace=False) for c in (0, 1)]))
Z_all = encode(X, [X], k)[0]                       # scaler+PCA fitted on the 800 TRAIN samples only
Z = Z_all[sub]
ops = feature_map_ops(name, k, reps, s, topo)

S = batched_states(ops, Z, k); K_ideal = kernel_from_states(S, S)
K_noisy, st = noisy_kernel(ops, k, Z, Z, symmetric=True)
np.save(f"results/tables/K_ideal_{tag}.npy", K_ideal); np.save(f"results/tables/K_noisy_{tag}.npy", K_noisy)
np.save(f"results/tables/subset_idx_{tag}.npy", sub)
# resources of one representative circuit
c = compile_noisy([fidelity_circuit(ops, k, Z[0], Z[1])])[0]
st.update(dict(tag=tag, map=name, topology=topo, k=k, reps=reps, s=s, n_sub=n_sub, subset_idx_seed=SEED,
               fidelity_circuit_depth=c.depth(), fidelity_circuit_cx=c.count_ops().get("cx", 0),
               fidelity_circuit_ops=dict(c.count_ops())))
json.dump(st, open(f"results/tables/noisy_run_{tag}.json", "w"), indent=2, default=float)
print("DONE", st)
