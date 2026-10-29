"""Resumable 300-sample noisy ZZ-ring kernel + predictions for official Q2 test set.

Run from the project root:
    python run_step6_noisy_test.py

Uses QFF2026-v2 unchanged, 4096 shots, k=6, reps=1, s=0.35, ring topology.
The 300-sample training subset is stratified and fixed by SEED=42. Progress is
checkpointed after each batch so the run can be interrupted and resumed safely.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.q2_data import load_q2, to_labels, stratified_subset
from src.qcv import encode
from src.qkernel import feature_map_ops, fidelity_circuit
from src.noisy_kernel import compile_noisy, SHOTS
from noise_model.benchmark_noise_model import noisy_simulator, VERSION, SEED as NOISE_SEED

SEED = 42
N_SUB = 300
K_QUBITS = 6
REPS = 1
SCALE = 0.35
TOPOLOGY = "ring"
MAP_NAME = "zz"
C_VALUE = 1.0  # frozen from the existing 120-sample noisy CV study; do not tune on test data
BATCH_SIZE = 250
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "zz_ring_k6_r1_n300"


def _checkpoint_paths(kind: str):
    return OUT / f"K_{kind}_{TAG}.npy", OUT / f"K_{kind}_{TAG}_done.npy"


def _run_checkpointed(ops, ZA, ZB, *, kind: str, symmetric: bool):
    """Evaluate kernel pairs in resumable batches; writes K and done mask each batch."""
    n_a, n_b = len(ZA), len(ZB)
    k_path, d_path = _checkpoint_paths(kind)
    if k_path.exists() and d_path.exists():
        K = np.load(k_path)
        done = np.load(d_path)
        if K.shape != (n_a, n_b) or done.shape != (n_a, n_b):
            raise RuntimeError(f"Checkpoint shape mismatch for {kind}; move/delete {k_path.name} and {d_path.name} to restart")
        print(f"Resuming {kind}: {int(done.sum())} / {n_a*n_b} matrix cells already checkpointed", flush=True)
    else:
        K = np.zeros((n_a, n_b), dtype=np.float32)
        done = np.zeros((n_a, n_b), dtype=bool)
        np.save(k_path, K); np.save(d_path, done)

    pairs = []
    for i in range(n_a):
        j_start = i if symmetric else 0
        for j in range(j_start, n_b):
            if not done[i, j]:
                pairs.append((i, j))
    sim = noisy_simulator(seed=NOISE_SEED, method="density_matrix")
    t0 = time.perf_counter()
    total = len(pairs)
    for start in range(0, total, BATCH_SIZE):
        batch = pairs[start:start+BATCH_SIZE]
        circuits = compile_noisy([fidelity_circuit(ops, K_QUBITS, ZA[i], ZB[j]) for i, j in batch])
        result = sim.run(circuits, shots=SHOTS).result()
        zero = "0" * K_QUBITS
        for idx, (i, j) in enumerate(batch):
            value = result.get_counts(idx).get(zero, 0) / SHOTS
            K[i, j] = value
            done[i, j] = True
            if symmetric:
                K[j, i] = value
                done[j, i] = True
        np.save(k_path, K)
        np.save(d_path, done)
        completed = int(done.sum())
        print(f"{kind}: {completed}/{n_a*n_b} cells complete; batch {min(start+BATCH_SIZE,total)}/{total} pending pairs; elapsed {time.perf_counter()-t0:.0f}s", flush=True)
    return K


def main():
    X, y, X_test, sample_ids = load_q2()
    # 150 from each class; indices sorted to make the selected subset deterministic.
    subset = stratified_subset(y, N_SUB, SEED)
    np.save(OUT / f"subset_idx_{TAG}.npy", subset)
    # Fit scaling/PCA on labeled training data only. Test features are transformed, never used to fit.
    Z_train_all, Z_test = encode(X, [X, X_test], K_QUBITS)
    Z_train = Z_train_all[subset]
    y_sub = y[subset]
    ops = feature_map_ops(MAP_NAME, K_QUBITS, REPS, SCALE, TOPOLOGY)

    meta_path = OUT / f"noisy_run_{TAG}.json"
    meta = {
        "tag": TAG, "map": MAP_NAME, "topology": TOPOLOGY, "k": K_QUBITS,
        "reps": REPS, "s": SCALE, "n_sub": N_SUB, "subset_idx_seed": SEED,
        "class_counts": {"A": int(np.sum(y_sub == 0)), "B": int(np.sum(y_sub == 1))},
        "shots": SHOTS, "noise_model_version": VERSION, "C_frozen_from_existing_noisy_CV": C_VALUE,
        "train_kernel_expected_circuits": N_SUB * (N_SUB + 1) // 2,
        "test_kernel_expected_circuits": len(X_test) * N_SUB,
        "note": "Train subset and official test cross-kernel; PCA/scaler fitted on all labeled training features only. Test labels are not accessed."
    }
    K_train = _run_checkpointed(ops, Z_train, Z_train, kind="noisy_train", symmetric=True)
    K_test = _run_checkpointed(ops, Z_test, Z_train, kind="noisy_test", symmetric=False)

    # Sanity checks before generating predictions.
    if not np.isfinite(K_train).all() or not np.isfinite(K_test).all():
        raise RuntimeError("Kernel contains non-finite values")
    if not np.allclose(K_train, K_train.T, atol=1e-7):
        raise RuntimeError("Training kernel is not symmetric")
    if not np.all(np.load(_checkpoint_paths("noisy_train")[1])):
        raise RuntimeError("Training kernel checkpoint is incomplete")
    if not np.all(np.load(_checkpoint_paths("noisy_test")[1])):
        raise RuntimeError("Test kernel checkpoint is incomplete")

    model = SVC(kernel="precomputed", C=C_VALUE, class_weight=None)
    model.fit(K_train, y_sub)
    pred = model.predict(K_test)
    out = pd.DataFrame({"sample_id": sample_ids, "label": to_labels(pred)})
    pred_path = ROOT / "predictions_q2.csv"
    out.to_csv(pred_path, index=False)
    meta.update({
        "train_kernel_shape": list(K_train.shape), "test_kernel_shape": list(K_test.shape),
        "train_kernel_mean_diagonal": float(np.mean(np.diag(K_train))),
        "train_kernel_mean_offdiagonal": float(K_train[~np.eye(N_SUB, dtype=bool)].mean()),
        "train_kernel_offdiagonal_std": float(K_train[~np.eye(N_SUB, dtype=bool)].std()),
        "prediction_counts": {str(k): int(v) for k, v in out.label.value_counts().items()},
        "prediction_file": str(pred_path.name), "status": "complete",
        "completed_at_epoch": time.time()
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print("\nNoisy subset prediction run complete.")
    print(json.dumps(meta, indent=2))
    print(f"Predictions written to {pred_path}")

if __name__ == "__main__":
    main()
