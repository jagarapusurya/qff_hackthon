"""Noisy fidelity-kernel evaluation under the organizer noise model QFF2026-v2.
Every circuit is transpiled to the fully-noised basis (rx, ry, rz, cx; opt level 1, seed 42) and checked with
assert_all_gates_noisy() before it runs. Simulator: Aer density-matrix (method accepted by noisy_simulator())."""
import time
import numpy as np
from qiskit import transpile
from noise_model.benchmark_noise_model import (noisy_simulator, assert_all_gates_noisy, DEFAULT_BASIS, SEED)
from .qkernel import fidelity_circuit

SHOTS = 4096


def compile_noisy(circs):
    out = transpile(circs, basis_gates=DEFAULT_BASIS, optimization_level=1, seed_transpiler=SEED)
    for c in out:
        assert_all_gates_noisy(c)
    return out


def noisy_kernel(ops, k, ZA, ZB, symmetric=False, shots=SHOTS, seed=SEED, chunk=500, verbose=True):
    """K[i,j] ~ P(0..0) of U(zB_j)^dag U(zA_i) under noise. symmetric=True fills K from the upper triangle
    (diagonal INCLUDED, measured like every other entry)."""
    na, nb = len(ZA), len(ZB)
    idx = [(i, j) for i in range(na) for j in range(i if symmetric else 0, nb)]
    sim = noisy_simulator(seed, method="density_matrix")
    K = np.zeros((na, nb)); zero = "0" * k
    t0 = time.perf_counter(); stats = {"depth": [], "cx": [], "size": []}
    for c0 in range(0, len(idx), chunk):
        part = idx[c0:c0 + chunk]
        circs = compile_noisy([fidelity_circuit(ops, k, ZA[i], ZB[j]) for i, j in part])
        for c in circs:
            stats["depth"].append(c.depth()); stats["cx"].append(c.count_ops().get("cx", 0)); stats["size"].append(c.size())
        res = sim.run(circs, shots=shots).result()
        for n, (i, j) in enumerate(part):
            K[i, j] = res.get_counts(n).get(zero, 0) / shots
            if symmetric:
                K[j, i] = K[i, j]
        if verbose:
            print(f"  {c0 + len(part)}/{len(idx)} circuits, {time.perf_counter() - t0:.0f}s", flush=True)
    return K, dict(n_circuits=len(idx), seconds=time.perf_counter() - t0, shots=shots,
                   mean_depth=float(np.mean(stats["depth"])), mean_cx=float(np.mean(stats["cx"])),
                   mean_ops=float(np.mean(stats["size"])))
