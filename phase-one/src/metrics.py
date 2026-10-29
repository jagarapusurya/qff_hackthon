"""Circuit metrics (depth, CX, SWAP overhead) and kernel-evaluation cost / fidelity benchmarks."""
import time
import numpy as np
import pandas as pd
from .problem import make_fidelity_circuit, ideal_kernel, N_QUBITS
from .routing import transpile_for_processor, transpile_all_to_all


def circuit_metrics(c):
    ops = c.count_ops()
    return {"Depth": c.depth(), "Total ops": c.size(), "CX": ops.get("cx", 0)}


def compilation_table(qc, processors):
    """Depth/CX of one circuit on an all-to-all reference and each processor, plus routing overhead."""
    basis = processors[0].basis_gates
    rows = [{"Target": "All-to-all (reference)", **circuit_metrics(transpile_all_to_all(qc, basis))}]
    for p in processors:
        rows.append({"Target": p.name, **circuit_metrics(transpile_for_processor(qc, p))})
    df = pd.DataFrame(rows)
    df["Routing CX overhead"] = df["CX"] - df.loc[0, "CX"]
    df["Est. SWAPs (overhead/3)"] = df["Routing CX overhead"] / 3   # one SWAP = 3 CX
    return df


def binomial_standard_error(p, shots):
    """SE of an estimate p_hat of a probability from `shots` samples: sqrt(p(1-p)/N)."""
    return float(np.sqrt(max(p * (1 - p), 0.0) / shots))


def benchmark_kernel_cost(proc, X_train, n_test, n_pairs=30, seed=42):
    """Cost and accuracy of kernel evaluation on one processor, for a fixed set of random train pairs."""
    backend = proc.backend
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, len(X_train), size=(n_pairs, 2))     # same pairs for every processor

    t_comp, t_exec, errs, ses, depth, cx, size = [], [], [], [], [], [], []
    for i, j in pairs:
        qc = make_fidelity_circuit(X_train[i], X_train[j])
        t0 = time.perf_counter()
        c = transpile_for_processor(qc, proc)
        t_comp.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        counts = backend.run(c, shots=proc.shots, seed_simulator=seed).result().get_counts()
        t_exec.append(time.perf_counter() - t1)

        est = counts.get("0" * N_QUBITS, 0) / proc.shots
        errs.append(abs(est - ideal_kernel(X_train[i], X_train[j])))
        ses.append(binomial_standard_error(est, proc.shots))
        m = circuit_metrics(c)
        depth.append(m["Depth"]); cx.append(m["CX"]); size.append(m["Total ops"])

    n = len(X_train)
    n_entries = n * (n + 1) // 2 + n_test * n          # symmetric K_train + K_test
    per_eval = float(np.mean(t_comp) + np.mean(t_exec))
    return {
        "Processor": proc.name, "Physical qubits": proc.n_qubits, "Pairs benchmarked": n_pairs,
        "Mean depth": float(np.mean(depth)), "Mean total ops": float(np.mean(size)),
        "Mean CX gates": float(np.mean(cx)),
        "Mean compile time (s)": float(np.mean(t_comp)),
        "Mean execution time (s)": float(np.mean(t_exec)),
        "Mean time / evaluation (s)": per_eval,
        "Kernel entries needed (sym.)": n_entries,
        "Est. total time (s)": per_eval * n_entries,
        "Total shots needed": n_entries * proc.shots,
        "Mean |noisy - ideal|": float(np.mean(errs)),
        "Max |noisy - ideal|": float(np.max(errs)),
        "Mean binomial std. error": float(np.mean(ses)),
    }
