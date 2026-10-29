"""Benchmark candidate Processor D topologies with the official QFT-3 benchmark
and the project's 6-qubit ZZ-ring fidelity circuit. Run from project root:
    python run_processor_d_sweep.py
Requires the pinned QFF2026 environment.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from qiskit import transpile
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))
from benchmark.qff_qml_benchmark import task_b_qft3, validate_processor_d
from src.qkernel import feature_map_ops, fidelity_circuit
from src.qcv import encode
from src.q2_data import load_q2
from noise_model.benchmark_noise_model import DEFAULT_BASIS, SEED, assert_all_gates_noisy

N = 6
BASIS = ["rx", "ry", "rz", "cx"]
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def undirected_edges(topology):
    if topology == "linear": return [(i, i+1) for i in range(N-1)]
    if topology == "ring": return [(i, (i+1) % N) for i in range(N)]
    if topology == "star": return [(0, i) for i in range(1, N)]
    if topology == "grid2x3": return [(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5)]
    raise ValueError(topology)


def config(topology):
    edges = undirected_edges(topology)
    directed = [[a,b] for a,b in edges] + [[b,a] for a,b in edges]
    return {
      "processor_id": f"qff26_procD_{topology}_6q_v1", "n_qubits": N,
      "coupling_map": directed, "native_gate_set": BASIS,
      "qubit_layout": topology if topology in {"linear", "ring", "star", "grid"} else "custom",
      "noise_assumptions": {"single_qubit_error_rate":0.01,"two_qubit_error_rate":0.05,
        "readout_error_rate":0.02,"t1_us":100,"t2_us":80,
        "notes":"Reported assumptions for design context only. The official Task B uses QFF2026-v2 unchanged."},
      "design_rationale": "A six-qubit bidirectional topology supports the six-qubit Q2 feature map with direct nearest-neighbour links. The rx/ry/rz/cx native gate set is universal and matches the organizer's fully-noised basis. Two-way links avoid direction-reversal overhead during routing. This candidate is selected by comparing the official QFT-3 noisy fidelity and the compiled depth and CX cost of the Q2 ring fidelity circuit; the benchmark noise model is unchanged.",
      "benchmark_results": {}
    }


def main():
    X, y, _, _ = load_q2()
    Z = encode(X, [X], 6)[0]
    ops = feature_map_ops("zz", 6, 1, 0.35, "ring")
    # Representative fidelity circuit uses the same six-qubit ring feature map.
    sample_circ = fidelity_circuit(ops, 6, Z[0], Z[1])
    rows=[]; configs={}
    for topology in ("linear", "ring", "star", "grid2x3"):
        cfg=config(topology)
        errs=validate_processor_d(cfg, require_results=False)
        if errs: raise RuntimeError(f"{topology} Processor D config invalid: {errs}")
        qft=task_b_qft3(cfg)
        directed=cfg["coupling_map"]
        from qiskit.transpiler import CouplingMap
        cm=CouplingMap(directed)
        compiled=transpile(sample_circ, basis_gates=BASIS, coupling_map=cm,
                           optimization_level=1, seed_transpiler=SEED)
        assert_all_gates_noisy(compiled)
        row={"topology":topology, **qft,
             "q2_ring_depth_after_transpile":int(compiled.depth()),
             "q2_ring_cx_count":int(compiled.count_ops().get("cx",0)),
             "q2_ring_total_ops":int(compiled.size()),
             "directed_coupling_edges":len(directed)}
        rows.append(row); configs[topology]=cfg
        print(json.dumps(row, indent=2), flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT / "processor_d_topology_sweep.csv", index=False)
    # Prioritize noisy QFT-3 fidelity; tie-break on lower Q2 circuit depth, then CX count.
    best=df.sort_values(["qft3_fidelity_noisy","q2_ring_depth_after_transpile","q2_ring_cx_count"],
                        ascending=[False,True,True]).iloc[0]
    selected=configs[best["topology"]]
    selected["benchmark_results"]={k:best[k].item() if hasattr(best[k],"item") else best[k] for k in (
       "qft3_circuit_depth_before_transpile","qft3_circuit_depth_after_transpile",
       "qft3_two_qubit_gate_count","qft3_swap_overhead","qft3_fidelity_ideal",
       "qft3_fidelity_noisy","benchmark_noise_model_version")}
    (ROOT / "processor_d_config.json").write_text(json.dumps(selected, indent=2))
    (OUT / "processor_d_topology_sweep.json").write_text(df.to_json(orient="records", indent=2))
    print(f"\nSelected topology: {best['topology']}; wrote processor_d_config.json and results/tables/processor_d_topology_sweep.csv")

if __name__ == "__main__": main()
