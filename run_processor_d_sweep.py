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
    # Ring plus one chord between qubits two apart: the chord closes a triangle (0-1-2), which the
    # three interacting pairs of QFT-3 (Nielsen & Chuang, Box 5.1) need; a grid/ring/line has none.
    if topology == "ring_chord": return [(i, (i+1) % N) for i in range(N)] + [(0, 2)]
    # Grid plus a diagonal (also closes triangles), for comparison at one extra coupler.
    if topology == "grid_diag": return [(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5),(0,4)]
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
      "design_rationale": "",   # filled in by build_rationale() for the selected topology
      "benchmark_results": {}
    }


def build_rationale(best, df):
    """Design rationale for the selected topology, with every number taken from the sweep table."""
    r = {t: df[df.topology == t].iloc[0] for t in df.topology}
    n_und = len(undirected_edges(best["topology"]))
    q2 = f"{int(best['q2_ring_cx_count'])} CX and depth {int(best['q2_ring_depth_after_transpile'])}"
    txt = (f"Our PS-Q2 kernel circuit is a 6-qubit ZZ fidelity circuit whose RZZ entanglers (CX-RZ-CX) act on a "
           f"logical ring ({q2} in total). ")
    if best["topology"] in ("ring_chord", "grid_diag"):
        txt += ("The topology contains the physical 6-cycle, so the ring is placed with zero SWAPs at the same cost as "
                "all-to-all. A chord closes a triangle, and the textbook QFT-3 (Nielsen and Chuang, Box 5.1) needs exactly "
                "that: H with controlled-S and controlled-T gives interacting pairs (0,1), (0,2), (1,2), and the final "
                "SWAP acts on (0,2). ")
        if "grid2x3" in r and "ring" in r:
            txt += (f"With the triangle the official QFT-3 needs {int(best['qft3_two_qubit_gate_count'])} two-qubit gates, "
                    f"depth {int(best['qft3_circuit_depth_after_transpile'])}, swap overhead {int(best['qft3_swap_overhead'])} "
                    f"and noisy fidelity {best['qft3_fidelity_noisy']:.3f}, versus "
                    f"{int(r['grid2x3']['qft3_two_qubit_gate_count'])} gates, depth {int(r['grid2x3']['qft3_circuit_depth_after_transpile'])}, "
                    f"overhead {int(r['grid2x3']['qft3_swap_overhead'])} and {r['grid2x3']['qft3_fidelity_noisy']:.3f} for the 2x3 grid, and "
                    f"{r['ring']['qft3_fidelity_noisy']:.3f} for the bare ring. A grid, ring, line or star has no triangle, "
                    f"so one QFT pair is never adjacent and routing adds a SWAP (3 CX). ")
    txt += (f"All links are two-way because the benchmark enforces direction and one-way links add reversal gates. "
            f"The rx, ry, rz and cx native gates are the organizer's fully-noised basis, so every compiled gate carries "
            f"QFF2026-v2 noise. The cost is {n_und} couplers ({2*n_und} directed edges), one more than a bare ring.")
    return txt


def main():
    X, y, _, _ = load_q2()
    Z = encode(X, [X], 6)[0]
    ops = feature_map_ops("zz", 6, 1, 0.35, "ring")
    # Representative fidelity circuit uses the same six-qubit ring feature map.
    sample_circ = fidelity_circuit(ops, 6, Z[0], Z[1])
    rows=[]; configs={}
    for topology in ("linear", "ring", "star", "grid2x3", "ring_chord", "grid_diag"):
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
    # Prioritize noisy QFT-3 fidelity; tie-break on lower Q2 circuit depth, then CX count, then fewer couplers.
    best=df.sort_values(["qft3_fidelity_noisy","q2_ring_depth_after_transpile","q2_ring_cx_count",
                         "directed_coupling_edges"], ascending=[False,True,True,True]).iloc[0]
    selected=configs[best["topology"]]
    selected["design_rationale"]=build_rationale(best, df)
    selected["benchmark_results"]={k:best[k].item() if hasattr(best[k],"item") else best[k] for k in (
       "qft3_circuit_depth_before_transpile","qft3_circuit_depth_after_transpile",
       "qft3_two_qubit_gate_count","qft3_swap_overhead","qft3_fidelity_ideal",
       "qft3_fidelity_noisy","benchmark_noise_model_version")}
    (ROOT / "processor_d_config.json").write_text(json.dumps(selected, indent=2))
    (OUT / "processor_d_topology_sweep.json").write_text(df.to_json(orient="records", indent=2))
    print(f"\nSelected topology: {best['topology']}; wrote processor_d_config.json and results/tables/processor_d_topology_sweep.csv")

if __name__ == "__main__": main()
