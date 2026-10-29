"""Processor D trade-off study: two-way vs one-way links on the selected 2x3 grid, measured on the official
QFT-3 benchmark AND on this project's Q2 kernel circuit (6-qubit ZZ-ring fidelity circuit). Run from project root:
    python run_processor_d_direction.py
Writes results/tables/processor_d_direction.json. Does not modify processor_d_config.json."""
import copy, json, sys
from pathlib import Path
from qiskit import transpile
from qiskit.transpiler import CouplingMap
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark"))
from benchmark.qff_qml_benchmark import task_b_qft3, validate_processor_d
from src.qkernel import feature_map_ops, fidelity_circuit
from src.qcv import encode
from src.q2_data import load_q2
from noise_model.benchmark_noise_model import SEED, assert_all_gates_noisy

cfg2 = json.loads((ROOT / "processor_d_config.json").read_text())
cfg1 = copy.deepcopy(cfg2)
cfg1["coupling_map"] = [[a, b] for a, b in cfg2["coupling_map"] if a < b]      # keep one direction per link
cfg1["benchmark_results"] = {}

X, y, _, _ = load_q2()
Z = encode(X, [X], 6)[0]
circ = fidelity_circuit(feature_map_ops("zz", 6, 1, 0.35, "ring"), 6, Z[0], Z[1])
logical = transpile(circ, basis_gates=cfg2["native_gate_set"], optimization_level=1, seed_transpiler=SEED)

out = {"q2_circuit_all_to_all": dict(depth=logical.depth(), cx=logical.count_ops().get("cx", 0), ops=logical.size())}
for name, cfg in [("two_way", cfg2), ("one_way", cfg1)]:
    assert not validate_processor_d(cfg, require_results=False), name
    t = transpile(circ, basis_gates=cfg["native_gate_set"], coupling_map=CouplingMap(cfg["coupling_map"]),
                  optimization_level=1, seed_transpiler=SEED)
    assert_all_gates_noisy(t)
    layout = t.layout.initial_index_layout(filter_ancillas=True)
    out[name] = dict(directed_edges=len(cfg["coupling_map"]), qft3=task_b_qft3(cfg),
                     q2_depth=t.depth(), q2_cx=t.count_ops().get("cx", 0), q2_ops=t.size(),
                     q2_swap_overhead_cx=t.count_ops().get("cx", 0) - out["q2_circuit_all_to_all"]["cx"],
                     q2_logical_to_physical=layout)
(ROOT / "results" / "tables" / "processor_d_direction.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
