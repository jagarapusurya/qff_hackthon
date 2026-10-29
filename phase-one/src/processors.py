"""Processor topology definitions and noise-model loaders (reads processors/*.json)."""
import json
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


@dataclass
class Processor:
    name: str
    n_qubits: int
    edges: list
    layout: list
    basis_gates: list
    p1: float
    p2: float
    p_readout: float
    optimization_level: int
    seed_transpiler: int
    shots: int
    topology: str = ""
    role: str = ""

    @property
    def coupling_map(self):
        e = [list(x) for x in self.edges] + [[b, a] for a, b in self.edges]
        return CouplingMap(e)

    @property
    def graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(self.n_qubits))
        g.add_edges_from(self.edges)
        return g

    @property
    def noise_model(self):
        return build_noise_model(self.basis_gates, self.p1, self.p2, self.p_readout)

    @property
    def backend(self):
        return AerSimulator(noise_model=self.noise_model)


def build_noise_model(basis_gates, p1, p2, p_ro):
    """1q depolarizing on rz/sx/x, 2q depolarizing on cx, symmetric readout error."""
    nm = NoiseModel(basis_gates=basis_gates)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def load_processor(path):
    cfg = json.loads(Path(path).read_text())
    return Processor(
        name=cfg["name"], n_qubits=cfg["n_qubits"],
        edges=[tuple(e) for e in cfg["coupling_edges"]],
        layout=cfg["initial_layout"], basis_gates=cfg["basis_gates"],
        p1=cfg["noise"]["p1"], p2=cfg["noise"]["p2"], p_readout=cfg["noise"]["p_readout"],
        optimization_level=cfg["transpile"]["optimization_level"],
        seed_transpiler=cfg["transpile"]["seed_transpiler"], shots=cfg["shots"],
        topology=cfg.get("topology", ""), role=cfg.get("role", ""))


def native_ring_edges(proc, n_logical=4):
    """How many of the logical ring edges are native edges under the processor's layout."""
    es = {frozenset(e) for e in proc.edges}
    ring = [(q, (q + 1) % n_logical) for q in range(n_logical)]
    return sum(frozenset((proc.layout[a], proc.layout[b])) in es for a, b in ring), len(ring)
