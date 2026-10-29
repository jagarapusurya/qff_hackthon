"""Transpilation and layout-routing routines."""
from qiskit import transpile


def transpile_for_processor(qc, proc):
    """Route + compile onto a processor: its coupling map, fixed layout, basis gates, opt level, seed."""
    return transpile(qc, basis_gates=proc.basis_gates, coupling_map=proc.coupling_map,
                     initial_layout=proc.layout, optimization_level=proc.optimization_level,
                     seed_transpiler=proc.seed_transpiler)


def transpile_all_to_all(qc, basis_gates, optimization_level=1, seed=42):
    """Reference compile with no connectivity constraint (no routing)."""
    return transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level,
                     seed_transpiler=seed)
