"""Quantum feature maps and fidelity kernels for PS-Q2.

Each feature map is ONE op-list specification, used two ways:
  * build_circuit(): a Qiskit circuit for a single sample (resource counts, noisy runs, verification)
  * batched_states(): a fast batched numpy statevector simulator (ideal kernel for CV and final fit)
Both come from the same spec, and tests/verification compare them against qiskit.quantum_info.Statevector.

Inputs `z` are PCA components of standardised data, rescaled to unit variance on TRAIN only.
Rotation angles are  s * z_i  (single-qubit)  and  s^2 * z_i * z_j  (two-qubit RZZ),  s = bandwidth.
"""
import numpy as np
from qiskit import QuantumCircuit

# ------------------------------------------------------------------ specs
def ring_edges(k):
    return [(i, (i + 1) % k) for i in range(k)] if k > 2 else [(0, 1)]

def linear_edges(k):
    return [(i, i + 1) for i in range(k - 1)]

def all_edges(k):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]

EDGES = {"ring": ring_edges, "linear": linear_edges, "all": all_edges}


def feature_map_ops(name, k, reps, s, topology="ring"):
    """Op list. Angle functions take z of shape (n, k) and return (n,)."""
    ops = []
    ang1 = lambda i, sc=s: (lambda z: sc * z[:, i])
    for _ in range(reps):
        if name == "product":                      # control: no entanglement
            for q in range(k): ops.append(("h", q))
            for q in range(k): ops.append(("rz", q, ang1(q)))
        elif name == "zz":                         # IQP / ZZ-style, diagonal entanglers
            for q in range(k): ops.append(("h", q))
            for q in range(k): ops.append(("rz", q, ang1(q)))
            for a, b in EDGES[topology](k):
                ops.append(("rzz", a, b, (lambda a=a, b=b: (lambda z: s * s * z[:, a] * z[:, b]))()))
        elif name == "reup":                       # custom: RY + neighbour-RZ re-uploading, CX ring (non-diagonal)
            for q in range(k): ops.append(("ry", q, ang1(q)))
            for q in range(k): ops.append(("rz", q, ang1((q + 1) % k)))
            for a, b in EDGES[topology](k): ops.append(("cx", a, b))
        else:
            raise ValueError(name)
    return ops


# ------------------------------------------------------------------ qiskit side
def build_circuit(ops, k, z_single):
    """Qiskit circuit for one sample (z_single shape (k,)). rzz is expanded to CX-RZ-CX."""
    z = np.asarray(z_single, float)[None, :]
    qc = QuantumCircuit(k)
    for op in ops:
        g = op[0]
        if g == "h": qc.h(op[1])
        elif g in ("rx", "ry", "rz"): getattr(qc, g)(float(op[2](z)[0]), op[1])
        elif g == "cx": qc.cx(op[1], op[2])
        elif g == "rzz":
            qc.cx(op[1], op[2]); qc.rz(float(op[3](z)[0]), op[2]); qc.cx(op[1], op[2])
    return qc


def fidelity_circuit(ops, k, z1, z2, barrier=True):
    """Compute-uncompute: U(z2)^dagger U(z1), measure all. K = P(0...0).
    A barrier separates U and U^dagger so the transpiler cannot cancel them (fixed, honest circuit cost)."""
    qc = QuantumCircuit(k, k)
    qc.compose(build_circuit(ops, k, z1), inplace=True)
    if barrier:
        qc.barrier()
    qc.compose(build_circuit(ops, k, z2).inverse(), inplace=True)
    qc.measure(range(k), range(k))
    return qc


# ------------------------------------------------------------------ batched numpy simulator
def _apply_1q(state, mats, q, k):
    n = state.shape[0]
    st = state.reshape(n, 2 ** (k - 1 - q), 2, 2 ** q)
    if mats.ndim == 2:
        out = np.einsum("ab,nxby->nxay", mats, st)
    else:
        out = np.einsum("nab,nxby->nxay", mats, st)
    return out.reshape(n, -1)

def _cx_perm(c, t, k):
    idx = np.arange(2 ** k)
    return np.where((idx >> c) & 1, idx ^ (1 << t), idx)

def batched_states(ops, z, k):
    z = np.asarray(z, float)
    n = z.shape[0]
    st = np.zeros((n, 2 ** k), complex); st[:, 0] = 1.0
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    def rot(g, th):
        c, s_ = np.cos(th / 2), np.sin(th / 2)
        m = np.zeros((len(th), 2, 2), complex)
        if g == "rz": m[:, 0, 0] = np.exp(-0.5j * th); m[:, 1, 1] = np.exp(0.5j * th)
        elif g == "ry": m[:, 0, 0] = c; m[:, 0, 1] = -s_; m[:, 1, 0] = s_; m[:, 1, 1] = c
        elif g == "rx": m[:, 0, 0] = c; m[:, 0, 1] = -1j * s_; m[:, 1, 0] = -1j * s_; m[:, 1, 1] = c
        return m
    for op in ops:
        g = op[0]
        if g == "h": st = _apply_1q(st, H, op[1], k)
        elif g in ("rx", "ry", "rz"): st = _apply_1q(st, rot(g, op[2](z)), op[1], k)
        elif g == "cx": st = st[:, _cx_perm(op[1], op[2], k)]
        elif g == "rzz":
            th = op[3](z); a, b = op[1], op[2]
            idx = np.arange(2 ** k)
            parity = (((idx >> a) & 1) ^ ((idx >> b) & 1))            # 0 -> z_a z_b = +1
            sign = 1 - 2 * parity                                      # exp(-i th/2 * Z_a Z_b)
            st = st * np.exp(-0.5j * th[:, None] * sign[None, :])
    return st


def kernel_from_states(SA, SB):
    return np.abs(SA.conj() @ SB.T) ** 2


# ------------------------------------------------------------------ diagnostics
def offdiag_stats(K):
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    return float(off.mean()), float(off.std())

def target_alignment(K, y01):
    y = 2 * np.asarray(y01) - 1
    return float((y @ K @ y) / (len(y) * np.linalg.norm(K)))
