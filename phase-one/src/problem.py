"""Problem formulation for PS-Q2 (quantum kernel estimator + SVM).

This problem is kernel-based rather than variational, so there is no cost
Hamiltonian to minimise. The analogue is the data-dependent ZZ interaction
used in the feature map; the quantity to estimate is the fidelity kernel

    K(x, x') = |<phi(x)|phi(x')>|^2,   |phi(x)> = U(x)|0000>.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SEED = 42
N_QUBITS = 4


def generate_dataset(n_samples=100, noise=0.1, seed=SEED, test_size=0.2):
    """make_moons(n_samples=100, noise=0.1, random_state=42), split then scaled to [0, pi]."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = MinMaxScaler(feature_range=(0, np.pi))   # fit on train only
    return {"X": X, "y": y,
            "X_train": scaler.fit_transform(Xtr_raw), "X_test": scaler.transform(Xte_raw),
            "y_train": ytr, "y_test": yte}


def feature_map(x, n_qubits=N_QUBITS):
    """Data-dependent ZZ-style feature map U(x): H + Rz(2x) encoding, then ring ZZ entanglers."""
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(2 * float(x[q % 2]), q)
    phi = 2 * (np.pi - float(x[0])) * (np.pi - float(x[1]))
    for q in range(n_qubits):
        a, b = q, (q + 1) % n_qubits          # ring: (0,1),(1,2),(2,3),(3,0)
        qc.cx(a, b)
        qc.rz(phi, b)
        qc.cx(a, b)
    return qc


def make_fidelity_circuit(x1, x2, n_qubits=N_QUBITS):
    """Compute-uncompute circuit U(x2)^dagger U(x1); P(0...0) = K(x1, x2)."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.compose(feature_map(x1, n_qubits), inplace=True)
    qc.compose(feature_map(x2, n_qubits).inverse(), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def statevector(x, n_qubits=N_QUBITS):
    return Statevector.from_instruction(feature_map(x, n_qubits)).data


def ideal_kernel(x1, x2):
    return float(np.abs(np.vdot(statevector(x1), statevector(x2))) ** 2)


def ideal_kernel_matrix(XA, XB):
    SA = [statevector(x) for x in XA]
    SB = [statevector(x) for x in XB]
    return np.array([[np.abs(np.vdot(a, b)) ** 2 for b in SB] for a in SA])


def classification_metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)}


def train_precomputed_svm(K_train, K_test, y_train, y_test, C=1.0):
    svm = SVC(kernel="precomputed", C=C).fit(K_train, y_train)
    pred = svm.predict(K_test)
    return svm, pred, classification_metrics(y_test, pred)


def rbf_baseline(X_train, X_test, y_train, y_test, C=1.0):
    pred = SVC(kernel="rbf", C=C).fit(X_train, y_train).predict(X_test)
    return classification_metrics(y_test, pred)
