from __future__ import annotations

import numpy as np
import pennylane as qml


def parameter_count(n_qubits: int, layers: int) -> int:
    return 3 * n_qubits * layers


def make_parameters(n_qubits: int, layers: int, seed: int = 2026):
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-np.pi, np.pi, size=(n_qubits, 3))
    weights = rng.uniform(-np.pi, np.pi, size=(layers, n_qubits, 3))
    return inputs, weights


def qae_circuit(inputs: np.ndarray, weights: np.ndarray) -> None:
    """QAE-Net Fig. 1 circuit with a scalable linear CNOT chain."""
    n_qubits = inputs.shape[0]
    if inputs.shape != (n_qubits, 3):
        raise ValueError("inputs must have shape (n_qubits, 3)")
    if weights.ndim != 3 or weights.shape[1:] != (n_qubits, 3):
        raise ValueError("weights must have shape (layers, n_qubits, 3)")

    for q in range(n_qubits):
        qml.Hadamard(wires=q)
        qml.RZ(inputs[q, 0], wires=q)
        qml.RY(inputs[q, 1], wires=q)
        qml.RZ(inputs[q, 2], wires=q)

    for layer in range(weights.shape[0]):
        for q in range(n_qubits):
            qml.RZ(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
