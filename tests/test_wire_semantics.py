"""Wire-semantic correctness tests.

All statevector comparisons use qml.state() on default.qubit as the oracle.
Each test group covers one gap identified in docs/review.md P1:
  - named (non-integer) wires
  - non-contiguous integer wires
  - reversed multi-qubit gate wire order
  - custom initial_state
  - randomized circuit smoke test
"""

import numpy as np
import pennylane as qml
import pytest

from pennylane_einsum import CircuitToEinsum, contract_einsum


def _compare_state(circuit_func, n_qubits, device_wires=None, params=None, atol=1e-7):
    """Compare our einsum statevector against qml.state() on default.qubit."""
    converter = CircuitToEinsum.for_qubits(n_qubits)
    einsum_data = converter.circuit_to_einsum(circuit_func, params=params)
    expr, tensors = converter.generate_full_einsum(einsum_data)
    einsum_state = contract_einsum(expr, tensors).reshape(-1)

    wires = device_wires if device_wires is not None else n_qubits
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def oracle():
        if params is None:
            circuit_func()
        else:
            circuit_func(*params)
        return qml.state()

    assert np.allclose(einsum_state, oracle(), atol=atol)


# ---------------------------------------------------------------------------
# Named wires
# ---------------------------------------------------------------------------

def test_named_wires_single_qubit():
    def circuit():
        qml.Hadamard(wires="a")
        qml.RZ(0.3, wires="a")

    _compare_state(circuit, n_qubits=1, device_wires=["a"])


def test_named_wires_entangling():
    def circuit():
        qml.Hadamard(wires="q0")
        qml.CNOT(wires=["q0", "q1"])

    _compare_state(circuit, n_qubits=2, device_wires=["q0", "q1"])


def test_named_wires_three_qubit():
    def circuit():
        qml.Hadamard(wires="q0")
        qml.CNOT(wires=["q0", "q1"])
        qml.RY(0.5, wires="q2")
        qml.CZ(wires=["q1", "q2"])

    _compare_state(circuit, n_qubits=3, device_wires=["q0", "q1", "q2"])


# ---------------------------------------------------------------------------
# Non-contiguous integer wires
# ---------------------------------------------------------------------------

def test_non_contiguous_gate_skips_middle_wire():
    """CNOT on wires [0, 2] in a 3-qubit register — wire 1 untouched."""
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 2])

    _compare_state(circuit, n_qubits=3)


def test_non_contiguous_independent_gates():
    def circuit():
        qml.Hadamard(wires=0)
        qml.RY(0.7, wires=2)
        qml.CNOT(wires=[0, 2])

    _compare_state(circuit, n_qubits=3)


def test_non_contiguous_gate_on_high_wire():
    """Gate acts on wire 2 only in a 3-qubit register."""
    def circuit():
        qml.PauliX(wires=2)
        qml.RZ(0.4, wires=2)

    _compare_state(circuit, n_qubits=3)


# ---------------------------------------------------------------------------
# Reversed multi-qubit gate wire order
# ---------------------------------------------------------------------------

def test_reversed_cnot_control_on_higher_wire():
    """CNOT(wires=[1, 0]): control=1, target=0."""
    def circuit():
        qml.PauliX(wires=1)
        qml.CNOT(wires=[1, 0])

    _compare_state(circuit, n_qubits=2)


def test_reversed_cnot_matches_oracle_entangled():
    """CNOT([1,0]) from a superposition should match oracle."""
    def circuit():
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[1, 0])

    _compare_state(circuit, n_qubits=2)


def test_swap_reversed_wires_same_as_forward():
    """SWAP is symmetric: SWAP([0,1]) and SWAP([1,0]) must produce the same state."""
    def circuit_fwd():
        qml.PauliX(wires=0)
        qml.SWAP(wires=[0, 1])

    def circuit_rev():
        qml.PauliX(wires=0)
        qml.SWAP(wires=[1, 0])

    _compare_state(circuit_fwd, n_qubits=2)
    _compare_state(circuit_rev, n_qubits=2)


def test_reversed_toffoli_wires():
    """Toffoli with controls on wires [2, 1] and target on wire 0."""
    def circuit():
        qml.PauliX(wires=1)
        qml.PauliX(wires=2)
        qml.Toffoli(wires=[2, 1, 0])

    _compare_state(circuit, n_qubits=3)


# ---------------------------------------------------------------------------
# Custom initial_state
# ---------------------------------------------------------------------------

def test_custom_initial_state_excited_single_qubit():
    """|1> initial state: H|1> = (|0> - |1>)/sqrt(2)."""
    def circuit():
        qml.Hadamard(wires=0)

    initial = np.array([0.0 + 0j, 1.0 + 0j])
    converter = CircuitToEinsum.for_qubits(1)
    data = converter.circuit_to_einsum(circuit)
    expr, tensors = converter.generate_full_einsum(data, initial_state=initial)
    result = contract_einsum(expr, tensors).reshape(-1)

    expected = np.array([1.0, -1.0]) / np.sqrt(2)
    assert np.allclose(result, expected, atol=1e-7)


def test_custom_initial_state_matches_oracle():
    """Custom initial state + circuit matches PennyLane StatePrep + same circuit."""
    initial = np.array([1.0, 1.0, 0.0, 0.0], dtype=complex) / np.sqrt(2)

    def circuit():
        qml.CNOT(wires=[0, 1])

    converter = CircuitToEinsum.for_qubits(2)
    data = converter.circuit_to_einsum(circuit)
    expr, tensors = converter.generate_full_einsum(data, initial_state=initial)
    einsum_state = contract_einsum(expr, tensors).reshape(-1)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def oracle():
        qml.StatePrep(initial, wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        return qml.state()

    assert np.allclose(einsum_state, oracle(), atol=1e-7)


def test_custom_initial_state_wrong_shape_raises():
    def circuit():
        qml.Hadamard(wires=0)

    converter = CircuitToEinsum.for_qubits(2)
    data = converter.circuit_to_einsum(circuit)
    bad_state = np.array([1.0, 0.0])  # shape (2,) but n_qubits=2 needs (4,)

    with pytest.raises(ValueError, match="initial_state"):
        converter.generate_full_einsum(data, initial_state=bad_state)


# ---------------------------------------------------------------------------
# Randomized circuit smoke test
# ---------------------------------------------------------------------------

def test_randomized_circuit_matches_oracle():
    """Random 4-qubit parameterized circuit vs qml.state() oracle."""
    rng = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, 12)

    def circuit(*p):
        for i in range(4):
            qml.RY(p[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[1, 2])
        for i in range(4):
            qml.RZ(p[4 + i], wires=i)
        qml.CNOT(wires=[0, 3])
        for i in range(4):
            qml.RY(p[8 + i], wires=i)

    _compare_state(circuit, n_qubits=4, params=list(params))
