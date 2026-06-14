import numpy as np
import pennylane as qml
import pytest

from pennylane_einsum import CircuitToEinsum, contract_einsum


def _batched_state(circuit_func, n_qubits):
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(circuit_func)
    expr, tensors = converter.generate_full_einsum(data)
    state = contract_einsum(expr, tensors)
    return data, state, state.reshape(data["batch_size"], -1)


def _pennylane_state(circuit_func, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_func()
        return qml.state()

    return np.array(qnode())


def _assert_batched_matches_pennylane(batch_states, sample_circuit, n_qubits):
    for i, state in enumerate(batch_states):
        expected = _pennylane_state(lambda i=i: sample_circuit(i), n_qubits)
        np.testing.assert_allclose(state, expected, atol=1e-10)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_batched_rx_ry_rz_matches_pennylane(batch_size):
    theta = np.linspace(0.1, 0.7, batch_size)

    def batched_circuit():
        qml.Hadamard(wires=0)
        qml.RX(theta, wires=0)
        qml.RY(theta + 0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(theta * 0.5, wires=1)

    def sample_circuit(i):
        qml.Hadamard(wires=0)
        qml.RX(theta[i], wires=0)
        qml.RY(theta[i] + 0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(theta[i] * 0.5, wires=1)

    data, state_tensor, batch_states = _batched_state(batched_circuit, n_qubits=2)

    assert data["batch_size"] == batch_size
    assert data["batch_index"] is not None
    assert state_tensor.shape == (batch_size, 2, 2)
    assert batch_states.shape == (batch_size, 4)
    _assert_batched_matches_pennylane(batch_states, sample_circuit, n_qubits=2)


def test_batched_rot_matches_pennylane():
    phi = np.array([0.1, 0.2, 0.3])
    theta = np.array([0.4, 0.5, 0.6])
    omega = np.array([0.7, 0.8, 0.9])

    def batched_circuit():
        qml.Rot(phi, theta, omega, wires=0)
        qml.CNOT(wires=[0, 1])

    def sample_circuit(i):
        qml.Rot(phi[i], theta[i], omega[i], wires=0)
        qml.CNOT(wires=[0, 1])

    _, _, batch_states = _batched_state(batched_circuit, n_qubits=2)
    _assert_batched_matches_pennylane(batch_states, sample_circuit, n_qubits=2)


def test_batched_two_qubit_parameterized_gate_matches_pennylane():
    phi = np.array([0.1, 0.2, 0.3])
    theta = np.array([0.4, 0.5, 0.6])
    omega = np.array([0.7, 0.8, 0.9])

    def batched_circuit():
        qml.Hadamard(wires=0)
        qml.CRot(phi, theta, omega, wires=[0, 1])

    def sample_circuit(i):
        qml.Hadamard(wires=0)
        qml.CRot(phi[i], theta[i], omega[i], wires=[0, 1])

    _, state_tensor, batch_states = _batched_state(batched_circuit, n_qubits=2)

    assert state_tensor.shape == (3, 2, 2)
    _assert_batched_matches_pennylane(batch_states, sample_circuit, n_qubits=2)


def test_mismatched_batched_gate_sizes_raise_value_error():
    theta = np.array([0.1, 0.2])
    phi = np.array([0.3, 0.4, 0.5])

    def circuit():
        qml.RX(theta, wires=0)
        qml.RY(phi, wires=0)

    converter = CircuitToEinsum.for_qubits(1)

    with pytest.raises(ValueError, match="same leading batch size"):
        converter.circuit_to_einsum(circuit)
