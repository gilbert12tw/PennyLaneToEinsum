import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum


def _compare_state(circuit_func, n_qubits, params=None, atol=1e-7):
    converter = CircuitToEinsum.for_qubits(n_qubits)
    einsum_data = converter.circuit_to_einsum(circuit_func, params=params)
    einsum_expr, tensors = converter.generate_full_einsum(einsum_data)
    einsum_state = np.einsum(einsum_expr, *tensors).reshape(-1)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def pennylane_circuit():
        if params is None:
            circuit_func()
        else:
            circuit_func(*params)
        return qml.state()

    pl_state = pennylane_circuit()
    assert np.allclose(einsum_state, pl_state, atol=atol)


def test_single_qubit_gates():
    def circuit():
        qml.Hadamard(wires=0)
        qml.RZ(0.25, wires=0)

    _compare_state(circuit, n_qubits=1)


def test_two_qubit_entangling():
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])

    _compare_state(circuit, n_qubits=2)


def test_parameterized_circuit():
    def circuit(theta):
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RX(theta / 2.0, wires=1)

    _compare_state(circuit, n_qubits=2, params=[0.43])


def test_three_qubit_chain():
    def circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RZ(0.3, wires=2)

    _compare_state(circuit, n_qubits=3)

