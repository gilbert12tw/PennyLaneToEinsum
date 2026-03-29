import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum


def test_toffoli_state_matches():
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Toffoli(wires=[0, 1, 2])

    converter = CircuitToEinsum.for_qubits(3)
    einsum_data = converter.circuit_to_einsum(circuit)
    expr, tensors = converter.generate_full_einsum(einsum_data)
    einsum_state = np.einsum(expr, *tensors).reshape(-1)

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def pl_circuit():
        circuit()
        return qml.state()

    pl_state = pl_circuit()
    assert np.allclose(einsum_state, pl_state)

