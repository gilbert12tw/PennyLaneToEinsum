import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, contract_einsum
from pennylane_einsum.qae_circuit import make_parameters, qae_circuit


def test_qae_circuit_uses_linear_chain_without_closing_cnot():
    inputs, weights = make_parameters(5, 2)
    converter = CircuitToEinsum.for_qubits(5)
    data = converter.circuit_to_einsum(lambda: qae_circuit(inputs, weights))
    cnots = [op["wires"] for op in data["operations"] if op["gate_name"] == "CNOT"]

    assert cnots == [[0, 1], [1, 2], [2, 3], [3, 4]] * 2


def test_direct_z_expectations_match_pennylane():
    n_qubits = 4
    inputs, weights = make_parameters(n_qubits, 2)
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(lambda: qae_circuit(inputs, weights))
    actual = []
    for wire in range(n_qubits):
        expr, tensors = converter.generate_expectation_einsum(data, {wire: "Z"})
        actual.append(float(np.real(contract_einsum(expr, tensors))))

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def reference():
        qae_circuit(inputs, weights)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    np.testing.assert_allclose(actual, reference(), atol=1e-10)
