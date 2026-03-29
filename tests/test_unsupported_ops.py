import pytest
import pennylane as qml

from pennylane_einsum import CircuitToEinsum


def test_three_qubit_gate_supported():
    def circuit():
        qml.Toffoli(wires=[0, 1, 2])

    converter = CircuitToEinsum.for_qubits(3)
    einsum_data = converter.circuit_to_einsum(circuit)
    assert einsum_data["operations"]


def test_state_prep_not_supported():
    def circuit():
        qml.StatePrep([1.0, 0.0], wires=0)

    converter = CircuitToEinsum.for_qubits(1)
    with pytest.raises(Exception):
        converter.circuit_to_einsum(circuit)

