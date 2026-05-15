import pytest
import pennylane as qml

from pennylane_einsum import CircuitToEinsum
from pennylane_einsum.exceptions import UnsupportedOperationError


def test_three_qubit_gate_supported():
    def circuit():
        qml.Toffoli(wires=[0, 1, 2])

    converter = CircuitToEinsum.for_qubits(3)
    einsum_data = converter.circuit_to_einsum(circuit)
    assert einsum_data["operations"]


def test_state_prep_raises_unsupported_operation_error():
    def circuit():
        qml.StatePrep([1.0, 0.0], wires=0)

    converter = CircuitToEinsum.for_qubits(1)
    with pytest.raises(UnsupportedOperationError) as exc_info:
        converter.circuit_to_einsum(circuit)

    assert "StatePrep" in str(exc_info.value)


def test_unsupported_error_includes_wires():
    def circuit():
        qml.StatePrep([1.0, 0.0], wires=0)

    converter = CircuitToEinsum.for_qubits(1)
    with pytest.raises(UnsupportedOperationError) as exc_info:
        converter.circuit_to_einsum(circuit)

    assert "0" in str(exc_info.value)
