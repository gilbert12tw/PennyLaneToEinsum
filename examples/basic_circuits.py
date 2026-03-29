import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, contract_einsum


def simple_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(0.5, wires=1)


def run_example():
    converter = CircuitToEinsum.for_qubits(2)
    einsum_data = converter.circuit_to_einsum(simple_circuit)
    expr, tensors = converter.generate_full_einsum(einsum_data)

    result = contract_einsum(expr, tensors, optimize="optimal")
    print("Einsum expression:", expr)
    print("Statevector (einsum):", result.reshape(-1))

    try:
        import opt_einsum as oe
    except Exception:
        oe = None

    if oe is not None:
        oe_result = oe.contract(expr, *tensors, optimize="optimal")
        print("Statevector (opt_einsum):", oe_result.reshape(-1))

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def pl_circuit():
        simple_circuit()
        return qml.state()

    pl_state = pl_circuit()
    print("Statevector (pennylane):", pl_state)
    print("All close:", np.allclose(result.reshape(-1), pl_state))


if __name__ == "__main__":
    run_example()

