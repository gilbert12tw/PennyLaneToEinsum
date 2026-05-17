"""Runnable examples for explaining PennyLane-to-einsum conversion.

This script is intentionally verbose. It is meant to be used as a presentation
companion: run it section by section, then point at the printed QuantumTape,
index flow, tensor shapes, and final einsum expression.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, contract_einsum


def simple_circuit() -> None:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(0.5, wires=1)


def named_wire_circuit() -> None:
    qml.Hadamard(wires="q0")
    qml.CNOT(wires=["q0", "q1"])
    qml.RY(0.25, wires="q1")


def qk_learnob_style_ansatz(x: np.ndarray, layers: int, wires: Iterable[int]) -> None:
    """Small version of the circuit pattern covered by the integration tests."""
    wires = list(wires)
    for _ in range(layers):
        for i, wire in enumerate(wires):
            qml.RX(x[i], wires=wire)
        for i, wire in enumerate(wires):
            qml.CNOT(wires=[wire, wires[(i + 1) % len(wires)]])


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def build_tape(circuit: Callable[..., Any], params: Iterable[Any] | None = None):
    with qml.tape.QuantumTape() as tape:
        if params is None:
            circuit()
        else:
            circuit(*params)
    return tape


def inspect_quantum_tape(
    circuit: Callable[..., Any],
    params: Iterable[Any] | None = None,
) -> None:
    """Show what PennyLane records before this project converts anything."""
    tape = build_tape(circuit, params=params)
    print("Tape wires:", list(tape.wires))
    print("Number of operations:", len(tape.operations))

    for i, op in enumerate(tape.operations):
        try:
            matrix_shape = op.matrix().shape
        except Exception as exc:
            matrix_shape = f"unavailable ({type(exc).__name__}: {exc})"

        print(
            f"  op[{i}] name={op.name:<9} wires={list(op.wires)!r:<10} "
            f"num_params={op.num_params} matrix_shape={matrix_shape}"
        )
        if op.parameters:
            print(f"        params={op.parameters}")


def print_converter_data(einsum_data: dict[str, Any]) -> None:
    """Show the intermediate data returned by CircuitToEinsum.circuit_to_einsum."""
    print("Initial indices:", einsum_data["initial_indices"])

    current_indices = dict(einsum_data["initial_indices"])
    for i, op in enumerate(einsum_data["operations"]):
        in_part, tensor_indices = op["einsum"].split(",")
        n_wires = len(op["wires"])
        out_part = tensor_indices[n_wires:]

        print(f"  converted_op[{i}] {op['gate_name']} on wires={op['wires']}")
        print(f"      input indices:  {list(in_part)}")
        print(f"      output indices: {list(out_part)}")
        print(f"      gate term:      {tensor_indices}")
        print(f"      tensor shape:   {op['tensor'].shape}")
        print(f"      params:         {op['op_params']}")

        for wire, out_idx in zip(op["wires"], out_part):
            current_indices[wire] = out_idx
        print(f"      index snapshot: {current_indices}")

    print("Final indices:", einsum_data["final_indices"])


def compare_with_pennylane(
    circuit: Callable[..., Any],
    n_qubits: int,
    state: np.ndarray,
    device_wires: int | list[Any] | None = None,
    params: Iterable[Any] | None = None,
) -> None:
    wires = device_wires if device_wires is not None else n_qubits
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def qnode():
        if params is None:
            circuit()
        else:
            circuit(*params)
        return qml.state()

    pennylane_state = np.asarray(qnode())
    print("Statevector (einsum):   ", state.reshape(-1))
    print("Statevector (PennyLane):", pennylane_state)
    print("All close:", np.allclose(state.reshape(-1), pennylane_state))


def convert_and_print(
    circuit: Callable[..., Any],
    n_qubits: int,
    params: Iterable[Any] | None = None,
    initial_state: np.ndarray | None = None,
) -> tuple[str, list[np.ndarray], np.ndarray, dict[str, Any]]:
    converter = CircuitToEinsum.for_qubits(n_qubits)
    einsum_data = converter.circuit_to_einsum(circuit, params=params)
    expr, tensors = converter.generate_full_einsum(
        einsum_data,
        initial_state=initial_state,
    )
    state = contract_einsum(expr, tensors, optimize="optimal")

    print_converter_data(einsum_data)
    print("Full einsum expression:", expr)
    print("Tensor count:", len(tensors))
    print("Tensor shapes:", [tensor.shape for tensor in tensors])

    return expr, tensors, state, einsum_data


def demo_basic_conversion() -> None:
    section("1. Basic circuit: QuantumTape -> gate tensors -> full einsum")
    inspect_quantum_tape(simple_circuit)
    _, _, state, _ = convert_and_print(simple_circuit, n_qubits=2)
    compare_with_pennylane(simple_circuit, n_qubits=2, state=state)


def demo_named_wires() -> None:
    section("2. Named wires: PennyLane labels are mapped to contiguous positions")
    inspect_quantum_tape(named_wire_circuit)
    _, _, state, _ = convert_and_print(named_wire_circuit, n_qubits=2)
    compare_with_pennylane(
        named_wire_circuit,
        n_qubits=2,
        state=state,
        device_wires=["q0", "q1"],
    )


def demo_custom_initial_state() -> None:
    section("3. Custom initial state: generate_full_einsum(initial_state=...)")

    def circuit() -> None:
        qml.Hadamard(wires=0)

    initial = np.array([0.0 + 0.0j, 1.0 + 0.0j])
    _, _, state, _ = convert_and_print(circuit, n_qubits=1, initial_state=initial)

    expected = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
    print("Initial state:", initial)
    print("Statevector (einsum):", state.reshape(-1))
    print("Expected H|1>:", expected)
    print("All close:", np.allclose(state.reshape(-1), expected))


def demo_qk_learnob_pattern() -> None:
    section("4. QK-Learnob-style ansatz: RX layers + ring CNOTs")

    x = np.array([0.2, 0.7])
    layers = 2

    def circuit() -> None:
        qk_learnob_style_ansatz(x=x, layers=layers, wires=range(2))

    inspect_quantum_tape(circuit)
    _, _, state, _ = convert_and_print(circuit, n_qubits=2)
    compare_with_pennylane(circuit, n_qubits=2, state=state)


def run_example() -> None:
    demo_basic_conversion()
    demo_named_wires()
    demo_custom_initial_state()
    demo_qk_learnob_pattern()


if __name__ == "__main__":
    run_example()
