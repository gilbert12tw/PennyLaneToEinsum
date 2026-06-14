from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, contract_einsum


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def gate_count(n_qubits: int, layers: int) -> int:
    return layers * (3 * n_qubits + n_qubits + (n_qubits - 1))


def batched_layered_circuit(x_batch: np.ndarray, layers: int, n_qubits: int) -> None:
    for layer in range(layers):
        offset = 0.01 * layer
        for q in range(n_qubits):
            xq = x_batch[:, q]
            qml.RX(xq + offset, wires=q)
            qml.RY(0.5 * xq + offset, wires=q)
            qml.RZ(0.25 * xq - offset, wires=q)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])

        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])


def unbatched_layered_circuit(x: np.ndarray, layers: int, n_qubits: int) -> None:
    for layer in range(layers):
        offset = 0.01 * layer
        for q in range(n_qubits):
            qml.RX(x[q] + offset, wires=q)
            qml.RY(0.5 * x[q] + offset, wires=q)
            qml.RZ(0.25 * x[q] - offset, wires=q)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])

        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])


def parse_batch_sizes(values: Iterable[str]) -> list[int]:
    return [int(value) for value in values]


def time_call(fn, repeats: int) -> list[float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def summarize(timings: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }


def build_batched_network(
    x_batch: np.ndarray, layers: int, n_qubits: int
) -> tuple[str, list[np.ndarray]]:
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(
        lambda: batched_layered_circuit(x_batch, layers, n_qubits)
    )
    assert data["batch_size"] == len(x_batch)
    return converter.generate_full_einsum(data)


def build_loop_networks(
    x_batch: np.ndarray, layers: int, n_qubits: int
) -> list[tuple[str, list[np.ndarray]]]:
    networks = []
    for x in x_batch:
        converter = CircuitToEinsum.for_qubits(n_qubits)
        data = converter.circuit_to_einsum(
            lambda x=x: unbatched_layered_circuit(x, layers, n_qubits)
        )
        assert data["batch_size"] is None
        networks.append(converter.generate_full_einsum(data))
    return networks


def contract_batched(
    expr: str, tensors: list[np.ndarray], batch_size: int, n_qubits: int
) -> np.ndarray:
    state = contract_einsum(expr, tensors)
    assert state.shape == (batch_size,) + (2,) * n_qubits
    return state


def contract_loop(networks: list[tuple[str, list[np.ndarray]]], n_qubits: int) -> None:
    expected_shape = (2,) * n_qubits
    for expr, tensors in networks:
        state = contract_einsum(expr, tensors)
        assert state.shape == expected_shape


def build_pennylane_qnodes(layers: int, n_qubits: int):
    batched_dev = qml.device("default.qubit", wires=n_qubits)
    loop_dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(batched_dev)
    def batched_state(x_batch):
        batched_layered_circuit(x_batch, layers, n_qubits)
        return qml.state()

    @qml.qnode(loop_dev)
    def loop_state(x):
        unbatched_layered_circuit(x, layers, n_qubits)
        return qml.state()

    return batched_state, loop_state


def run_pennylane_batched_state(qnode, x_batch: np.ndarray, n_qubits: int) -> None:
    state = qnode(x_batch)
    assert np.asarray(state).shape == (len(x_batch), 2**n_qubits)


def run_pennylane_loop_state(qnode, x_batch: np.ndarray, n_qubits: int) -> None:
    for x in x_batch:
        state = qnode(x)
        assert np.asarray(state).shape == (2**n_qubits,)


def validate_batched_state(x_batch: np.ndarray, layers: int, n_qubits: int) -> None:
    expr, tensors = build_batched_network(x_batch, layers, n_qubits)
    einsum_states = contract_batched(expr, tensors, len(x_batch), n_qubits).reshape(
        len(x_batch), -1
    )

    batched_state, _ = build_pennylane_qnodes(layers, n_qubits)
    pl_states = np.asarray(batched_state(x_batch))
    np.testing.assert_allclose(einsum_states, pl_states, atol=1e-10)


def benchmark_case(
    batch_size: int, n_qubits: int, layers: int, repeats: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(2026 + 1000 * n_qubits + 10 * layers + batch_size)
    x_batch = rng.uniform(0.0, 2.0 * np.pi, (batch_size, n_qubits))

    # Conversion happens before timing. The benchmark below times contraction only.
    batched_expr, batched_tensors = build_batched_network(x_batch, layers, n_qubits)
    loop_networks = build_loop_networks(x_batch, layers, n_qubits)
    pl_batched_state, pl_loop_state = build_pennylane_qnodes(layers, n_qubits)

    # Warm up each timed path.
    contract_batched(batched_expr, batched_tensors, batch_size, n_qubits)
    contract_loop(loop_networks, n_qubits)
    run_pennylane_batched_state(pl_batched_state, x_batch, n_qubits)
    run_pennylane_loop_state(pl_loop_state, x_batch, n_qubits)

    batched_contract = summarize(
        time_call(
            lambda: contract_batched(
                batched_expr, batched_tensors, batch_size, n_qubits
            ),
            repeats,
        )
    )
    loop_contract = summarize(
        time_call(lambda: contract_loop(loop_networks, n_qubits), repeats)
    )
    pl_batched = summarize(
        time_call(
            lambda: run_pennylane_batched_state(pl_batched_state, x_batch, n_qubits),
            repeats,
        )
    )
    pl_loop = summarize(
        time_call(
            lambda: run_pennylane_loop_state(pl_loop_state, x_batch, n_qubits),
            repeats,
        )
    )

    batched_contract_median = batched_contract["median_seconds"]
    loop_contract_median = loop_contract["median_seconds"]
    pl_batched_median = pl_batched["median_seconds"]
    pl_loop_median = pl_loop["median_seconds"]

    return {
        "n_qubits": n_qubits,
        "layers": layers,
        "gates_per_sample": gate_count(n_qubits, layers),
        "batch_size": batch_size,
        "repeats": repeats,
        "einsum_batched_contract_median_seconds": batched_contract_median,
        "einsum_loop_contract_median_seconds": loop_contract_median,
        "einsum_contract_speedup": loop_contract_median / batched_contract_median,
        "pennylane_batched_state_median_seconds": pl_batched_median,
        "pennylane_loop_state_median_seconds": pl_loop_median,
        "pennylane_state_speedup": pl_loop_median / pl_batched_median,
        "einsum_batched_contract_per_sample_ms": 1000.0
        * batched_contract_median
        / batch_size,
        "pennylane_batched_state_per_sample_ms": 1000.0
        * pl_batched_median
        / batch_size,
    }


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "n_qubits",
        "layers",
        "gates_per_sample",
        "batch_size",
        "repeats",
        "einsum_batched_contract_median_seconds",
        "einsum_loop_contract_median_seconds",
        "einsum_contract_speedup",
        "pennylane_batched_state_median_seconds",
        "pennylane_loop_state_median_seconds",
        "pennylane_state_speedup",
        "einsum_batched_contract_per_sample_ms",
        "pennylane_batched_state_per_sample_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["batch_size"]))
    batch_sizes = [int(row["batch_size"]) for row in rows]
    einsum_speedups = [float(row["einsum_contract_speedup"]) for row in rows]
    pl_speedups = [float(row["pennylane_state_speedup"]) for row in rows]
    einsum_per_sample = [
        float(row["einsum_batched_contract_per_sample_ms"]) for row in rows
    ]
    pl_per_sample = [float(row["pennylane_batched_state_per_sample_ms"]) for row in rows]

    fig, (ax_speedup, ax_sample) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=160)

    ax_speedup.plot(batch_sizes, einsum_speedups, marker="o", label="einsum contract")
    ax_speedup.plot(batch_sizes, pl_speedups, marker="s", label="PennyLane state")
    ax_speedup.axhline(1.0, color="0.5", linewidth=1, linestyle="--")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xticks(batch_sizes)
    ax_speedup.set_xticklabels([str(size) for size in batch_sizes])
    ax_speedup.set_xlabel("Batch size")
    ax_speedup.set_ylabel("Speedup vs loop")
    ax_speedup.set_title("Batch speedup")
    ax_speedup.grid(True, alpha=0.3)
    ax_speedup.legend()

    ax_sample.plot(
        batch_sizes, einsum_per_sample, marker="o", label="einsum contract"
    )
    ax_sample.plot(batch_sizes, pl_per_sample, marker="s", label="PennyLane state")
    ax_sample.set_xscale("log", base=2)
    ax_sample.set_xticks(batch_sizes)
    ax_sample.set_xticklabels([str(size) for size in batch_sizes])
    ax_sample.set_xlabel("Batch size")
    ax_sample.set_ylabel("Median time per sample (ms)")
    ax_sample.set_title("Batched per-sample cost")
    ax_sample.grid(True, alpha=0.3)
    ax_sample.legend()

    fig.suptitle(
        f'Batch contraction benchmark: {int(rows[0]["n_qubits"])} qubits, '
        f'{int(rows[0]["layers"])} layers, '
        f'{int(rows[0]["gates_per_sample"])} gates/sample'
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark batched einsum contraction against looped contraction "
            "and PennyLane state execution."
        )
    )
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        action="append",
        default=[],
        help="Batch size to test. May be passed multiple times.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/benchmarks/batch_contraction.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("docs/assets/batch_contraction_benchmark.png"),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip one correctness check against PennyLane batched state.",
    )
    args = parser.parse_args()

    batch_sizes = (
        parse_batch_sizes(args.batch_size)
        if args.batch_size
        else DEFAULT_BATCH_SIZES
    )

    if not args.skip_validation:
        rng = np.random.default_rng(42)
        validation_batch = rng.uniform(
            0.0, 2.0 * np.pi, (min(3, max(batch_sizes)), args.n_qubits)
        )
        validate_batched_state(validation_batch, args.layers, args.n_qubits)

    rows = [
        benchmark_case(batch_size, args.n_qubits, args.layers, args.repeats)
        for batch_size in batch_sizes
    ]
    write_csv(rows, args.csv)
    write_plot(rows, args.plot)

    print(
        f"Benchmark circuit: {args.n_qubits} qubits, {args.layers} layers, "
        f"{gate_count(args.n_qubits, args.layers)} gates/sample"
    )
    for row in rows:
        print(
            f'B={int(row["batch_size"]):2d}: '
            f'einsum contract {1000.0 * float(row["einsum_batched_contract_median_seconds"]):8.3f} ms '
            f'vs loop {1000.0 * float(row["einsum_loop_contract_median_seconds"]):8.3f} ms '
            f'({float(row["einsum_contract_speedup"]):5.2f}x); '
            f'PennyLane state {1000.0 * float(row["pennylane_batched_state_median_seconds"]):8.3f} ms '
            f'vs loop {1000.0 * float(row["pennylane_loop_state_median_seconds"]):8.3f} ms '
            f'({float(row["pennylane_state_speedup"]):5.2f}x)'
        )
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
