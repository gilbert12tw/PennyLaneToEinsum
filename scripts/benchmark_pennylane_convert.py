from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum


DEFAULT_CASES = [
    (4, 5),
    (6, 8),
    (8, 12),
    (10, 16),
    (12, 20),
    (14, 24),
]


def layered_large_circuit(x: np.ndarray, layers: int, n_qubits: int) -> None:
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


def gate_count(n_qubits: int, layers: int) -> int:
    return layers * (3 * n_qubits + n_qubits + (n_qubits - 1))


def parse_cases(values: Iterable[str]) -> list[tuple[int, int]]:
    cases = []
    for value in values:
        n_qubits, layers = value.split("x", maxsplit=1)
        cases.append((int(n_qubits), int(layers)))
    return cases


def benchmark_case(n_qubits: int, layers: int, repeats: int) -> dict[str, float | int]:
    rng = np.random.default_rng(1234 + n_qubits * 100 + layers)
    x = rng.uniform(0.0, 2.0 * np.pi, n_qubits)
    converter = CircuitToEinsum.for_qubits(n_qubits)

    # Warm up PennyLane object construction before timing.
    converter.circuit_to_einsum(lambda: layered_large_circuit(x, layers, n_qubits))

    timings = []
    n_operations = gate_count(n_qubits, layers)
    for _ in range(repeats):
        converter = CircuitToEinsum.for_qubits(n_qubits)
        start = time.perf_counter()
        data = converter.circuit_to_einsum(
            lambda: layered_large_circuit(x, layers, n_qubits)
        )
        elapsed = time.perf_counter() - start
        assert len(data["operations"]) == n_operations
        timings.append(elapsed)

    return {
        "n_qubits": n_qubits,
        "layers": layers,
        "gates": n_operations,
        "repeats": repeats,
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "n_qubits",
        "layers",
        "gates",
        "repeats",
        "median_seconds",
        "mean_seconds",
        "min_seconds",
        "max_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["gates"]))
    gates = [int(row["gates"]) for row in rows]
    medians_ms = [1000.0 * float(row["median_seconds"]) for row in rows]
    labels = [f'{int(row["n_qubits"])}q/{int(row["layers"])}L' for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(gates, medians_ms, marker="o", linewidth=2)
    for i, (gate, median_ms, label) in enumerate(zip(gates, medians_ms, labels)):
        if i == len(gates) - 1:
            ax.annotate(
                label,
                (gate, median_ms),
                textcoords="offset points",
                xytext=(-4, 8),
                ha="right",
            )
        else:
            ax.annotate(
                label,
                (gate, median_ms),
                textcoords="offset points",
                xytext=(4, 6),
            )

    ax.set_title("PennyLane conversion time vs gate count")
    ax.set_xlabel("Gate count")
    ax.set_ylabel("Median conversion time (ms)")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.05, y=0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PennyLane circuit_to_einsum conversion time."
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Benchmark case formatted as NQUBITSxLAYERS, for example 8x12.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/benchmarks/pennylane_convert_large_circuits.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("docs/assets/pennylane_convert_large_circuits.png"),
    )
    args = parser.parse_args()

    cases = parse_cases(args.case) if args.case else DEFAULT_CASES
    rows = [benchmark_case(n_qubits, layers, args.repeats) for n_qubits, layers in cases]

    write_csv(rows, args.csv)
    write_plot(rows, args.plot)

    for row in rows:
        print(
            f'{int(row["n_qubits"]):2d} qubits, {int(row["layers"]):2d} layers, '
            f'{int(row["gates"]):4d} gates: '
            f'{1000.0 * float(row["median_seconds"]):8.3f} ms median'
        )
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
