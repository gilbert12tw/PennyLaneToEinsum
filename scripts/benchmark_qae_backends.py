from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, contract_einsum
from pennylane_einsum.qae_circuit import make_parameters, qae_circuit


def synchronize() -> None:
    try:
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    except ImportError:
        pass


def timed(call, repeats: int) -> tuple[float, object]:
    samples = []
    result = None
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        result = call()
        synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def gpu_memory() -> tuple[int | None, int | None]:
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        return total - free, total
    except ImportError:
        return None, None


def build_network(n_qubits: int, layers: int, measurement: str):
    inputs, weights = make_parameters(n_qubits, layers)
    converter = CircuitToEinsum.for_qubits(n_qubits)
    start = time.perf_counter()
    data = converter.circuit_to_einsum(lambda: qae_circuit(inputs, weights))
    if measurement == "state":
        network = converter.generate_full_einsum(data)
    else:
        network = converter.generate_expectation_einsum(data, {0: "Z"})
    return network, time.perf_counter() - start, inputs, weights


def run_ours(n_qubits: int, layers: int, measurement: str, repeats: int):
    (expr, tensors), conversion, _, _ = build_network(n_qubits, layers, measurement)
    call = lambda: contract_einsum(expr, tensors, backend="cuquantum")
    call()
    elapsed, result = timed(call, repeats)
    value = result.reshape(-1)[0].get()
    del result
    return elapsed, conversion, value


def run_pennylane(
    n_qubits: int, layers: int, method: str, measurement: str, repeats: int
):
    _, _, inputs, weights = build_network(n_qubits, layers, measurement)
    if method == "pennylane-tensornet":
        device = qml.device("default.tensor", wires=n_qubits, method="tn")
    elif method == "pennylane-statevector":
        device = qml.device("lightning.gpu", wires=n_qubits)
    else:
        raise ValueError(method)

    @qml.qnode(device)
    def circuit():
        qae_circuit(inputs, weights)
        if measurement == "state":
            return qml.state()
        return qml.expval(qml.PauliZ(0))

    circuit()
    elapsed, result = timed(circuit, repeats)
    return elapsed, 0.0, np.asarray(result).reshape(-1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", nargs="+", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cuquantum", "pennylane-tensornet", "pennylane-statevector"],
    )
    parser.add_argument("--measurements", nargs="+", default=["expval", "state"])
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks/qae_backends.csv")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_qubits in args.qubits:
        for measurement in args.measurements:
            for method in args.methods:
                row = {
                    "job_id": os.environ.get("SLURM_JOB_ID", "local"),
                    "host": platform.node(),
                    "n_qubits": n_qubits,
                    "layers": args.layers,
                    "method": method,
                    "measurement": measurement,
                    "repeats": args.repeats,
                }
                try:
                    if method == "cuquantum":
                        elapsed, conversion, value = run_ours(
                            n_qubits, args.layers, measurement, args.repeats
                        )
                    else:
                        elapsed, conversion, value = run_pennylane(
                            n_qubits, args.layers, method, measurement, args.repeats
                        )
                    used, total = gpu_memory()
                    row.update(
                        status="ok",
                        median_seconds=elapsed,
                        conversion_seconds=conversion,
                        value_real=float(np.real(value)),
                        gpu_memory_used=used,
                        gpu_memory_total=total,
                        error="",
                    )
                except Exception as exc:
                    row.update(status="error", error=repr(exc))
                rows.append(row)
                print(row, flush=True)
                gc.collect()
                try:
                    import cupy as cp

                    cp.get_default_memory_pool().free_all_blocks()
                except ImportError:
                    pass

    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
