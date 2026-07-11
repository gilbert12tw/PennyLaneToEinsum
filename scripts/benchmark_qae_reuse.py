from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, CuQuantumContractor, contract_einsum
from pennylane_einsum.qae_circuit import make_parameters, qae_circuit


def synchronize() -> None:
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()


def measure(call, repeats: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        call()
        synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_seconds": statistics.median(ordered),
        "min_seconds": ordered[0],
        "p25_seconds": float(np.percentile(ordered, 25)),
        "p75_seconds": float(np.percentile(ordered, 75)),
        "max_seconds": ordered[-1],
    }


def build_network(n_qubits: int, layers: int, measurement: str):
    inputs, weights = make_parameters(n_qubits, layers)
    converter = CircuitToEinsum.for_qubits(n_qubits)
    start = time.perf_counter()
    data = converter.circuit_to_einsum(lambda: qae_circuit(inputs, weights))
    if measurement == "expval":
        expr, tensors = converter.generate_expectation_einsum(data, {0: "Z"})
    else:
        expr, tensors = converter.generate_full_einsum(data)
    return expr, tensors, inputs, weights, time.perf_counter() - start


def benchmark_cuquantum(
    expr: str,
    tensors: list[np.ndarray],
    repeats: int,
    warmups: int,
    optimizer_samples: int,
    autotune_iterations: int,
):
    import cupy as cp

    synchronize()
    start = time.perf_counter()
    device_tensors = [cp.asarray(tensor) for tensor in tensors]
    synchronize()
    h2d = time.perf_counter() - start

    start = time.perf_counter()
    contractor = CuQuantumContractor(
        expr, device_tensors, options={"blocking": True, "memory_limit": "80%"}
    )
    construction = time.perf_counter() - start
    try:
        optimize = {"samples": optimizer_samples} if optimizer_samples else None
        start = time.perf_counter()
        _, info = contractor.contract_path(optimize=optimize)
        planning = time.perf_counter() - start

        autotune = 0.0
        if autotune_iterations:
            start = time.perf_counter()
            contractor.autotune(iterations=autotune_iterations)
            synchronize()
            autotune = time.perf_counter() - start

        samples = measure(contractor.contract, repeats, warmups)
        result = contractor.contract()
        synchronize()
        value = complex(result.reshape(-1)[0].get())
        largest = getattr(info, "largest_intermediate", None)
        flops = getattr(info, "opt_cost", None)
    finally:
        contractor.free()

    return {
        **summary(samples),
        "h2d_seconds": h2d,
        "network_construction_seconds": construction,
        "path_planning_seconds": planning,
        "autotune_seconds": autotune,
        "largest_intermediate": largest,
        "estimated_flops": flops,
        "value_real": value.real,
        "value_imag": value.imag,
    }


def benchmark_one_shot(
    expr: str, tensors: list[np.ndarray], repeats: int, warmups: int
):
    samples = measure(
        lambda: contract_einsum(expr, tensors, backend="cuquantum"), repeats, warmups
    )
    return summary(samples)


def benchmark_pennylane(
    method: str,
    measurement: str,
    inputs: np.ndarray,
    weights: np.ndarray,
    repeats: int,
    warmups: int,
):
    n_qubits = inputs.shape[0]
    if method == "lightning-tensor":
        device = qml.device(
            "lightning.tensor",
            wires=n_qubits,
            method="tn",
            backend="cutensornet",
            c_dtype=np.complex128,
        )
    elif method == "lightning-gpu":
        device = qml.device("lightning.gpu", wires=n_qubits, c_dtype=np.complex128)
    else:
        raise ValueError(method)

    @qml.qnode(device)
    def circuit():
        qae_circuit(inputs, weights)
        if measurement == "expval":
            return qml.expval(qml.PauliZ(0))
        return qml.state()

    samples = measure(circuit, repeats, warmups)
    value = complex(np.asarray(circuit()).reshape(-1)[0])
    return {**summary(samples), "value_real": value.real, "value_imag": value.imag}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", nargs="+", type=int, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--measurements", nargs="+", default=["expval"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cuquantum-reuse", "cuquantum-one-shot", "lightning-tensor"],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--optimizer-samples", type=int, default=0)
    parser.add_argument("--autotune-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for n_qubits in args.qubits:
        for layers in args.layers:
            for measurement in args.measurements:
                expr, tensors, inputs, weights, conversion = build_network(
                    n_qubits, layers, measurement
                )
                for method in args.methods:
                    row = {
                        "n_qubits": n_qubits,
                        "layers": layers,
                        "measurement": measurement,
                        "method": method,
                        "repeats": args.repeats,
                        "warmups": args.warmups,
                        "conversion_seconds": conversion if method.startswith("cuquantum") else 0.0,
                        "plan_reused": method == "cuquantum-reuse",
                    }
                    try:
                        if method == "cuquantum-reuse":
                            metrics = benchmark_cuquantum(
                                expr,
                                tensors,
                                args.repeats,
                                args.warmups,
                                args.optimizer_samples,
                                args.autotune_iterations,
                            )
                        elif method == "cuquantum-one-shot":
                            metrics = benchmark_one_shot(
                                expr, tensors, args.repeats, args.warmups
                            )
                        else:
                            metrics = benchmark_pennylane(
                                method,
                                measurement,
                                inputs,
                                weights,
                                args.repeats,
                                args.warmups,
                            )
                        row.update(status="ok", error="", **metrics)
                    except Exception as exc:
                        row.update(status="error", error=repr(exc))
                    rows.append(row)
                    print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
