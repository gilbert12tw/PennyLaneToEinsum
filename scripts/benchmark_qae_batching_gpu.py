from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import pennylane as qml

from pennylane_einsum import CircuitToEinsum, CuQuantumContractor
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


def stats(samples: list[float], batch_size: int) -> dict[str, float]:
    median = statistics.median(samples)
    return {
        "median_seconds": median,
        "p25_seconds": float(np.percentile(samples, 25)),
        "p75_seconds": float(np.percentile(samples, 75)),
        "samples_per_second": batch_size / median,
        "milliseconds_per_sample": 1000.0 * median / batch_size,
    }


def network_for(inputs: np.ndarray, weights: np.ndarray):
    n_qubits = inputs.shape[-2]
    converter = CircuitToEinsum.for_qubits(n_qubits)
    data = converter.circuit_to_einsum(lambda: qae_circuit(inputs, weights))
    return converter.generate_expectation_einsum(data, {0: "Z"})


def make_contractor(
    expr: str, tensors: list[np.ndarray], optimizer_samples: int = 0
):
    import cupy as cp

    device_tensors = [cp.asarray(tensor) for tensor in tensors]
    contractor = CuQuantumContractor(
        expr, device_tensors, options={"blocking": True, "memory_limit": "80%"}
    )
    optimize = {"samples": optimizer_samples} if optimizer_samples else None
    contractor.contract_path(optimize=optimize)
    return contractor, device_tensors


def benchmark_cuquantum_batch(
    input_batch: np.ndarray,
    weights: np.ndarray,
    repeats: int,
    warmups: int,
    optimizer_samples: int,
):
    expr, tensors = network_for(input_batch, weights)
    contractor, device_tensors = make_contractor(expr, tensors, optimizer_samples)
    try:
        samples = measure(contractor.contract, repeats, warmups)
        result = contractor.contract().get()
    finally:
        contractor.free()
    return stats(samples, len(input_batch)), np.asarray(result), device_tensors


def benchmark_cuquantum_updates(
    input_batches: list[np.ndarray],
    weights: np.ndarray,
    repeats: int,
    warmups: int,
    optimizer_samples: int,
):
    import cupy as cp

    networks = [network_for(inputs, weights) for inputs in input_batches]
    expr, initial_tensors = networks[0]
    contractor, device_tensors = make_contractor(
        expr, initial_tensors, optimizer_samples
    )
    update_tensors = [tensors for _, tensors in networks[1:]]
    changed = [
        index
        for index in range(len(initial_tensors))
        if any(
            not np.array_equal(initial_tensors[index], tensors[index])
            for tensors in update_tensors
        )
    ]
    iteration = 0

    def update_and_contract():
        nonlocal iteration
        tensors = update_tensors[iteration % len(update_tensors)]
        for index in changed:
            device_tensors[index].set(tensors[index])
        result = contractor.contract()
        iteration += 1
        return result

    try:
        samples = measure(update_and_contract, repeats, warmups)
        result = update_and_contract().get()
    finally:
        contractor.free()
    metrics = stats(samples, len(input_batches[0]))
    metrics["updated_tensor_count"] = len(changed)
    return metrics, np.asarray(result)


def benchmark_cuquantum_loop(
    input_batch: np.ndarray,
    weights: np.ndarray,
    repeats: int,
    warmups: int,
    optimizer_samples: int,
):
    import cupy as cp

    networks = [network_for(sample, weights) for sample in input_batch]
    expr, first = networks[0]
    all_device_tensors = [[cp.asarray(tensor) for tensor in tensors] for _, tensors in networks]
    contractor = CuQuantumContractor(
        expr,
        all_device_tensors[0],
        options={"blocking": True, "memory_limit": "80%"},
    )
    optimize = {"samples": optimizer_samples} if optimizer_samples else None
    contractor.contract_path(optimize=optimize)

    def run_loop():
        outputs = []
        for device_tensors in all_device_tensors:
            contractor.reset_operands(device_tensors)
            outputs.append(contractor.contract())
        return outputs

    try:
        samples = measure(run_loop, repeats, warmups)
        outputs = run_loop()
        synchronize()
        result = cp.stack(outputs).get()
    finally:
        contractor.free()
    return stats(samples, len(input_batch)), np.asarray(result)


def benchmark_pennylane(
    method: str,
    input_batch: np.ndarray,
    weights: np.ndarray,
    repeats: int,
    warmups: int,
):
    n_qubits = input_batch.shape[1]
    if method in ("lightning-tensor-loop", "lightning-tensor-batch"):
        device = qml.device(
            "lightning.tensor",
            wires=n_qubits,
            method="tn",
            backend="cutensornet",
            c_dtype=np.complex128,
        )
    elif method in ("lightning-gpu-batch", "lightning-gpu-loop"):
        device = qml.device("lightning.gpu", wires=n_qubits, c_dtype=np.complex128)
    else:
        raise ValueError(method)

    @qml.qnode(device)
    def circuit(inputs):
        qae_circuit(inputs, weights)
        return qml.expval(qml.PauliZ(0))

    if method.endswith("-loop"):
        call = lambda: [circuit(sample) for sample in input_batch]
    else:
        call = lambda: circuit(input_batch)
    samples = measure(call, repeats, warmups)
    result = np.asarray(call())
    return stats(samples, len(input_batch)), result


def reference_values(input_batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_qubits = input_batch.shape[1]
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit(inputs):
        qae_circuit(inputs, weights)
        return qml.expval(qml.PauliZ(0))

    return np.asarray([circuit(sample) for sample in input_batch])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", nargs="+", type=int, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "cuquantum-batch-reuse",
            "cuquantum-batch-update",
            "cuquantum-loop-reuse",
        ],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--optimizer-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for n_qubits in args.qubits:
        for layers in args.layers:
            base_inputs, weights = make_parameters(n_qubits, layers)
            for batch_size in args.batch_sizes:
                rng = np.random.default_rng(2026 + n_qubits * 1000 + layers * 100 + batch_size)
                input_batch = base_inputs + rng.uniform(
                    -0.5, 0.5, size=(batch_size, n_qubits, 3)
                )
                update_batches = [
                    base_inputs
                    + rng.uniform(-0.5, 0.5, size=(batch_size, n_qubits, 3))
                    for _ in range(6)
                ]
                reference = reference_values(input_batch, weights)
                for method in args.methods:
                    row = {
                        "n_qubits": n_qubits,
                        "layers": layers,
                        "batch_size": batch_size,
                        "method": method,
                        "measurement": "expval-z0",
                        "shared_weights": True,
                        "plan_reused": method.startswith("cuquantum"),
                        "repeats": args.repeats,
                        "warmups": args.warmups,
                    }
                    try:
                        if method == "cuquantum-batch-reuse":
                            metrics, result, _ = benchmark_cuquantum_batch(
                                input_batch,
                                weights,
                                args.repeats,
                                args.warmups,
                                args.optimizer_samples,
                            )
                        elif method == "cuquantum-batch-update":
                            metrics, result = benchmark_cuquantum_updates(
                                update_batches,
                                weights,
                                args.repeats,
                                args.warmups,
                                args.optimizer_samples,
                            )
                            result_reference = reference_values(
                                update_batches[
                                    1
                                    + (args.warmups + args.repeats)
                                    % (len(update_batches) - 1)
                                ],
                                weights,
                            )
                        elif method == "cuquantum-loop-reuse":
                            metrics, result = benchmark_cuquantum_loop(
                                input_batch,
                                weights,
                                args.repeats,
                                args.warmups,
                                args.optimizer_samples,
                            )
                        else:
                            metrics, result = benchmark_pennylane(
                                method,
                                input_batch,
                                weights,
                                args.repeats,
                                args.warmups,
                            )
                        if method != "cuquantum-batch-update":
                            result_reference = reference
                        error = float(
                            np.max(
                                np.abs(
                                    np.asarray(result).reshape(-1)
                                    - result_reference.reshape(-1)
                                )
                            )
                        )
                        if error > 1e-8:
                            raise AssertionError(f"max_abs_error={error}")
                        row.update(status="ok", error="", max_abs_error=error, **metrics)
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
