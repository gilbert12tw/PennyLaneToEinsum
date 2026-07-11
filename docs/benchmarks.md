# Conversion Benchmarks

This benchmark measures PennyLane-to-einsum conversion time only. It does not
measure tensor contraction time.

The benchmark circuit uses larger synthetic layered circuits:

- `RX`, `RY`, and `RZ` on every qubit
- A ring of `CNOT` gates
- A nearest-neighbor chain of `CZ` gates

Each layer has `5 * n_qubits - 1` gates.

## Run

```bash
uv run --with matplotlib scripts/benchmark_pennylane_convert.py
```

Outputs:

- Raw CSV: `docs/benchmarks/pennylane_convert_large_circuits.csv`
- Plot: `docs/assets/pennylane_convert_large_circuits.png`

## Latest Local Result

Generated in this workspace on 2026-06-14.

![PennyLane conversion benchmark](assets/pennylane_convert_large_circuits.png)

| Qubits | Layers | Gates | Median conversion time |
|---:|---:|---:|---:|
| 4 | 5 | 95 | 1.738 ms |
| 6 | 8 | 232 | 4.237 ms |
| 8 | 12 | 468 | 8.175 ms |
| 10 | 16 | 784 | 14.391 ms |
| 12 | 20 | 1180 | 21.012 ms |
| 14 | 24 | 1656 | 29.656 ms |

On this run, conversion time scales roughly linearly with gate count and remains
under 30 ms at 1656 gates. This suggests conversion itself is
unlikely to be the bottleneck for QK-style experiments compared with repeated
contraction or model training work.

## Batch Contraction Benchmark

This benchmark checks whether contracting one batched einsum network is faster
than contracting each sample independently. Conversion is performed before the
timed region, so the `einsum contract` numbers measure `contract_einsum()` only.

It also includes PennyLane `default.qubit` statevector execution as a reference
point. The two timings are not the same operation:

- `einsum contract` measures opt_einsum contraction of already-built networks.
- `PennyLane state` measures PennyLane QNode execution returning `qml.state()`.

Run:

```bash
uv run --with matplotlib scripts/benchmark_batch_contraction.py
```

Outputs:

- Raw CSV: `docs/benchmarks/batch_contraction.csv`
- Plot: `docs/assets/batch_contraction_benchmark.png`

Latest local result uses 8 qubits, 8 layers, and 312 gates per sample.

![Batch contraction benchmark](assets/batch_contraction_benchmark.png)

| Batch | Einsum batched contract | Einsum loop contract | Einsum speedup | PennyLane batched state | PennyLane loop state | PennyLane speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.875 ms | 12.110 ms | 0.87x | 14.614 ms | 11.081 ms | 0.76x |
| 2 | 14.065 ms | 24.005 ms | 1.71x | 15.789 ms | 22.190 ms | 1.41x |
| 4 | 48.956 ms | 47.895 ms | 0.98x | 17.513 ms | 44.545 ms | 2.54x |
| 8 | 86.252 ms | 98.777 ms | 1.15x | 20.861 ms | 88.329 ms | 4.23x |
| 16 | 154.000 ms | 191.760 ms | 1.25x | 27.726 ms | 178.606 ms | 6.44x |
| 32 | 144.044 ms | 385.424 ms | 2.68x | 40.420 ms | 375.832 ms | 9.30x |

For this circuit, batched contraction is not a clean B-times speedup. It becomes
faster than looped contraction for larger batches, but the extra batch dimension
changes tensor shapes and contraction paths. The B=32 case shows about 2.7x
faster contraction than contracting samples one by one.

## QAE-Net cuQuantum Benchmark

This benchmark implements Fig. 1 of arXiv:2507.11217 with Hadamard preparation,
per-qubit `RZ-RY-RZ` encoding, repeated trainable `RZ-RY-RZ` rotations, and a
linear CNOT chain. It intentionally omits the closing `CNOT(n-1, 0)`.

The run used one NVIDIA H200 (143771 MiB), CUDA 12.6, Python 3.11,
cuQuantum 26.6.0, CuPy 14.1.1, and PennyLane 0.42.3 on nano4. Slurm jobs used
the internal `GOV109211` account and the `dev` partition.

Run:

```bash
sbatch scripts/slurm_qae_benchmark.sh
sbatch scripts/slurm_qae_comparison.sh
```

Raw successful-run data:

- cuQuantum scaling and three-backend comparison: Slurm job `177851`
- `docs/benchmarks/qae_cuquantum_expval_scaling_177851.csv`
- `docs/benchmarks/qae_cuquantum_state_scaling_177851.csv`
- `docs/benchmarks/qae_backend_comparison_177851.csv`

### Maximum Qubits

| Output | Largest successful case | Median contraction | Result at next case |
|---|---:|---:|---|
| Pauli-Z expectation | 64 qubits | 280.94 ms | Not reached; 64 was the configured scan limit |
| Full statevector | 30 qubits | 92.96 ms | 32 qubits: GPU out of memory |

The expectation result is a lower bound, not the hardware limit. Its product
initial state and scalar output avoid allocating a `2**n` statevector, and GPU
memory remained around 643 MB at 64 qubits for this shallow linear-chain
circuit. A follow-up scan above 64 qubits is needed to locate its actual limit.

The full-state result is constrained by the exponential output and contraction
workspace. At 30 qubits the measured GPU allocation was about 54.6 GB. The
32-qubit contraction failed while requesting another 70.9 GB after about
139.6 GB had already been allocated.

### Backend Comparison

Median execution times from job `177851` are shown below. Conversion time is
excluded. `pennylane-tensornet` is `default.tensor(method="tn")`, while
`pennylane-statevector` is `lightning.gpu`.

| Qubits | Output | cuQuantum | PennyLane tensor | PennyLane statevector |
|---:|---|---:|---:|---:|
| 4 | expval | 24.66 ms | 11.82 ms | 1.83 ms |
| 12 | expval | 48.95 ms | 22.46 ms | 3.63 ms |
| 20 | expval | 76.57 ms | 34.29 ms | 6.25 ms |
| 24 | expval | 91.42 ms | 41.10 ms | 29.60 ms |
| 4 | state | 13.72 ms | 6.62 ms | 2.24 ms |
| 12 | state | 27.85 ms | 18.55 ms | 4.23 ms |
| 20 | state | 42.30 ms | 40.82 ms | 11.25 ms |
| 24 | state | 50.10 ms | 204.85 ms | 99.71 ms |

All compared state amplitudes and Pauli-Z expectations agree numerically. For
this depth-one linear circuit, direct statevector simulation is fastest through
20 qubits. At 24 qubits cuQuantum is faster for full-state output, while
PennyLane's tensor backend remains faster for the scalar expectation benchmark.
