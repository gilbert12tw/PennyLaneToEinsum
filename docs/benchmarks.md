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

## QAE cuQuantum Plan-Reuse and Batching Benchmarks

The current GPU benchmark separates conversion, host-to-device transfer,
cuTensorNet network construction, path planning, and repeated execution. It
compares a reusable cuTensorNet plan with cuQuantum one-shot contraction,
PennyLane `lightning.tensor`, and PennyLane `lightning.gpu`.

The batching workload uses per-sample QAE inputs with shared VQC weights and a
batched Pauli-Z expectation output. It reports both fixed-input replay and new
input tensors, and compares true batched contraction with cuQuantum and
PennyLane sample loops plus `lightning.gpu` broadcasting.

Key results:

- Reusing the plan improves cuQuantum execution by roughly 72x to 137x over
  one-shot contraction in the tested cases.
- At 16 qubits and VQC depth 4, reusable cuQuantum execution is 5.29x faster
  than `lightning.gpu` and 278.7x faster than `lightning.tensor`.
- For new inputs, shared weights, and batch size 256, cuQuantum reaches roughly
  95,627 samples/s at 16 qubits and depth 1, versus 273 samples/s for
  `lightning.gpu` broadcasting.
- Batch performance can be non-monotonic because different batch dimensions
  select different contraction paths. The report includes the unfavorable
  cases rather than assuming that larger batches are always faster.

See the full methodology, limitations, tables, plots, and raw-data index in
[`qae_reuse_batching_report_zh.md`](qae_reuse_batching_report_zh.md).
