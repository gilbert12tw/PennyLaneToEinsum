from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "cuquantum-reuse": "cuQuantum plan reuse",
    "cuquantum-one-shot": "cuQuantum one-shot",
    "lightning-tensor": "PennyLane lightning.tensor",
    "lightning-gpu": "PennyLane lightning.gpu",
    "cuquantum-batch-reuse": "cuQuantum batch, fixed inputs",
    "cuquantum-batch-update": "cuQuantum batch, new inputs",
    "cuquantum-loop-reuse": "cuQuantum sample loop",
    "lightning-tensor-loop": "lightning.tensor loop",
    "lightning-gpu-batch": "lightning.gpu broadcast",
    "lightning-gpu-loop": "lightning.gpu loop",
}
COLORS = {
    "cuquantum-reuse": "#7c3aed",
    "cuquantum-one-shot": "#d97706",
    "lightning-tensor": "#2563eb",
    "lightning-gpu": "#059669",
    "cuquantum-batch-reuse": "#7c3aed",
    "cuquantum-batch-update": "#a855f7",
    "cuquantum-loop-reuse": "#d97706",
    "lightning-tensor-loop": "#2563eb",
    "lightning-gpu-batch": "#059669",
    "lightning-gpu-loop": "#64748b",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def plot_reuse(data: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for axis, layer in zip(axes, (1, 4, 8)):
        selected_layer = [row for row in data if int(row["layers"]) == layer]
        for method in (
            "cuquantum-reuse",
            "cuquantum-one-shot",
            "lightning-tensor",
            "lightning-gpu",
        ):
            selected = [row for row in selected_layer if row["method"] == method]
            if not selected:
                continue
            selected.sort(key=lambda row: int(row["n_qubits"]))
            axis.plot(
                [int(row["n_qubits"]) for row in selected],
                [1000 * float(row["median_seconds"]) for row in selected],
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_title(f"VQC depth {layer}")
        axis.set_xlabel("Qubits")
        axis.set_xticks([8, 16, 24, 32])
    axes[0].set_ylabel("Steady-state median latency (ms, log scale)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("Plan reuse changes the cuQuantum performance regime", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_phases(data: list[dict[str, str]], output: Path) -> None:
    selected = [
        row
        for row in data
        if row["method"] == "cuquantum-reuse" and int(row["layers"]) == 4
    ]
    selected.sort(key=lambda row: int(row["n_qubits"]))
    labels = [str(row["n_qubits"]) for row in selected]
    phases = (
        ("conversion_seconds", "Conversion", "#64748b"),
        ("h2d_seconds", "H2D", "#059669"),
        ("network_construction_seconds", "Network construction", "#d97706"),
        ("path_planning_seconds", "Path planning", "#dc2626"),
        ("median_seconds", "One reused execution", "#7c3aed"),
    )
    x = range(len(selected))
    width = 0.15
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    for index, (field, label, color) in enumerate(phases):
        axis.bar(
            [value + (index - 2) * width for value in x],
            [1000 * float(row[field]) for row in selected],
            width,
            label=label,
            color=color,
        )
    axis.set_yscale("log")
    axis.set_xticks(list(x), labels)
    axis.set_xlabel("Qubits, VQC depth 4")
    axis.set_ylabel("Time (ms, log scale)")
    axis.set_title("cuQuantum setup phases versus reused execution")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_batching(data: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, n_qubits, layers in zip(
        axes.flat,
        (8, 8, 16, 16),
        (1, 4, 1, 4),
    ):
        case = [
            row
            for row in data
            if int(row["n_qubits"]) == n_qubits and int(row["layers"]) == layers
        ]
        for method in (
            "cuquantum-batch-reuse",
            "cuquantum-batch-update",
            "cuquantum-loop-reuse",
            "lightning-tensor-loop",
            "lightning-gpu-batch",
            "lightning-gpu-loop",
        ):
            selected = [row for row in case if row["method"] == method]
            if not selected:
                continue
            selected.sort(key=lambda row: int(row["batch_size"]))
            axis.plot(
                [int(row["batch_size"]) for row in selected],
                [float(row["samples_per_second"]) for row in selected],
                marker="o",
                linewidth=1.8,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"{n_qubits} qubits, depth {layers}")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Throughput (samples/s, log scale)")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper left")
    figure.suptitle("QAE inference batching with shared weights", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="177982")
    parser.add_argument("--data-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reuse = []
    for suffix in ("cuquantum", "lightning_tensor", "lightning_gpu"):
        reuse.extend(rows(args.data_dir / f"qae_reuse_{suffix}_{args.job_id}.csv"))

    batching = []
    for suffix in (
        "cuquantum_batch_reuse",
        "cuquantum_batch_update",
        "cuquantum_loop_reuse",
        "lightning_tensor_loop",
        "lightning_gpu_batch",
        "lightning_gpu_loop",
    ):
        batching.extend(rows(args.data_dir / f"qae_batch_{suffix}_{args.job_id}.csv"))

    configure()
    plot_reuse(reuse, args.output_dir / "qae_plan_reuse_comparison.png")
    plot_phases(reuse, args.output_dir / "qae_cuquantum_phase_breakdown.png")
    plot_batching(batching, args.output_dir / "qae_batching_throughput.png")


if __name__ == "__main__":
    main()
