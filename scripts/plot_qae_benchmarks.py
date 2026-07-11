from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "cuquantum": "#d97706",
    "pennylane-tensornet": "#2563eb",
    "pennylane-statevector": "#059669",
}
LABELS = {
    "cuquantum": "cuQuantum contract",
    "pennylane-tensornet": "PennyLane tensor network",
    "pennylane-statevector": "PennyLane lightning.gpu",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def plot_backend_comparison(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, measurement, title in zip(
        axes,
        ("expval", "state"),
        ("Pauli-Z expectation", "Full statevector"),
    ):
        for method in COLORS:
            selected = [
                row
                for row in rows
                if row["measurement"] == measurement
                and row["method"] == method
                and row["status"] == "ok"
            ]
            axis.plot(
                [int(row["n_qubits"]) for row in selected],
                [1000 * float(row["median_seconds"]) for row in selected],
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.set_title(title)
        axis.set_xlabel("Qubits")
        axis.set_xticks(sorted({int(row["n_qubits"]) for row in rows}))
    axes[0].set_ylabel("Median execution time (ms), lower is better")
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle("QAE-Net backend comparison on one NVIDIA H200", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_scaling(
    expval_rows: list[dict[str, str]], state_rows: list[dict[str, str]], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    expval_ok = [row for row in expval_rows if row["status"] == "ok"]
    state_ok = [row for row in state_rows if row["status"] == "ok"]
    axis.plot(
        [int(row["n_qubits"]) for row in expval_ok],
        [1000 * float(row["median_seconds"]) for row in expval_ok],
        marker="o",
        linewidth=2,
        color="#7c3aed",
        label="Pauli-Z expectation",
    )
    axis.plot(
        [int(row["n_qubits"]) for row in state_ok],
        [1000 * float(row["median_seconds"]) for row in state_ok],
        marker="s",
        linewidth=2,
        color="#d97706",
        label="Full statevector",
    )
    failed = [row for row in state_rows if row["status"] != "ok"]
    if failed:
        failed_qubit = int(failed[0]["n_qubits"])
        axis.scatter(
            [failed_qubit],
            [100],
            marker="X",
            s=90,
            color="#dc2626",
            zorder=5,
            label=f"Statevector OOM ({failed_qubit} qubits)",
        )
    axis.set_title("cuQuantum scaling for the QAE-Net circuit")
    axis.set_xlabel("Qubits")
    axis.set_ylabel("Median contraction time (ms)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_memory(
    expval_rows: list[dict[str, str]], state_rows: list[dict[str, str]], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    for rows, label, marker, color in (
        (expval_rows, "Pauli-Z expectation", "o", "#7c3aed"),
        (state_rows, "Full statevector", "s", "#d97706"),
    ):
        selected = [
            row
            for row in rows
            if row["status"] == "ok" and row.get("gpu_memory_used")
        ]
        axis.plot(
            [int(row["n_qubits"]) for row in selected],
            [float(row["gpu_memory_used"]) / 2**30 for row in selected],
            marker=marker,
            linewidth=2,
            color=color,
            label=label,
        )
    axis.axhline(
        float(expval_rows[0]["gpu_memory_total"]) / 2**30,
        color="#64748b",
        linestyle="--",
        label="H200 reported capacity",
    )
    axis.set_yscale("log")
    axis.set_title("GPU memory after contraction")
    axis.set_xlabel("Qubits")
    axis.set_ylabel("Allocated GPU memory (GiB, log scale)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="177851")
    parser.add_argument("--data-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = read_rows(
        args.data_dir / f"qae_backend_comparison_{args.job_id}.csv"
    )
    expval = read_rows(
        args.data_dir / f"qae_cuquantum_expval_scaling_{args.job_id}.csv"
    )
    state = read_rows(
        args.data_dir / f"qae_cuquantum_state_scaling_{args.job_id}.csv"
    )

    configure_style()
    plot_backend_comparison(comparison, args.output_dir / "qae_backend_comparison.png")
    plot_scaling(expval, state, args.output_dir / "qae_cuquantum_scaling.png")
    plot_memory(expval, state, args.output_dir / "qae_gpu_memory.png")


if __name__ == "__main__":
    main()
