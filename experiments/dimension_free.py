#!/usr/bin/env python3
"""Reproduce the dimension-free interval-hitting calculation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharp_root_ac.incidence import estimate_dimension_free_ratio


DEFAULT_DIMENSIONS = (2, 5, 10, 20, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--figures-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    estimates = [
        estimate_dimension_free_ratio(
            dimension,
            args.delta,
            args.epsilon,
            samples=args.samples,
            seed=args.seed + dimension,
        )
        for dimension in DEFAULT_DIMENSIONS
    ]

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "dimension_free.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dimension",
                "delta",
                "epsilon",
                "samples",
                "hits",
                "probability_estimate",
                "exact_probability",
                "ratio_estimate",
                "ratio_ci_low",
                "ratio_ci_high",
                "exact_ratio",
                "seed",
            ]
        )
        for estimate in estimates:
            writer.writerow(
                [
                    estimate.dimension,
                    estimate.delta,
                    estimate.epsilon,
                    estimate.samples,
                    estimate.hits,
                    estimate.probability_estimate,
                    estimate.exact_probability,
                    estimate.ratio_estimate,
                    estimate.ratio_interval[0],
                    estimate.ratio_interval[1],
                    estimate.exact_ratio,
                    args.seed + estimate.dimension,
                ]
            )

    dimensions = [item.dimension for item in estimates]
    ratios = [item.ratio_estimate for item in estimates]
    lower = [item.ratio_estimate - item.ratio_interval[0] for item in estimates]
    upper = [item.ratio_interval[1] - item.ratio_estimate for item in estimates]
    exact = estimates[0].exact_ratio

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(5.6, 3.25), constrained_layout=True)
    axis.axhline(exact, color="black", linewidth=1.35, label=r"exact $1/(4\delta)$")
    axis.errorbar(
        dimensions,
        ratios,
        yerr=[lower, upper],
        color="#1f4ba8",
        marker="o",
        markersize=4.5,
        linestyle="none",
        capsize=3,
        label="Monte Carlo (pointwise 95% Wilson interval)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(dimensions, [str(value) for value in dimensions])
    axis.set_xlabel(r"ambient coefficient dimension $N$")
    axis.set_ylabel(r"estimated hitting ratio $\widehat c_N(\varepsilon)$")
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.legend(frameon=False, loc="best")
    axis.margins(x=0.04, y=0.20)

    for suffix in ("pdf", "png"):
        metadata = (
            {
                "Creator": "sharp-root-anticoncentration artifact",
                "CreationDate": None,
                "ModDate": None,
            }
            if suffix == "pdf"
            else None
        )
        fig.savefig(
            args.figures_dir / f"dimension_free.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(fig)

    print("Dimension-free cube experiment")
    print(f"delta={args.delta:g}, epsilon={args.epsilon:g}, samples={args.samples:,}")
    print("   N    estimate       95% CI        exact")
    for item in estimates:
        print(
            f"{item.dimension:4d}    {item.ratio_estimate:8.5f}   "
            f"[{item.ratio_interval[0]:.5f}, {item.ratio_interval[1]:.5f}]   "
            f"{item.exact_ratio:.5f}"
        )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
# Maintenance annotation; functional content is unchanged.
