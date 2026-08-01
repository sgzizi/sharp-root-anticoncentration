#!/usr/bin/env python3
"""Illustrate singular coefficient geometry and regular root translation.

The figure is an illustration of an exact one-scalar parameterization, not a
numerical proof of singularity.  It uses the fixed monic polynomial

    psi(gamma) = (gamma + 2)(gamma - 1/2)(gamma - 3/2)

and a uniform common offset V.  No roots are estimated numerically: the roots
and ascending coefficients are part of the deterministic fixture.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharp_root_ac.common_offset import (
    evaluate_polynomial,
    normalized_subleading_curve,
    shift_coefficients,
    translate_roots,
)


FIXTURE_ROOTS = np.array([-2.0, 0.5, 1.5], dtype=float)
# (x + 2)(x - 1/2)(x - 3/2), in ascending-power order.  Its zero
# next-to-leading coefficient makes the polynomial coefficient curve visibly
# curved even over the modest offset range used below.
FIXTURE_COEFFICIENTS = np.array([1.5, -3.25, 0.0, 1.0], dtype=float)
DISPLAYED_ROOT_INDEX = 1


def verify_fixture(eta: float) -> None:
    """Validate the fixture's shift and root-translation identities."""

    evaluation_points = np.linspace(-3.0, 3.0, 17)
    for offset in np.linspace(-eta, eta, 5):
        shifted = shift_coefficients(FIXTURE_COEFFICIENTS, float(offset))
        np.testing.assert_allclose(
            evaluate_polynomial(shifted, evaluation_points),
            evaluate_polynomial(FIXTURE_COEFFICIENTS, evaluation_points + offset),
            rtol=2e-14,
            atol=2e-14,
        )
        translated = translate_roots(FIXTURE_ROOTS, float(offset))
        np.testing.assert_allclose(
            evaluate_polynomial(shifted, translated),
            0.0,
            rtol=0.0,
            atol=2e-14,
        )


def build_figure(
    *,
    samples: int = 50_000,
    eta: float = 0.25,
    seed: int = 17,
) -> plt.Figure:
    """Build the deterministic coefficient-curve/root-law illustration."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if eta <= 0:
        raise ValueError("eta must be positive")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    offset_grid = np.linspace(-eta, eta, 401)
    curve = normalized_subleading_curve(FIXTURE_COEFFICIENTS, offset_grid)

    rng = np.random.default_rng(seed)
    sampled_offsets = rng.uniform(-eta, eta, size=samples)
    selected_root = FIXTURE_ROOTS[DISPLAYED_ROOT_INDEX]
    translated_samples = translate_roots(selected_root * np.ones(samples), sampled_offsets)

    blue = "#1f5aa6"
    orange = "#d97706"
    gray = "#4b5563"
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))

    left = axes[0]
    left.plot(curve[:, 0], curve[:, 1], color=blue, linewidth=2.2)
    marker_offsets = np.array([-eta, 0.0, eta])
    marker_curve = normalized_subleading_curve(FIXTURE_COEFFICIENTS, marker_offsets)
    left.scatter(
        marker_curve[:, 0],
        marker_curve[:, 1],
        color=[gray, orange, gray],
        edgecolor="white",
        linewidth=0.7,
        s=42,
        zorder=3,
    )
    labels = (r"$V=-\eta$", r"$V=0$", r"$V=\eta$")
    offsets_xy = ((5, 7), (6, 8), (-44, 7))
    for point, label, text_offset in zip(marker_curve, labels, offsets_xy, strict=True):
        left.annotate(
            label,
            xy=point,
            xytext=text_offset,
            textcoords="offset points",
            fontsize=9,
        )
    left.set_xlabel(r"$b_{m-1}(V)/b_m$")
    left.set_ylabel(r"$b_{m-2}(V)/b_m$")
    left.set_title("Shifted coefficient curve", fontsize=11)
    left.grid(alpha=0.18, linewidth=0.6)

    right = axes[1]
    support_left = selected_root - eta
    support_right = selected_root + eta
    exact_density = 1.0 / (2.0 * eta)
    right.hist(
        translated_samples,
        bins=36,
        range=(support_left, support_right),
        density=True,
        color=blue,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.35,
        label="sample histogram",
    )
    right.plot(
        [support_left, support_left, support_right, support_right],
        [0.0, exact_density, exact_density, 0.0],
        color=orange,
        linewidth=2.1,
        label=r"exact density $1/(2\eta)$",
    )
    right.set_xlim(support_left - 0.08, support_right + 0.08)
    right.set_ylim(0.0, exact_density * 1.28)
    right.set_xlabel(r"translated root $z_2-V$")
    right.set_ylabel("density")
    right.set_title("Rigidly translated root law", fontsize=11)
    right.legend(frameon=False, fontsize=8.5, loc="upper right")
    right.grid(axis="y", alpha=0.18, linewidth=0.6)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.17, wspace=0.30)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "common_offset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_fixture(args.eta)
    figure = build_figure(samples=args.samples, eta=args.eta, seed=args.seed)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    figure_base = args.figures_dir / "common_offset_demo"
    figure.savefig(
        figure_base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "sharp-root-anticoncentration artifact",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(figure_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)

    offset_grid = np.linspace(-args.eta, args.eta, 401)
    curve = normalized_subleading_curve(FIXTURE_COEFFICIENTS, offset_grid)
    curve_path = args.results_dir / "coefficient_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["offset", "normalized_b_m_minus_1", "normalized_b_m_minus_2"])
        writer.writerows(zip(offset_grid, curve[:, 0], curve[:, 1], strict=True))

    summary_path = args.results_dir / "configuration.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["parameter", "value"])
        writer.writerows(
            [
                ("fixture_roots", ";".join(map(str, FIXTURE_ROOTS))),
                ("displayed_root", FIXTURE_ROOTS[DISPLAYED_ROOT_INDEX]),
                ("eta", args.eta),
                ("exact_root_density", 1.0 / (2.0 * args.eta)),
                ("identity_check_offsets", 5),
                ("identity_check_grid_points", 17),
                ("samples", args.samples),
                ("seed", args.seed),
            ]
        )
    print(f"wrote {figure_base.with_suffix('.pdf')} and {figure_base.with_suffix('.png')}")
    print(f"wrote tables to {args.results_dir}")


if __name__ == "__main__":
    main()
