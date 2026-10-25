#!/usr/bin/env python3
"""Reproduce the cost-sensitive Gaussian--RBF graph-SSL illustration.

The experiment uses a fixed (oblivious) sequence of synthetic base instances.
Only the cost pairs and Hedge actions are resampled.  For every horizon, Hedge
uses its own mesh ``h=T^{-1/2}``.  Reported ``grid_regret`` is exact for the
algorithm grid; ``dense_proxy_regret`` uses a finite evaluation grid and is
not presented as exact regret to the continuous comparator.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from sharp_root_ac.harmonic import (  # noqa: E402
    cube_incidence_speed,
    harmonic_scores,
    harmonic_scores_with_derivative,
)
from sharp_root_ac.hedge import (  # noqa: E402
    FullInformationHedge,
    dense_proxy_regret,
    exact_grid_regret,
    parameter_grid,
)


@dataclass(frozen=True)
class BaseInstance:
    dissimilarities: np.ndarray
    labels: np.ndarray
    labeled: np.ndarray
    unlabeled: np.ndarray


@dataclass(frozen=True)
class TransitionSelection:
    round_index: int
    local_vertex: int
    maximum_sampled_sign_changes: int
    selected_sampled_sign_changes: int
    rule: str


GRID_REGRET_LABEL = "Regret to algorithm grid (exact)"
DENSE_PROXY_LABEL = "Dense-plus-algorithm-grid finite proxy"
CI_METHOD = "pointwise normal approximation: mean +/- 1.96 sample standard errors"
CI_CONDITIONING = "conditional on the fixed oblivious base-instance sequence"
CI_RANDOMNESS = "independent cost pairs and Hedge actions"
TRANSITION_LOCATION_METHOD = (
    "bisection of the harmonic margin within adjacent dense-grid sign-change brackets"
)


def select_illustrative_transition(
    margins: np.ndarray,
    gammas: np.ndarray,
) -> TransitionSelection:
    """Apply and report the deterministic illustrative-curve selection rule."""

    if gammas.ndim != 1 or gammas.size < 2 or np.any(np.diff(gammas) <= 0.0):
        raise ValueError("gammas must be a strictly increasing one-dimensional grid")
    if (
        margins.ndim != 3
        or margins.shape[0] == 0
        or margins.shape[1] != gammas.size
        or margins.shape[2] == 0
    ):
        raise ValueError("margins must have shape (round, gamma, vertex)")
    sign_changes = np.sum(margins[:, :-1, :] * margins[:, 1:, :] < 0.0, axis=1)
    maximum_changes = int(np.max(sign_changes))
    if maximum_changes > 0:
        midpoint = 0.5 * (gammas[0] + gammas[-1])
        candidates = np.argwhere(sign_changes == maximum_changes)

        def centrality(candidate: np.ndarray) -> float:
            t, vertex = (int(candidate[0]), int(candidate[1]))
            changes = np.flatnonzero(
                margins[t, :-1, vertex] * margins[t, 1:, vertex] < 0.0
            )
            approximate_crossings = 0.5 * (gammas[changes] + gammas[changes + 1])
            return float(np.min(np.abs(approximate_crossings - midpoint)))

        round_index, local_vertex = map(int, min(candidates, key=centrality))
        rule = (
            "maximize sampled sign changes over all round-vertex curves; "
            "break ties by the crossing nearest the parameter-interval midpoint, "
            "then by round and local-vertex index"
        )
    else:
        closest = np.min(np.abs(margins), axis=1)
        flat_index = int(np.argmin(closest))
        round_index, local_vertex = map(
            int, np.unravel_index(flat_index, closest.shape)
        )
        rule = (
            "no sampled sign change found; choose the round-vertex curve with "
            "the smallest sampled absolute margin"
        )
    return TransitionSelection(
        round_index=round_index,
        local_vertex=local_vertex,
        maximum_sampled_sign_changes=maximum_changes,
        selected_sampled_sign_changes=int(sign_changes[round_index, local_vertex]),
        rule=rule,
    )


def generate_base_instances(
    count: int,
    n: int,
    labeled_count: int,
    distance_cap: float,
    cluster_separation: float,
    cluster_noise: float,
    seed: int,
) -> list[BaseInstance]:
    """Generate and then fix an oblivious sequence of two-cluster instances."""

    if n < 4 or labeled_count < 2 or labeled_count >= n:
        raise ValueError("use n>=4 and 2<=labeled_count<n")
    if distance_cap <= 0.0 or cluster_separation <= 0.0 or cluster_noise <= 0.0:
        raise ValueError("distance and cluster scales must be positive")

    rng = np.random.default_rng(seed)
    class_sizes = (n // 2, n - n // 2)
    instances: list[BaseInstance] = []
    for _ in range(count):
        labels = np.concatenate(
            (np.zeros(class_sizes[0], dtype=np.int8), np.ones(class_sizes[1], dtype=np.int8))
        )
        centers = np.where(labels[:, None] == 0, (-cluster_separation / 2.0, 0.0),
                           (cluster_separation / 2.0, 0.0))
        points = centers + cluster_noise * rng.standard_normal((n, 2))
        permutation = rng.permutation(n)
        points = points[permutation]
        labels = labels[permutation]

        squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        maximum = float(np.max(squared))
        if maximum <= 0.0:
            raise RuntimeError("the generated point cloud is degenerate")
        dissimilarities = distance_cap * squared / maximum
        np.fill_diagonal(dissimilarities, 0.0)

        # Stratification prevents an uninformative all-zero or all-one labeled
        # boundary in the illustration.  It is still a fixed legal instance.
        labeled_zero_count = max(1, labeled_count // 2)
        labeled_one_count = labeled_count - labeled_zero_count
        if labeled_one_count == 0:
            labeled_one_count = 1
            labeled_zero_count -= 1
        zeros = np.flatnonzero(labels == 0)
        ones = np.flatnonzero(labels == 1)
        if labeled_zero_count > zeros.size or labeled_one_count > ones.size:
            raise ValueError("labeled_count is incompatible with the two class sizes")
        labeled = np.concatenate(
            (
                rng.choice(zeros, size=labeled_zero_count, replace=False),
                rng.choice(ones, size=labeled_one_count, replace=False),
            )
        )
        labeled.sort()
        mask = np.ones(n, dtype=bool)
        mask[labeled] = False
        unlabeled = np.flatnonzero(mask)
        instances.append(
            BaseInstance(dissimilarities, labels, labeled, unlabeled)
        )
    return instances


def score_tensor(instances: list[BaseInstance], gammas: np.ndarray) -> np.ndarray:
    """Precompute harmonic scores; these do not depend on the random costs."""

    unlabeled_count = instances[0].unlabeled.size
    scores = np.empty((len(instances), gammas.size, unlabeled_count), dtype=float)
    for t, instance in enumerate(instances):
        y_l = instance.labels[instance.labeled]
        for j, gamma in enumerate(gammas):
            unlabeled, value = harmonic_scores(
                instance.dissimilarities, gamma, instance.labeled, y_l
            )
            if not np.array_equal(unlabeled, instance.unlabeled):
                raise RuntimeError("the unlabeled vertex order changed")
            scores[t, j] = value
    return scores


def utility_matrix(
    scores: np.ndarray,
    instances: list[BaseInstance],
    false_negative_costs: np.ndarray,
    false_positive_costs: np.ndarray,
    cost_scale: float,
) -> np.ndarray:
    """Vectorized evaluation of the exact normalized cost-sensitive utility."""

    rounds, _, unlabeled_count = scores.shape
    if len(instances) != rounds:
        raise ValueError("the score tensor and instance sequence disagree")
    c_fn = np.asarray(false_negative_costs, dtype=float)
    c_fp = np.asarray(false_positive_costs, dtype=float)
    if c_fn.shape != (rounds,) or c_fp.shape != (rounds,):
        raise ValueError("one cost pair is required per round")
    if cost_scale <= 0.0:
        raise ValueError("cost_scale must be positive")
    if (
        np.any(c_fn < 0.0)
        or np.any(c_fp < 0.0)
        or np.any(c_fn > cost_scale)
        or np.any(c_fp > cost_scale)
    ):
        raise ValueError("costs must lie in [0,cost_scale]")

    truth = np.stack([instance.labels[instance.unlabeled] for instance in instances])
    if truth.shape != (rounds, unlabeled_count):
        raise RuntimeError("unlabeled set sizes must be constant")
    margins = (
        (c_fn + c_fp)[:, None, None] * scores - c_fp[:, None, None]
    )
    predictions = margins >= 0.0  # the fixed tie rule predicts one
    false_negatives = np.sum((truth[:, None, :] == 1) & ~predictions, axis=2)
    false_positives = np.sum((truth[:, None, :] == 0) & predictions, axis=2)
    loss = c_fn[:, None] * false_negatives + c_fp[:, None] * false_positives
    return 1.0 - loss / (cost_scale * unlabeled_count)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    horizons = sorted({int(row["horizon"]) for row in rows})
    for horizon in horizons:
        selected = [row for row in rows if int(row["horizon"]) == horizon]
        entry: dict[str, object] = {
            "horizon": horizon,
            "runs": len(selected),
            "ci_level": 0.95,
            "ci_method": CI_METHOD,
            "ci_conditioning": CI_CONDITIONING,
            "ci_randomness": CI_RANDOMNESS,
        }
        for key in ("grid_regret", "dense_proxy_regret", "proxy_grid_gap"):
            values = np.asarray([float(row[key]) for row in selected])
            standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
            entry[f"mean_{key}"] = float(np.mean(values))
            entry[f"ci95_{key}"] = 1.96 * standard_error
        result.append(entry)
    return result


def save_figure(figure: plt.Figure, output_base: Path) -> None:
    """Write matching vector and raster copies of a figure."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "sharp-root-anticoncentration artifact",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")


def plot_regret(summary: list[dict[str, object]], output_base: Path) -> None:
    horizons = np.asarray([row["horizon"] for row in summary])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    styles = (
        ("grid_regret", GRID_REGRET_LABEL, "o"),
        ("dense_proxy_regret", DENSE_PROXY_LABEL, "s"),
    )
    for key, label, marker in styles:
        mean = np.asarray([row[f"mean_{key}"] for row in summary])
        ci = np.asarray([row[f"ci95_{key}"] for row in summary])
        axes[0].errorbar(horizons, mean, yerr=ci, marker=marker, capsize=3, label=label)
        axes[1].errorbar(
            horizons,
            mean / np.sqrt(horizons),
            yerr=ci / np.sqrt(horizons),
            marker=marker,
            capsize=3,
            label=label,
        )
    axes[0].set_xlabel("Horizon $T$")
    axes[0].set_ylabel("Mean realized regret")
    axes[1].set_xlabel("Horizon $T$")
    axes[1].set_ylabel(r"Mean regret$/\sqrt{T}$")
    for axis in axes:
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.legend(frameon=False, fontsize=7.4)
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)


def write_transition_outputs(
    figures_dir: Path,
    results_dir: Path,
    instances: list[BaseInstance],
    gammas: np.ndarray,
    dense_scores: np.ndarray,
    dense_utilities: np.ndarray,
    c_fn: np.ndarray,
    c_fp: np.ndarray,
    distance_cap: float,
    labeled_count: int,
    bandwidth_cap: float,
    reference_horizon: int,
    reference_repetition: int,
) -> None:
    margins = (c_fn + c_fp)[:, None, None] * dense_scores - c_fp[:, None, None]
    selection = select_illustrative_transition(margins, gammas)
    round_index = selection.round_index
    local_vertex = selection.local_vertex

    instance = instances[round_index]
    global_vertex = int(instance.unlabeled[local_vertex])
    y_l = instance.labels[instance.labeled]
    score = np.empty(gammas.size)
    derivative = np.empty(gammas.size)
    for j, gamma in enumerate(gammas):
        unlabeled, values, values_prime = harmonic_scores_with_derivative(
            instance.dissimilarities, gamma, instance.labeled, y_l
        )
        position = int(np.flatnonzero(unlabeled == global_vertex)[0])
        score[j] = values[position]
        derivative[j] = values_prime[position]
    incidence = cube_incidence_speed(score, derivative)
    cost_sum = c_fn[round_index] + c_fp[round_index]
    if cost_sum <= 0.0:
        raise RuntimeError("the null cost pair has no cost-sensitive threshold")
    threshold = c_fp[round_index] / cost_sum
    derivative_bound = (
        (instance.labels.size - 1)
        * distance_cap
        / (labeled_count * math.exp(-bandwidth_cap * distance_cap))
    )
    transition_rows = [
        {
            "gamma": float(gamma),
            "score": float(score_value),
            "score_derivative": float(derivative_value),
            "cost_threshold": float(threshold),
            "margin": float((c_fn[round_index] + c_fp[round_index]) * score_value - c_fp[round_index]),
            "cube_incidence_speed": float(incidence_value),
            "theoretical_derivative_bound": float(derivative_bound),
            "round": round_index,
            "vertex": global_vertex,
        }
        for gamma, score_value, derivative_value, incidence_value in zip(
            gammas, score, derivative, incidence
        )
    ]
    write_csv(
        results_dir / "transition_curve.csv",
        list(transition_rows[0]),
        transition_rows,
    )

    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.plot(gammas, score, color="#1f4e79", label=rf"$s_{{t,v}}(\gamma)$, $t={round_index}$, $v={global_vertex}$")
    axis.axhline(threshold, color="#b22222", linestyle="--", label="Cost threshold")
    selected_margin = (
        (c_fn[round_index] + c_fp[round_index]) * score - c_fp[round_index]
    )
    crossing_indices = np.flatnonzero(
        selected_margin[:-1] * selected_margin[1:] <= 0.0
    )

    def margin_at(gamma: float) -> float:
        unlabeled, values = harmonic_scores(
            instance.dissimilarities,
            gamma,
            instance.labeled,
            y_l,
        )
        position = int(np.flatnonzero(unlabeled == global_vertex)[0])
        return float(cost_sum * values[position] - c_fp[round_index])

    crossing_gammas: list[float] = []
    for index in crossing_indices:
        left = float(gammas[index])
        right = float(gammas[index + 1])
        left_margin = float(selected_margin[index])
        right_margin = float(selected_margin[index + 1])
        if left_margin == 0.0:
            crossing_gamma = left
        elif right_margin == 0.0:
            crossing_gamma = right
        else:
            for _ in range(60):
                center = 0.5 * (left + right)
                center_margin = margin_at(center)
                if abs(center_margin) <= 1e-12 or right - left <= 1e-10:
                    left = right = center
                    break
                if left_margin * center_margin <= 0.0:
                    right = center
                    right_margin = center_margin
                else:
                    left = center
                    left_margin = center_margin
            crossing_gamma = 0.5 * (left + right)
        crossing_gammas.append(crossing_gamma)

    # Adjacent dense-grid brackets can share an exactly sampled zero.
    crossing_gammas = list(
        dict.fromkeys(round(value, 10) for value in crossing_gammas)
    )

    diagnostic_rows = [
        {
            "reference_horizon": reference_horizon,
            "reference_repetition": reference_repetition,
            "round": round_index,
            "vertex": global_vertex,
            "curve_role": "illustrative, not statistically representative",
            "selection_scope": "all round-vertex curves in the reference run",
            "selection_rule": selection.rule,
            "dense_grid_points": gammas.size,
            "dense_grid_spacing": float(gammas[1] - gammas[0]),
            "maximum_sampled_sign_changes": selection.maximum_sampled_sign_changes,
            "selected_sampled_sign_changes": selection.selected_sampled_sign_changes,
            "transition_location_method": TRANSITION_LOCATION_METHOD,
            "transition_location_tolerance": "margin <= 1e-12 or bracket width <= 1e-10",
            "numerically_isolated_transition_count": len(crossing_gammas),
            "numerically_isolated_crossings": ";".join(map(str, crossing_gammas)),
            "max_abs_score_derivative": float(np.max(np.abs(derivative))),
            "theoretical_derivative_bound": float(derivative_bound),
            "max_derivative_to_bound_ratio": float(
                np.max(np.abs(derivative)) / derivative_bound
            ),
            "max_cube_incidence_speed": float(np.max(incidence)),
            "cost_threshold": float(threshold),
            "false_negative_cost": float(c_fn[round_index]),
            "false_positive_cost": float(c_fp[round_index]),
        }
    ]
    write_csv(
        results_dir / "transition_diagnostics.csv",
        list(diagnostic_rows[0]),
        diagnostic_rows,
    )
    if crossing_gammas:
        axis.scatter(
            crossing_gammas,
            [threshold] * len(crossing_gammas),
            s=28,
            color="black",
            zorder=4,
            label="Numerically isolated prediction transition",
        )
    axis.set_xlabel(r"Inverse bandwidth $\gamma$")
    axis.set_ylabel("Harmonic score")
    vertical_min = min(float(np.min(score)), float(threshold))
    vertical_max = max(float(np.max(score)), float(threshold))
    padding = max(0.012, 0.16 * (vertical_max - vertical_min))
    axis.set_ylim(max(0.0, vertical_min - padding), min(1.0, vertical_max + padding))
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir / "transition_curve")
    plt.close(fig)

    utility_change_counts = np.count_nonzero(
        np.abs(np.diff(dense_utilities, axis=1)) > 1e-12,
        axis=1,
    )
    chosen_rounds = [round_index]
    for candidate in np.argsort(-utility_change_counts, kind="stable"):
        value = int(candidate)
        if value not in chosen_rounds:
            chosen_rounds.append(value)
        if len(chosen_rounds) == min(4, len(instances)):
            break
    illustrative_rounds = np.asarray(chosen_rounds, dtype=int)
    utility_selection_rows = [
        {
            "reference_horizon": reference_horizon,
            "reference_repetition": reference_repetition,
            "round": int(t),
            "selection_rank": rank,
            "curve_role": "illustrative, not statistically representative",
            "dense_grid_utility_changes": int(utility_change_counts[t]),
            "selection_rule": (
                "include transition round, then largest dense-grid utility-change counts"
            ),
        }
        for rank, t in enumerate(illustrative_rounds, start=1)
    ]
    write_csv(
        results_dir / "utility_selection.csv",
        list(utility_selection_rows[0]),
        utility_selection_rows,
    )
    utility_rows: list[dict[str, object]] = []
    for j, gamma in enumerate(gammas):
        row: dict[str, object] = {"gamma": float(gamma)}
        for t in illustrative_rounds:
            row[f"utility_round_{t}"] = float(dense_utilities[t, j])
        utility_rows.append(row)
    write_csv(results_dir / "utility_curves.csv", list(utility_rows[0]), utility_rows)

    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    for t in illustrative_rounds:
        axis.step(gammas, dense_utilities[t], where="post", label=f"round {t}")
    axis.set_xlabel(r"Inverse bandwidth $\gamma$")
    axis.set_ylabel("Normalized utility")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir / "utility_curves")
    plt.close(fig)


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or horizons[0] <= 0:
        raise argparse.ArgumentTypeError("horizons must be positive comma-separated integers")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPOSITORY / "figures" / "rbf_graph_ssl",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY / "results" / "rbf_graph_ssl",
    )
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("100,200,500,1000"))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--dense-points", type=int, default=401)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--labeled", type=int, default=5)
    parser.add_argument("--B", type=float, default=3.0)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=1.0)
    parser.add_argument("--cluster-separation", type=float, default=1.25)
    parser.add_argument("--cluster-noise", type=float, default=0.75)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument("--cost-seed", type=int, default=314159)
    parser.add_argument("--hedge-seed", type=int, default=271828)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.dense_points < 3:
        parser.error("repetitions must be positive and dense-points must be at least three")
    if args.B <= 0.0 or args.R <= 0.0 or args.D < 1.0:
        parser.error("B and R must be positive, and D must be at least one")

    figures_dir: Path = args.figures_dir
    results_dir: Path = args.results_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    maximum_horizon = max(args.horizons)
    instances = generate_base_instances(
        maximum_horizon,
        args.n,
        args.labeled,
        args.D,
        args.cluster_separation,
        args.cluster_noise,
        args.base_seed,
    )

    dense_grid = np.linspace(0.0, args.B, args.dense_points)
    dense_scores = score_tensor(instances, dense_grid)
    run_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    example_costs: tuple[np.ndarray, np.ndarray] | None = None
    example_dense_utilities: np.ndarray | None = None

    for horizon in args.horizons:
        grid = parameter_grid(args.B, horizon)
        algorithm_scores = score_tensor(instances[:horizon], grid)
        for repetition in range(args.repetitions):
            cost_rng = np.random.default_rng(
                np.random.SeedSequence([args.cost_seed, horizon, repetition])
            )
            c_fn = cost_rng.uniform(0.0, args.R, size=horizon)
            c_fp = cost_rng.uniform(0.0, args.R, size=horizon)
            grid_utilities = utility_matrix(
                algorithm_scores, instances[:horizon], c_fn, c_fp, args.R
            )
            dense_utilities = utility_matrix(
                dense_scores[:horizon], instances[:horizon], c_fn, c_fp, args.R
            )

            hedge = FullInformationHedge(
                grid,
                horizon,
                seed=np.random.SeedSequence([args.hedge_seed, horizon, repetition]),
            )
            selected_indices: list[int] = []
            selected_gammas: list[float] = []
            received: list[float] = []
            for t in range(horizon):
                # Crucially, act() has no access to grid_utilities[t].
                index, gamma = hedge.act()
                reward = hedge.observe(grid_utilities[t])
                selected_indices.append(index)
                selected_gammas.append(gamma)
                received.append(reward)

            index_array = np.asarray(selected_indices, dtype=int)
            received_array = np.asarray(received, dtype=float)
            grid_regret = exact_grid_regret(grid_utilities, index_array)
            # Include the algorithm grid in the declared proxy, so its
            # comparator is never weaker than the exact grid comparator.
            proxy_utilities = np.concatenate((dense_utilities, grid_utilities), axis=1)
            proxy_regret = dense_proxy_regret(proxy_utilities, received_array)
            best_grid = float(np.max(np.sum(grid_utilities, axis=0)))
            best_proxy = float(np.max(np.sum(proxy_utilities, axis=0)))
            run_rows.append(
                {
                    "horizon": horizon,
                    "repetition": repetition,
                    "mesh": horizon ** -0.5,
                    "grid_size": grid.size,
                    "grid_regret": grid_regret,
                    "dense_proxy_regret": proxy_regret,
                    "proxy_grid_gap": best_proxy - best_grid,
                    "learner_reward": float(np.sum(received_array)),
                    "best_grid_reward": best_grid,
                    "best_proxy_reward": best_proxy,
                }
            )

            if horizon == maximum_horizon and repetition == 0:
                example_costs = (c_fn.copy(), c_fp.copy())
                example_dense_utilities = dense_utilities.copy()
                trace_rows = [
                    {
                        "reference_horizon": maximum_horizon,
                        "reference_repetition": 0,
                        "round": t,
                        "action_index": selected_indices[t],
                        "gamma": selected_gammas[t],
                        "received_utility": received[t],
                        "false_negative_cost": c_fn[t],
                        "false_positive_cost": c_fp[t],
                    }
                    for t in range(horizon)
                ]

    write_csv(results_dir / "regret_runs.csv", list(run_rows[0]), run_rows)
    summary = summarize_runs(run_rows)
    write_csv(results_dir / "regret_summary.csv", list(summary[0]), summary)
    write_csv(results_dir / "online_trace.csv", list(trace_rows[0]), trace_rows)
    plot_regret(summary, figures_dir / "regret_scaling")

    if example_costs is None or example_dense_utilities is None:
        raise RuntimeError("the illustrative run was not recorded")
    write_transition_outputs(
        figures_dir,
        results_dir,
        instances,
        dense_grid,
        dense_scores,
        example_dense_utilities,
        example_costs[0],
        example_costs[1],
        args.D,
        args.labeled,
        args.B,
        maximum_horizon,
        0,
    )

    configuration_rows = [
        {"parameter": "horizons", "value": ";".join(map(str, args.horizons))},
        {"parameter": "maximum_horizon", "value": maximum_horizon},
        {"parameter": "n", "value": args.n},
        {"parameter": "ell", "value": args.labeled},
        {"parameter": "B", "value": args.B},
        {"parameter": "R", "value": args.R},
        {"parameter": "D", "value": args.D},
        {"parameter": "cluster_separation", "value": args.cluster_separation},
        {"parameter": "cluster_noise", "value": args.cluster_noise},
        {"parameter": "kappa_uniform_cost", "value": args.R ** -2},
        {"parameter": "A", "value": 4.0},
        {"parameter": "dense_points", "value": args.dense_points},
        {
            "parameter": "dense_grid_definition",
            "value": "uniform grid of dense_points values on [0,B]",
        },
        {
            "parameter": "algorithm_mesh_rule",
            "value": "h=T^(-1/2), constructed separately for each horizon",
        },
        {
            "parameter": "dense_proxy_definition",
            "value": (
                "finite union of the uniform dense evaluation grid and that "
                "horizon's algorithm grid"
            ),
        },
        {"parameter": "repetitions", "value": args.repetitions},
        {"parameter": "ci_level", "value": 0.95},
        {
            "parameter": "ci_method",
            "value": CI_METHOD,
        },
        {
            "parameter": "ci_conditioning",
            "value": CI_CONDITIONING,
        },
        {
            "parameter": "ci_randomness",
            "value": CI_RANDOMNESS,
        },
        {
            "parameter": "cost_distribution",
            "value": "independent Uniform([0,R]^2) pairs across rounds and repetitions",
        },
        {"parameter": "base_seed", "value": args.base_seed},
        {"parameter": "cost_seed", "value": args.cost_seed},
        {"parameter": "hedge_seed", "value": args.hedge_seed},
        {"parameter": "illustrative_reference_horizon", "value": maximum_horizon},
        {"parameter": "illustrative_reference_repetition", "value": 0},
        {
            "parameter": "illustrative_transition_selection",
            "value": (
                "maximize sampled sign changes across all round-vertex curves; "
                "tie-break by central crossing, then indices"
            ),
        },
        {
            "parameter": "transition_location_method",
            "value": TRANSITION_LOCATION_METHOD,
        },
    ]
    write_csv(results_dir / "configuration.csv", ["parameter", "value"], configuration_rows)
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote tables to {results_dir}")


if __name__ == "__main__":
    main()
# Maintenance annotation; functional content is unchanged.
