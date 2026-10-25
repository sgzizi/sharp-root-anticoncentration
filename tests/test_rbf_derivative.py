from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from sharp_root_ac.harmonic import (  # noqa: E402
    cube_incidence_speed,
    grounded_system_with_derivative,
    harmonic_scores,
    harmonic_scores_with_derivative,
    rbf_weight_matrices,
)


def fixed_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array(
        [
            [-1.0, 0.0],
            [-0.6, 0.4],
            [0.0, -0.2],
            [0.7, 0.3],
            [1.2, -0.1],
        ]
    )
    d = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    d /= np.max(d)
    np.fill_diagonal(d, 0.0)
    labeled = np.array([0, 4])
    labels = np.array([0.0, 1.0])
    return d, labeled, labels


def test_complete_graph_and_grounded_degree_include_labeled_edges() -> None:
    d, labeled, labels = fixed_problem()
    gamma = 0.9
    w, w_prime = rbf_weight_matrices(d, gamma)
    assert np.allclose(np.diag(w), 0.0)
    assert np.allclose(np.diag(w_prime), 0.0)
    assert np.all(w[~np.eye(w.shape[0], dtype=bool)] > 0.0)

    q, b, q_prime, b_prime, unlabeled = grounded_system_with_derivative(
        d, gamma, labeled, labels
    )
    expected_q = np.diag(np.sum(w, axis=1)[unlabeled]) - w[np.ix_(unlabeled, unlabeled)]
    expected_b = w[np.ix_(unlabeled, labeled)] @ labels
    expected_q_prime = np.diag(np.sum(w_prime, axis=1)[unlabeled]) - w_prime[
        np.ix_(unlabeled, unlabeled)
    ]
    expected_b_prime = w_prime[np.ix_(unlabeled, labeled)] @ labels
    assert np.allclose(q, expected_q)
    assert np.allclose(b, expected_b)
    assert np.allclose(q_prime, expected_q_prime)
    assert np.allclose(b_prime, expected_b_prime)
    assert np.min(np.linalg.eigvalsh(q)) > 0.0


def test_analytic_harmonic_derivative_matches_centered_difference() -> None:
    d, labeled, labels = fixed_problem()
    gamma = 1.1
    step = 1e-6
    unlabeled, scores, derivative = harmonic_scores_with_derivative(
        d, gamma, labeled, labels
    )
    unlabeled_plus, scores_plus = harmonic_scores(d, gamma + step, labeled, labels)
    unlabeled_minus, scores_minus = harmonic_scores(d, gamma - step, labeled, labels)
    finite_difference = (scores_plus - scores_minus) / (2.0 * step)

    assert np.array_equal(unlabeled, unlabeled_plus)
    assert np.array_equal(unlabeled, unlabeled_minus)
    assert np.allclose(derivative, finite_difference, atol=2e-8, rtol=2e-6)
    assert np.all((scores >= -1e-12) & (scores <= 1.0 + 1e-12))


def test_two_dimensional_cube_incidence_formula() -> None:
    scores = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    derivative = np.array([-0.4, 0.1, -0.2, 0.3, 0.0])
    expected = np.abs(derivative) / (
        4.0 * np.maximum(scores, 1.0 - scores) ** 2
    )
    actual = cube_incidence_speed(scores, derivative)
    assert np.allclose(actual, expected)
    assert np.all(actual <= np.abs(derivative) + 1e-15)
# Maintenance annotation; functional content is unchanged.
