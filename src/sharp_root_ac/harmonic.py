"""Grounded harmonic extension for the paper's complete Gaussian--RBF graph.

The routines in this module deliberately mirror the notation in Section 5 of
the paper.  In particular, the graph is complete with zero diagonal, the
degree of an unlabeled vertex includes its edges to labeled vertices, and all
linear systems are solved directly (no matrix inverse or pseudoinverse).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


Array = np.ndarray


def _validate_problem(
    dissimilarities: Array,
    labeled_indices: Iterable[int],
    labeled_values: Array,
) -> tuple[Array, Array, Array, Array]:
    d = np.asarray(dissimilarities, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        raise ValueError("dissimilarities must be a square matrix of size at least two")
    if not np.all(np.isfinite(d)):
        raise ValueError("dissimilarities must be finite")
    if np.min(d) < -1e-12:
        raise ValueError("dissimilarities must be nonnegative")
    if not np.allclose(d, d.T, atol=1e-12, rtol=1e-12):
        raise ValueError("dissimilarities must be symmetric")
    if not np.allclose(np.diag(d), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("the dissimilarity diagonal must be zero")

    labeled = np.asarray(tuple(labeled_indices), dtype=int)
    if labeled.ndim != 1 or labeled.size == 0:
        raise ValueError("at least one labeled vertex is required")
    if np.any(labeled < 0) or np.any(labeled >= d.shape[0]):
        raise IndexError("a labeled vertex index is out of range")
    if np.unique(labeled).size != labeled.size:
        raise ValueError("labeled vertex indices must be distinct")

    y_l = np.asarray(labeled_values, dtype=float)
    if y_l.shape != (labeled.size,):
        raise ValueError("labeled_values must correspond to labeled_indices")
    if not np.all(np.isfinite(y_l)) or np.any((y_l < 0.0) | (y_l > 1.0)):
        raise ValueError("labeled values must lie in [0, 1]")

    is_labeled = np.zeros(d.shape[0], dtype=bool)
    is_labeled[labeled] = True
    unlabeled = np.flatnonzero(~is_labeled)
    if unlabeled.size == 0:
        raise ValueError("at least one unlabeled vertex is required")
    return d, labeled, y_l, unlabeled


def rbf_weight_matrices(dissimilarities: Array, gamma: float) -> tuple[Array, Array]:
    """Return the complete-graph RBF adjacency ``W`` and derivative ``W'``.

    Off-diagonal entries are ``exp(-gamma * d[a,b])`` and diagonal entries are
    exactly zero.  The derivative is with respect to ``gamma``.
    """

    d = np.asarray(dissimilarities, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("dissimilarities must be square")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be a finite nonnegative number")
    if np.min(d) < -1e-12 or not np.all(np.isfinite(d)):
        raise ValueError("dissimilarities must be finite and nonnegative")
    if not np.allclose(d, d.T, atol=1e-12, rtol=1e-12):
        raise ValueError("dissimilarities must be symmetric")

    w = np.exp(-gamma * d)
    np.fill_diagonal(w, 0.0)
    w_prime = -d * w
    np.fill_diagonal(w_prime, 0.0)
    return w, w_prime


def grounded_system_with_derivative(
    dissimilarities: Array,
    gamma: float,
    labeled_indices: Iterable[int],
    labeled_values: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Construct ``Q, b, Q', b'`` and return the unlabeled vertex indices."""

    d, labeled, y_l, unlabeled = _validate_problem(
        dissimilarities, labeled_indices, labeled_values
    )
    w, w_prime = rbf_weight_matrices(d, gamma)

    # These are degrees in the full graph.  In particular, the diagonal of Q
    # includes every U--L edge, as required for the grounded Laplacian.
    degree = np.sum(w, axis=1)
    degree_prime = np.sum(w_prime, axis=1)
    q = np.diag(degree[unlabeled]) - w[np.ix_(unlabeled, unlabeled)]
    q_prime = np.diag(degree_prime[unlabeled]) - w_prime[
        np.ix_(unlabeled, unlabeled)
    ]
    b = w[np.ix_(unlabeled, labeled)] @ y_l
    b_prime = w_prime[np.ix_(unlabeled, labeled)] @ y_l
    return q, b, q_prime, b_prime, unlabeled


def harmonic_scores(
    dissimilarities: Array,
    gamma: float,
    labeled_indices: Iterable[int],
    labeled_values: Array,
) -> tuple[Array, Array]:
    """Solve the grounded harmonic system and return ``(U, s_U(gamma))``."""

    q, b, _, _, unlabeled = grounded_system_with_derivative(
        dissimilarities, gamma, labeled_indices, labeled_values
    )
    scores = np.linalg.solve(q, b)
    return unlabeled, scores


def harmonic_scores_with_derivative(
    dissimilarities: Array,
    gamma: float,
    labeled_indices: Iterable[int],
    labeled_values: Array,
) -> tuple[Array, Array, Array]:
    """Return harmonic scores and their analytic derivative.

    The derivative is obtained from

    ``Q s' = b' - Q' s``.
    """

    q, b, q_prime, b_prime, unlabeled = grounded_system_with_derivative(
        dissimilarities, gamma, labeled_indices, labeled_values
    )
    scores = np.linalg.solve(q, b)
    score_derivative = np.linalg.solve(q, b_prime - q_prime @ scores)
    return unlabeled, scores, score_derivative


def cost_sensitive_predictions(
    scores: Array,
    false_negative_cost: float,
    false_positive_cost: float,
) -> Array:
    """Apply the paper's cost-sensitive rule, with ties assigned label one."""

    s = np.asarray(scores, dtype=float)
    c_fn = float(false_negative_cost)
    c_fp = float(false_positive_cost)
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if c_fn < 0.0 or c_fp < 0.0 or not np.isfinite(c_fn + c_fp):
        raise ValueError("costs must be finite and nonnegative")
    margin = (c_fn + c_fp) * s - c_fp
    return (margin >= 0.0).astype(np.int8)


def cost_sensitive_utility(
    predictions: Array,
    true_labels: Array,
    false_negative_cost: float,
    false_positive_cost: float,
    cost_scale: float,
) -> float:
    """Return the normalized utility from equation ``eq:rbf-utility``."""

    predicted = np.asarray(predictions, dtype=np.int8)
    truth = np.asarray(true_labels, dtype=np.int8)
    if predicted.shape != truth.shape or predicted.ndim != 1 or predicted.size == 0:
        raise ValueError("predictions and true_labels must be nonempty matching vectors")
    if np.any((predicted != 0) & (predicted != 1)) or np.any(
        (truth != 0) & (truth != 1)
    ):
        raise ValueError("predictions and labels must be binary")
    c_fn = float(false_negative_cost)
    c_fp = float(false_positive_cost)
    r = float(cost_scale)
    if not np.isfinite(c_fn + c_fp + r) or c_fn < 0.0 or c_fp < 0.0 or r <= 0.0:
        raise ValueError("costs must be nonnegative and cost_scale must be positive")
    if c_fn > r + 1e-12 or c_fp > r + 1e-12:
        raise ValueError("costs must not exceed cost_scale")

    false_negatives = np.count_nonzero((truth == 1) & (predicted == 0))
    false_positives = np.count_nonzero((truth == 0) & (predicted == 1))
    loss = c_fn * false_negatives + c_fp * false_positives
    return float(1.0 - loss / (r * truth.size))


def cube_incidence_speed(scores: Array, score_derivative: Array) -> Array:
    """Evaluate the exact two-dimensional cube-incidence speed.

    For ``F=(s,1-s)``, this is
    ``|s'| / (4 * max(s, 1-s)^2)``.
    """

    s = np.asarray(scores, dtype=float)
    s_prime = np.asarray(score_derivative, dtype=float)
    if s.shape != s_prime.shape:
        raise ValueError("scores and derivatives must have the same shape")
    denominator = 4.0 * np.maximum(s, 1.0 - s) ** 2
    return np.abs(s_prime) / denominator
# Maintenance annotation; functional content is unchanged.
