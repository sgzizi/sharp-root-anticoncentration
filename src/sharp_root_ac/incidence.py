"""Numerical checks for the coefficient-body incidence formulas.

The routines in this module deliberately mirror two calculations in the
paper.  They are small, deterministic apart from the explicitly seeded Monte
Carlo routine, and have no dependency beyond the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isfinite
from random import Random
from statistics import NormalDist


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _dimension_free_parameters(delta: float, epsilon: float) -> tuple[float, float]:
    delta = _positive_finite(delta, "delta")
    epsilon = _positive_finite(epsilon, "epsilon")
    if epsilon > delta:
        raise ValueError("epsilon must not exceed delta for the exact formula")
    return delta, epsilon


def dimension_free_exact_probability(delta: float, epsilon: float) -> float:
    """Return the exact hit probability for the symmetric test interval.

    For ``F(x) = (1, x / delta, 0, ..., 0)`` and a coefficient vector
    uniform on ``[-1, 1]^N``, the root is

    ``x = -delta * alpha_1 / alpha_2``.

    On ``[-epsilon, epsilon]``, with ``0 < epsilon <= delta``, the hit
    probability is ``epsilon / (2 * delta)``.  The mathematical subscripts
    ``alpha_1, alpha_2`` correspond to Python coordinates 0 and 1.
    """

    delta, epsilon = _dimension_free_parameters(delta, epsilon)
    return epsilon / (2.0 * delta)


def dimension_free_exact_ratio(delta: float) -> float:
    """Return the exact interval-hitting ratio in Equation (19)."""

    delta = _positive_finite(delta, "delta")
    return 1.0 / (4.0 * delta)


def dimension_free_hit(
    alpha_1: float,
    alpha_2: float,
    delta: float,
    epsilon: float,
) -> bool:
    """Test the root event without dividing by a possibly tiny coefficient."""

    delta, epsilon = _dimension_free_parameters(delta, epsilon)
    return abs(delta * float(alpha_1)) <= epsilon * abs(float(alpha_2))


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie between zero and trials")
    confidence = float(confidence)
    if not isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2_over_n = z * z / trials
    denominator = 1.0 + z2_over_n
    center = (proportion + z2_over_n / 2.0) / denominator
    half_width = (
        z
        * (
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


@dataclass(frozen=True)
class DimensionFreeEstimate:
    """Result of a seeded cube Monte Carlo experiment."""

    dimension: int
    delta: float
    epsilon: float
    samples: int
    hits: int
    confidence: float
    probability_estimate: float
    probability_interval: tuple[float, float]
    ratio_estimate: float
    ratio_interval: tuple[float, float]
    exact_probability: float
    exact_ratio: float


def estimate_dimension_free_ratio(
    dimension: int,
    delta: float,
    epsilon: float,
    *,
    samples: int = 100_000,
    seed: int | None = 0,
    confidence: float = 0.95,
) -> DimensionFreeEstimate:
    """Estimate the Equation (19) hitting ratio by cube Monte Carlo.

    Only the first two cube coordinates are sampled because all remaining
    coordinates multiply zero features.  ``dimension`` is nevertheless kept
    in the interface and result so a reproduction script can sweep the
    ambient dimension.  Use distinct seeds if statistically independent
    estimates across dimensions are desired.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer")
    if samples <= 0:
        raise ValueError("samples must be positive")
    delta, epsilon = _dimension_free_parameters(delta, epsilon)

    generator = Random(seed)
    hits = 0
    threshold = epsilon / delta
    for _ in range(samples):
        alpha_1 = generator.uniform(-1.0, 1.0)
        alpha_2 = generator.uniform(-1.0, 1.0)
        # This inequality is equivalent to testing the root location, but is
        # numerically stable even when alpha_2 is very close to zero.
        hits += abs(alpha_1) <= threshold * abs(alpha_2)

    probability = hits / samples
    probability_interval = wilson_score_interval(hits, samples, confidence)
    interval_length = 2.0 * epsilon
    ratio_interval = tuple(endpoint / interval_length for endpoint in probability_interval)
    return DimensionFreeEstimate(
        dimension=dimension,
        delta=delta,
        epsilon=epsilon,
        samples=samples,
        hits=hits,
        confidence=float(confidence),
        probability_estimate=probability,
        probability_interval=probability_interval,
        ratio_estimate=probability / interval_length,
        ratio_interval=ratio_interval,
        exact_probability=dimension_free_exact_probability(delta, epsilon),
        exact_ratio=dimension_free_exact_ratio(delta),
    )


def _rbf_parameters(p: float, p_prime: float, radius: float) -> tuple[float, float, float]:
    p = float(p)
    p_prime = float(p_prime)
    radius = _positive_finite(radius, "radius")
    if not isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if not isfinite(p_prime):
        raise ValueError("p_prime must be finite")
    return p, p_prime, radius


def rbf_cube_incidence_speed(p: float, p_prime: float, radius: float = 1.0) -> float:
    """Return the closed-form cube incidence speed in Equation (56).

    Here ``F=(p,1-p)`` and ``F'=(p_prime,-p_prime)``.  Although ``radius``
    sets the size of ``Q_R=[-R,R]^2``, it cancels from the normalized speed.
    """

    p, p_prime, _ = _rbf_parameters(p, p_prime, radius)
    return abs(p_prime) / (4.0 * max(p, 1.0 - p) ** 2)


def rbf_cube_section_integral_numeric(
    p: float,
    p_prime: float,
    radius: float = 1.0,
    *,
    panels: int = 4096,
) -> float:
    """Numerically integrate the numerator of the two-dimensional section.

    The central section is parameterized by
    ``a(z)=(-(1-p)z,pz)`` for
    ``|z| <= radius/max(p,1-p)``.  Composite Simpson quadrature is applied
    directly to ``|<a(z),F'>| dH^1(a)``.  ``panels`` must be a positive
    multiple of four so the kink at ``z=0`` is a panel boundary.
    """

    p, p_prime, radius = _rbf_parameters(p, p_prime, radius)
    if isinstance(panels, bool) or not isinstance(panels, int):
        raise TypeError("panels must be an integer")
    if panels <= 0 or panels % 4:
        raise ValueError("panels must be a positive multiple of four")

    bound = radius / max(p, 1.0 - p)
    step = 2.0 * bound / panels
    feature_norm = hypot(p, 1.0 - p)

    values = []
    for index in range(panels + 1):
        z = -bound + index * step
        weight = 1 if index in (0, panels) else (4 if index % 2 else 2)
        integrand = abs(z * p_prime) * feature_norm
        values.append(weight * integrand)
    return step * fsum(values) / 3.0


def rbf_cube_incidence_speed_numeric(
    p: float,
    p_prime: float,
    radius: float = 1.0,
    *,
    panels: int = 4096,
) -> float:
    """Evaluate Equation (56) from its section integral definition."""

    p, p_prime, radius = _rbf_parameters(p, p_prime, radius)
    feature_norm = hypot(p, 1.0 - p)
    section_integral = rbf_cube_section_integral_numeric(
        p, p_prime, radius, panels=panels
    )
    cube_area = (2.0 * radius) ** 2
    return section_integral / (cube_area * feature_norm)


__all__ = [
    "DimensionFreeEstimate",
    "dimension_free_exact_probability",
    "dimension_free_exact_ratio",
    "dimension_free_hit",
    "estimate_dimension_free_ratio",
    "rbf_cube_incidence_speed",
    "rbf_cube_incidence_speed_numeric",
    "rbf_cube_section_integral_numeric",
    "wilson_score_interval",
]
