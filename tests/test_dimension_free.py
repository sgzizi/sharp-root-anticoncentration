import math

import pytest

from sharp_root_ac.incidence import (
    dimension_free_exact_probability,
    dimension_free_exact_ratio,
    dimension_free_hit,
    estimate_dimension_free_ratio,
    wilson_score_interval,
)


@pytest.mark.parametrize(
    ("delta", "epsilon", "probability", "ratio"),
    [
        (0.25, 0.025, 0.05, 1.0),
        (0.5, 0.1, 0.1, 0.5),
        (1.0, 1.0, 0.5, 0.25),
    ],
)
def test_dimension_free_closed_form(delta, epsilon, probability, ratio):
    assert dimension_free_exact_probability(delta, epsilon) == pytest.approx(probability)
    assert dimension_free_exact_ratio(delta) == pytest.approx(ratio)
    assert probability / (2.0 * epsilon) == pytest.approx(ratio)


def test_hit_test_uses_alpha_one_over_alpha_two_indexing():
    # x=-delta*alpha_1/alpha_2=-0.025 lies in [-0.03,0.03].
    assert dimension_free_hit(0.1, 1.0, delta=0.25, epsilon=0.03)
    # Swapping the mathematical coordinates gives x=-2.5, outside the interval.
    assert not dimension_free_hit(1.0, 0.1, delta=0.25, epsilon=0.03)
    # No division occurs when alpha_2 is exactly zero.
    assert not dimension_free_hit(0.1, 0.0, delta=0.25, epsilon=0.03)
    assert dimension_free_hit(0.0, 0.0, delta=0.25, epsilon=0.03)


def test_wilson_interval_known_value_and_ratio_rescaling():
    low, high = wilson_score_interval(5_000, 100_000, confidence=0.95)
    assert low == pytest.approx(0.0486663893, abs=1e-10)
    assert high == pytest.approx(0.0513681825, abs=1e-10)
    assert low / 0.05 == pytest.approx(0.973327786, abs=1e-9)
    assert high / 0.05 == pytest.approx(1.02736365, abs=1e-8)


@pytest.mark.parametrize("dimension", [2, 5, 20, 100])
def test_seeded_monte_carlo_agrees_with_exact_dimension_free_ratio(dimension):
    result = estimate_dimension_free_ratio(
        dimension,
        delta=0.25,
        epsilon=0.025,
        samples=100_000,
        seed=10_000 + dimension,
    )
    assert result.exact_probability == pytest.approx(0.05)
    assert result.exact_ratio == pytest.approx(1.0)
    assert result.ratio_interval[0] <= result.exact_ratio <= result.ratio_interval[1]
    standard_error = math.sqrt(0.05 * 0.95 / result.samples) / (2.0 * result.epsilon)
    assert abs(result.ratio_estimate - result.exact_ratio) <= 4.0 * standard_error


@pytest.mark.parametrize(
    "call",
    [
        lambda: dimension_free_exact_probability(0.0, 0.1),
        lambda: dimension_free_exact_probability(0.1, 0.2),
        lambda: estimate_dimension_free_ratio(1, 0.25, 0.025),
        lambda: estimate_dimension_free_ratio(2, 0.25, 0.025, samples=0),
        lambda: wilson_score_interval(2, 1),
    ],
)
def test_invalid_dimension_free_inputs_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()
