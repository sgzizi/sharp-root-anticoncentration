from __future__ import annotations

import numpy as np
import pytest

from sharp_root_ac.common_offset import (
    evaluate_polynomial,
    normalized_subleading_curve,
    polynomial_degree,
    shift_coefficients,
    translate_roots,
)


def test_binomial_shift_coefficients_for_quadratic() -> None:
    # p(x) = 2 - 3x + 4x^2 and v = 1/4.
    coefficients = np.array([2.0, -3.0, 4.0])
    shifted = shift_coefficients(coefficients, 0.25)
    np.testing.assert_allclose(shifted, np.array([1.5, -1.0, 4.0]), atol=0.0)


@pytest.mark.parametrize("offset", [-0.75, -0.2, 0.0, 0.35, 1.1])
def test_shift_identity_on_a_deterministic_grid(offset: float) -> None:
    coefficients = np.array([1.5, -3.25, 0.0, 1.0])
    shifted = shift_coefficients(coefficients, offset)
    points = np.linspace(-2.0, 3.0, 41)
    np.testing.assert_allclose(
        evaluate_polynomial(shifted, points),
        evaluate_polynomial(coefficients, points + offset),
        rtol=2e-14,
        atol=2e-14,
    )


@pytest.mark.parametrize("offset", [-0.4, 0.0, 0.625])
def test_known_simple_roots_translate_rigidly(offset: float) -> None:
    roots = np.array([-2.0, 0.5, 1.5])
    coefficients = np.array([1.5, -3.25, 0.0, 1.0])
    shifted = shift_coefficients(coefficients, offset)
    translated = translate_roots(roots, offset)

    np.testing.assert_allclose(
        evaluate_polynomial(shifted, translated),
        np.zeros_like(translated),
        rtol=0.0,
        atol=2e-14,
    )
    # A common shift preserves both order and every consecutive root gap.
    assert np.all(np.diff(translated) > 0)
    np.testing.assert_allclose(np.diff(translated), np.diff(roots), atol=0.0)


def test_aligned_root_samples_and_offsets_translate_elementwise() -> None:
    roots = np.full(4, 1.5)
    offsets = np.array([-0.25, -0.1, 0.0, 0.25])
    np.testing.assert_allclose(
        translate_roots(roots, offsets),
        np.array([1.75, 1.6, 1.5, 1.25]),
        atol=0.0,
    )


def test_normalized_subleading_curve_matches_closed_form() -> None:
    coefficients = np.array([1.5, -3.25, 0.0, 1.0])
    offsets = np.array([-0.25, 0.0, 0.25])
    curve = normalized_subleading_curve(coefficients, offsets)

    expected = np.column_stack(
        (
            3.0 * offsets,
            -3.25 + 3.0 * offsets**2,
        )
    )
    np.testing.assert_allclose(curve, expected, atol=0.0)


def test_degree_trimming_and_input_validation() -> None:
    assert polynomial_degree([1.0, 2.0, 0.0, 0.0]) == 1
    assert polynomial_degree([0.0, 0.0]) == -1

    with pytest.raises(ValueError, match="quadratic"):
        normalized_subleading_curve([1.0, 2.0], [0.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        shift_coefficients([[1.0, 2.0]], 0.0)
    with pytest.raises(ValueError, match="finite"):
        shift_coefficients([1.0, np.nan], 0.0)
# Maintenance annotation; functional content is unchanged.
