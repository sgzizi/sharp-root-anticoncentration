import pytest

from sharp_root_ac.incidence import (
    rbf_cube_incidence_speed,
    rbf_cube_incidence_speed_numeric,
    rbf_cube_section_integral_numeric,
)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("p_prime", [-3.0, 0.0, 0.7])
@pytest.mark.parametrize("radius", [0.25, 1.0, 5.0])
def test_numeric_cube_section_matches_equation_56(p, p_prime, radius):
    closed_form = rbf_cube_incidence_speed(p, p_prime, radius)
    from_section = rbf_cube_incidence_speed_numeric(
        p, p_prime, radius, panels=256
    )
    assert from_section == pytest.approx(closed_form, rel=2e-13, abs=2e-13)


def test_equation_56_special_scores():
    assert rbf_cube_incidence_speed(0.5, -2.0) == pytest.approx(2.0)
    assert rbf_cube_incidence_speed(0.0, -2.0) == pytest.approx(0.5)
    assert rbf_cube_incidence_speed(1.0, 2.0) == pytest.approx(0.5)


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_normalized_speed_is_independent_of_cube_radius(p):
    speeds = [rbf_cube_incidence_speed_numeric(p, 1.7, radius) for radius in (0.1, 1.0, 9.0)]
    assert speeds[0] == pytest.approx(speeds[1], rel=2e-13, abs=2e-13)
    assert speeds[1] == pytest.approx(speeds[2], rel=2e-13, abs=2e-13)


def test_raw_section_integral_scales_quadratically_with_radius():
    small = rbf_cube_section_integral_numeric(0.3, -1.2, radius=0.5)
    large = rbf_cube_section_integral_numeric(0.3, -1.2, radius=2.0)
    assert large / small == pytest.approx((2.0 / 0.5) ** 2, rel=2e-13)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rbf_cube_incidence_speed(-0.01, 1.0),
        lambda: rbf_cube_incidence_speed(1.01, 1.0),
        lambda: rbf_cube_incidence_speed(0.5, 1.0, radius=0.0),
        lambda: rbf_cube_incidence_speed_numeric(0.5, 1.0, panels=3),
    ],
)
def test_invalid_rbf_inputs_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()
