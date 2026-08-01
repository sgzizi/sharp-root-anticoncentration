from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from sharp_root_ac.harmonic import (  # noqa: E402
    cost_sensitive_predictions,
    cost_sensitive_utility,
)
from sharp_root_ac.hedge import (  # noqa: E402
    FullInformationHedge,
    dense_proxy_regret,
    exact_grid_regret,
    parameter_grid,
)


def test_act_observe_order_is_enforced() -> None:
    hedge = FullInformationHedge(np.array([0.0, 0.5, 1.0]), horizon=2, seed=7)
    with pytest.raises(RuntimeError, match="act must be called"):
        hedge.observe(np.array([0.2, 0.4, 0.7]))

    index, action = hedge.act()
    assert action == hedge.actions[index]
    with pytest.raises(RuntimeError, match="observe must be called"):
        hedge.act()
    reward = hedge.observe(np.array([0.2, 0.4, 0.7]))
    assert reward == np.array([0.2, 0.4, 0.7])[index]

    hedge.act()
    hedge.observe(np.array([0.1, 0.3, 0.9]))
    with pytest.raises(RuntimeError, match="horizon"):
        hedge.act()


def test_act_cannot_receive_the_current_utility_vector() -> None:
    # This signature check guards the central uninformed-online protocol: the
    # current full-information vector is accepted only by observe().
    assert list(inspect.signature(FullInformationHedge.act).parameters) == ["self"]
    assert "utilities" in inspect.signature(FullInformationHedge.observe).parameters


def test_cost_sensitive_threshold_and_normalized_utility() -> None:
    scores = np.array([0.2, 0.4, 0.8])
    predictions = cost_sensitive_predictions(
        scores, false_negative_cost=3.0, false_positive_cost=1.0
    )
    # The threshold is 1/(3+1)=0.25, not 1/2.
    assert np.array_equal(predictions, np.array([0, 1, 1]))
    utility = cost_sensitive_utility(
        predictions,
        true_labels=np.array([1, 0, 1]),
        false_negative_cost=3.0,
        false_positive_cost=1.0,
        cost_scale=3.0,
    )
    assert utility == pytest.approx(5.0 / 9.0)


def test_each_horizon_has_its_own_inverse_square_root_mesh() -> None:
    grid_100 = parameter_grid(3.0, horizon=100)
    grid_400 = parameter_grid(3.0, horizon=400)
    assert np.max(np.diff(grid_100)) <= 100 ** -0.5 + 1e-12
    assert np.max(np.diff(grid_400)) <= 400 ** -0.5 + 1e-12
    assert grid_400.size > grid_100.size
    assert grid_100[0] == grid_400[0] == 0.0
    assert grid_100[-1] == grid_400[-1] == 3.0


def test_exact_grid_and_dense_proxy_regret_are_distinct_quantities() -> None:
    grid_utilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    selected = np.array([0, 0])
    learner_rewards = grid_utilities[np.arange(2), selected]
    assert exact_grid_regret(grid_utilities, selected) == pytest.approx(0.0)

    proxy_utilities = np.array([[1.0, 0.5, 0.0], [0.0, 0.8, 1.0]])
    assert dense_proxy_regret(proxy_utilities, learner_rewards) == pytest.approx(0.3)
