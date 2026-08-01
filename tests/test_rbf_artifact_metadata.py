from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiments.rbf_graph_ssl import (  # noqa: E402
    CI_CONDITIONING,
    CI_METHOD,
    CI_RANDOMNESS,
    DENSE_PROXY_LABEL,
    GRID_REGRET_LABEL,
    select_illustrative_transition,
    summarize_runs,
)


def test_transition_selection_maximizes_changes_then_uses_central_crossing() -> None:
    gammas = np.arange(6.0)
    margins = np.ones((2, gammas.size, 2))
    margins[0, :, 0] = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    margins[1, :, 1] = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, -1.0])

    selection = select_illustrative_transition(margins, gammas)

    assert selection.maximum_sampled_sign_changes == 2
    assert selection.selected_sampled_sign_changes == 2
    assert (selection.round_index, selection.local_vertex) == (1, 1)
    assert "parameter-interval midpoint" in selection.rule


def test_transition_selection_records_no_change_fallback() -> None:
    gammas = np.array([0.0, 0.5, 1.0])
    margins = np.ones((2, gammas.size, 2))
    margins[1, 1, 0] = 0.01

    selection = select_illustrative_transition(margins, gammas)

    assert selection.maximum_sampled_sign_changes == 0
    assert selection.selected_sampled_sign_changes == 0
    assert (selection.round_index, selection.local_vertex) == (1, 0)
    assert "no sampled sign change" in selection.rule


def test_regret_labels_name_both_finite_comparator_sets() -> None:
    assert GRID_REGRET_LABEL == "Regret to algorithm grid (exact)"
    assert DENSE_PROXY_LABEL == "Dense-plus-algorithm-grid finite proxy"


def test_regret_summary_records_ci_interpretation() -> None:
    rows = [
        {
            "horizon": 10,
            "grid_regret": value,
            "dense_proxy_regret": value + 0.5,
            "proxy_grid_gap": 0.5,
        }
        for value in (1.0, 2.0, 3.0)
    ]

    summary = summarize_runs(rows)

    assert len(summary) == 1
    assert summary[0]["ci_level"] == 0.95
    assert summary[0]["ci_method"] == CI_METHOD
    assert summary[0]["ci_conditioning"] == CI_CONDITIONING
    assert summary[0]["ci_randomness"] == CI_RANDOMNESS
