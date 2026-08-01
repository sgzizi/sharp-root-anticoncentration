"""A small full-information Hedge implementation with an explicit protocol.

``act`` takes no utility argument and must precede ``observe``.  This makes it
hard for an experiment to accidentally use the current round's full utility
vector before committing to an action.
"""

from __future__ import annotations

import math

import numpy as np


Array = np.ndarray


def parameter_grid(width: float, horizon: int) -> Array:
    """Return the paper's horizon-specific grid with mesh ``T^{-1/2}``."""

    b = float(width)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("width must be positive and finite")
    if int(horizon) != horizon or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    h = float(horizon) ** -0.5
    last_multiple = int(math.floor(b / h + 1e-12))
    grid = np.arange(last_multiple + 1, dtype=float) * h
    if grid[-1] < b - 1e-12:
        grid = np.concatenate((grid, np.array([b])))
    else:
        grid[-1] = b
    return grid


class FullInformationHedge:
    """Hedge for rewards in ``[0,1]`` on a fixed finite action grid."""

    def __init__(
        self,
        actions: Array,
        horizon: int,
        *,
        seed: int | np.random.SeedSequence | None = None,
        learning_rate: float | None = None,
    ) -> None:
        action_array = np.asarray(actions, dtype=float)
        if action_array.ndim != 1 or action_array.size == 0:
            raise ValueError("actions must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("actions must be finite")
        if int(horizon) != horizon or horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        self.actions = action_array.copy()
        self.horizon = int(horizon)
        if learning_rate is None:
            if self.actions.size == 1:
                learning_rate = 1.0
            else:
                # The Hoeffding-lemma analysis gives a grid-expert reward
                # regret at most log(M)/eta + eta*T/8.
                learning_rate = math.sqrt(
                    8.0 * math.log(self.actions.size) / self.horizon
                )
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")

        self.learning_rate = float(learning_rate)
        self._rng = np.random.default_rng(seed)
        self._log_weights = np.zeros(self.actions.size, dtype=float)
        self._pending_index: int | None = None
        self._round = 0
        self.action_indices: list[int] = []
        self.received_rewards: list[float] = []

    @property
    def round(self) -> int:
        return self._round

    @property
    def probabilities(self) -> Array:
        shifted = self._log_weights - np.max(self._log_weights)
        weights = np.exp(shifted)
        return weights / np.sum(weights)

    def act(self) -> tuple[int, float]:
        """Commit to the next action using past observations only."""

        if self._pending_index is not None:
            raise RuntimeError("observe must be called before the next act")
        if self._round >= self.horizon:
            raise RuntimeError("the configured horizon has been exhausted")
        index = int(self._rng.choice(self.actions.size, p=self.probabilities))
        self._pending_index = index
        return index, float(self.actions[index])

    def observe(self, utilities: Array) -> float:
        """Reveal the full utility vector and update after the committed action."""

        if self._pending_index is None:
            raise RuntimeError("act must be called before observe")
        reward_vector = np.asarray(utilities, dtype=float)
        if reward_vector.shape != self.actions.shape:
            raise ValueError("utilities must contain one reward per action")
        if not np.all(np.isfinite(reward_vector)) or np.any(
            (reward_vector < 0.0) | (reward_vector > 1.0)
        ):
            raise ValueError("utilities must lie in [0,1]")

        index = self._pending_index
        received = float(reward_vector[index])
        self._log_weights += self.learning_rate * reward_vector
        # Centering is algebraically immaterial and prevents overflow in long
        # runs without changing the action probabilities.
        self._log_weights -= np.max(self._log_weights)
        self.action_indices.append(index)
        self.received_rewards.append(received)
        self._pending_index = None
        self._round += 1
        return received


def exact_grid_regret(utilities: Array, action_indices: Array) -> float:
    """Regret to the best expert in exactly the supplied algorithm grid."""

    reward_matrix = np.asarray(utilities, dtype=float)
    indices = np.asarray(action_indices, dtype=int)
    if reward_matrix.ndim != 2 or indices.shape != (reward_matrix.shape[0],):
        raise ValueError("utilities and action_indices have incompatible shapes")
    if np.any(indices < 0) or np.any(indices >= reward_matrix.shape[1]):
        raise IndexError("an action index is out of range")
    learner_reward = float(np.sum(reward_matrix[np.arange(indices.size), indices]))
    best_grid_reward = float(np.max(np.sum(reward_matrix, axis=0)))
    return best_grid_reward - learner_reward


def dense_proxy_regret(dense_utilities: Array, learner_rewards: Array) -> float:
    """Regret to a declared dense-grid proxy for the continuous comparator.

    This function intentionally does not call the result continuous regret.
    A finite evaluation grid may underestimate the true continuous comparator.
    """

    reward_matrix = np.asarray(dense_utilities, dtype=float)
    received = np.asarray(learner_rewards, dtype=float)
    if reward_matrix.ndim != 2 or received.shape != (reward_matrix.shape[0],):
        raise ValueError("dense_utilities and learner_rewards have incompatible shapes")
    return float(np.max(np.sum(reward_matrix, axis=0)) - np.sum(received))
