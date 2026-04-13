"""Optimization objectives for combinatorial bandit algorithms."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from obs.utils.utils import hungarian_best


class Objective(ABC):
    """Base class for optimization objectives in combinatorial bandits."""

    @abstractmethod
    def select_assignment(
        self, values: np.ndarray, cumulative: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Select optimal beam-rate assignment.

        Parameters
        ----------
        values : np.ndarray
            Array of shape (num_users, total_beams, num_rates) with expected throughputs.
        cumulative : np.ndarray
            Array of shape (num_users,) with cumulative throughput per user.

        Returns
        -------
        tuple
            Tuple of (chosen_beams, chosen_rates).
        """
        pass


class ThroughputObjective(Objective):
    """Maximize total throughput (default behavior)."""

    def select_assignment(
        self, values: np.ndarray, cumulative: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Select assignment maximizing total throughput."""
        return hungarian_best(values)


class ProportionalFairnessObjective(Objective):
    """Maximize proportional fairness (Nash bargaining solution).

    Optimizes sum(log(cumulative + throughput)) which:
    - Is the Nash bargaining solution
    - Proven to improve Jain's Fairness Index
    - Uses standard Hungarian on log-transformed values
    """

    def select_assignment(
        self, values: np.ndarray, cumulative: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Select assignment maximizing proportional fairness."""
        eps = 1e-10
        # Transform to log-utility: log(cumulative + expected_throughput)
        # cumulative: (num_users,) -> (num_users, 1, 1) for broadcasting
        log_values = np.log(cumulative[:, None, None] + values + eps)
        return hungarian_best(log_values)


def compute_jain_index(cumulative_throughputs: np.ndarray) -> float:
    """Compute Jain's Fairness Index.

    J(T) = (sum(G_m))^2 / (M * sum(G_m^2))

    Parameters
    ----------
    cumulative_throughputs : np.ndarray
        Array of cumulative throughputs per user.

    Returns
    -------
    float
        Jain's index in range [1/M, 1]. 1 means perfect fairness.
    """
    n = len(cumulative_throughputs)
    total = cumulative_throughputs.sum()
    sum_squares = (cumulative_throughputs**2).sum()
    return (total**2) / (n * sum_squares + 1e-10)
