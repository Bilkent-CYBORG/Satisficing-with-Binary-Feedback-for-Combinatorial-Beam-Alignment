"""Optimization objectives for combinatorial bandit algorithms."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from obs.utils.utils import capacitated_best, hungarian_best


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


class CapacitatedThroughputObjective(Objective):
    """Maximize total throughput subject to the per-BS RF-chain cap.

    Enforces the constraint that appears in the paper's super-arm set ``S``::

        |{m : b_m = b}| <= N_RF,b   for every BS b

    Lazily. The unconstrained Hungarian solution is computed first; if it
    already respects the cap then it IS the constrained optimum, and the LP is
    skipped. Only when the cap binds does this fall back to
    :func:`capacitated_best`, which solves the transportation LP exactly.

    This matters because the LP is ~163x slower than the Hungarian (13.4 ms vs
    82 us at M=15, B*K=360). Measured on played actions, the cap binds in under
    1% of rounds for SAT-CTS and about 57% for CUCB, so lazy evaluation turns a
    prohibitive cost into a marginal one while giving exactly the same answers.

    ``n_calls`` / ``n_capped`` count how often the fallback fired, so the
    violation rate can be reported rather than assumed.
    """

    def __init__(self, beam_to_bs: np.ndarray, cap: np.ndarray):
        self.beam_to_bs = np.asarray(beam_to_bs)
        self.cap = np.asarray(cap)
        self.num_bs = len(self.cap)
        self.n_calls = 0
        self.n_capped = 0

    def select_assignment(
        self, values: np.ndarray, cumulative: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Hungarian first; the LP only if the RF-chain cap is exceeded."""
        beams, rates = hungarian_best(values)
        self.n_calls += 1
        load = np.bincount(self.beam_to_bs[beams], minlength=self.num_bs)
        if (load <= self.cap).all():
            return beams, rates
        self.n_capped += 1
        return capacitated_best(values, self.beam_to_bs, self.cap)


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
