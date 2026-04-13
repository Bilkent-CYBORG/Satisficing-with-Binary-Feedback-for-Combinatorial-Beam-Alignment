"""Combinatorial Thompson Sampling (CTS) algorithm."""

from typing import List, Optional, Tuple

import numpy as np

from obs.algorithms.combinatorial.algorithm import CombinatorialAlgorithm
from obs.algorithms.combinatorial.objectives import Objective


class CTSAgent(CombinatorialAlgorithm):
    """Combinatorial Thompson Sampling with Beta posteriors.

    Maintains internal Beta(A, B) parameters for each (user, beam, rate) triplet.
    """

    def __init__(
        self,
        num_users: int,
        total_beams: int,
        rate_set: np.ndarray,
        objective: Optional[Objective] = None,
    ):
        super().__init__(num_users, total_beams, rate_set, objective)
        self._init_state()

    def _init_state(self):
        """Initialize Beta parameters with uniform prior Beta(1, 1)."""
        self.A = np.ones((self.num_users, self.total_beams, self.num_rates))
        self.B = np.ones((self.num_users, self.total_beams, self.num_rates))

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action using Thompson Sampling."""
        psi_samples = np.random.beta(self.A, self.B)
        theta_samples = self.rate_set[None, None, :] * psi_samples
        return self.objective.select_assignment(
            theta_samples, self.cumulative_throughputs
        )

    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update Beta posteriors with observed feedback."""
        for u in range(self.num_users):
            b, r = beams[u], rates[u]
            self.A[u, b, r] += ack_nack[u]
            self.B[u, b, r] += 1 - ack_nack[u]
        self._update_cumulative(rates, ack_nack)

    def reset(self):
        """Reset to initial state."""
        self._init_state()
        self._reset_cumulative()
