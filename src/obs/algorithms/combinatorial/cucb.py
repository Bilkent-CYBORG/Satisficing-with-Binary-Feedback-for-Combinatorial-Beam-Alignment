"""Combinatorial UCB (CUCB) algorithm."""

from typing import List, Optional, Tuple

import numpy as np

from obs.algorithms.combinatorial.algorithm import CombinatorialAlgorithm
from obs.algorithms.combinatorial.objectives import Objective


class CUCBAgent(CombinatorialAlgorithm):
    """Combinatorial UCB algorithm with Hoeffding confidence bounds.

    Maintains internal n_plays and psi_hat (empirical success probability).
    The index is clipped at 1, which is the standard form: psi is a
    probability, so an optimistic estimate above 1 is not attainable.
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
        """Initialize internal state."""
        self.n_plays = np.zeros(
            (self.num_users, self.total_beams, self.num_rates), dtype=int
        )
        self.psi_hat = np.zeros((self.num_users, self.total_beams, self.num_rates))

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action using UCB on success probabilities.

        Matches Algorithm 2: the confidence radius is sqrt(3 log t / 2n), the
        index is capped at 1, and an arm that has never been played carries
        UCB = +infinity so it is always preferred over any finite index.
        """
        seen = self.n_plays > 0
        n = np.maximum(1, self.n_plays)
        conf = np.sqrt(1.5 * np.log(max(2, t)) / n)  # = sqrt(3 log t / 2n)
        # psi is a probability, so an index above 1 is not attainable. The
        # standard CUCB index is min(psi_hat + rad, 1); leaving it unclipped
        # inflates rarely-played arms without bound.
        ucb = np.minimum(self.psi_hat + conf, 1.0)
        theta_ucb = self.rate_set[None, None, :] * ucb
        # +inf cannot be handed to the assignment solver. Any sentinel strictly
        # above the largest achievable finite index is equivalent: psi <= 1, so
        # every finite theta is at most max(rate_set). Unplayed arms all share
        # one sentinel, leaving the tie-break to the solver as +inf would.
        theta_ucb = np.where(seen, theta_ucb, 10.0 * float(self.rate_set.max()))
        return self.objective.select_assignment(theta_ucb, self.cumulative_throughputs)

    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update internal state with observed feedback."""
        for u in range(self.num_users):
            b, r = beams[u], rates[u]
            self.n_plays[u, b, r] += 1
            # Incremental mean update
            self.psi_hat[u, b, r] += (
                ack_nack[u] - self.psi_hat[u, b, r]
            ) / self.n_plays[u, b, r]
        self._update_cumulative(rates, ack_nack)

    def reset(self):
        """Reset to initial state."""
        self._init_state()
        self._reset_cumulative()
