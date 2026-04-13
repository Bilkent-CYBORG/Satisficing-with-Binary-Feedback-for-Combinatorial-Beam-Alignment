"""Base class for combinatorial bandit algorithms."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from obs.algorithms.combinatorial.objectives import Objective, ThroughputObjective


class CombinatorialAlgorithm(ABC):
    """Base class for combinatorial bandit algorithms.

    Combinatorial algorithms handle multi-user beam assignment with binary
    (ACK/NACK) feedback. Each user is assigned a unique beam and rate.
    """

    def __init__(
        self,
        num_users: int,
        total_beams: int,
        rate_set: np.ndarray,
        objective: Optional[Objective] = None,
    ):
        """Initialize combinatorial algorithm.

        Parameters
        ----------
        num_users : int
            Number of users to assign beams to.
        total_beams : int
            Total number of available beams.
        rate_set : np.ndarray
            Array of available rates (bits/symbol).
        objective : Objective, optional
            Optimization objective. Default is ThroughputObjective.
        """
        self.num_users = num_users
        self.total_beams = total_beams
        self.rate_set = np.array(rate_set)
        self.num_rates = len(rate_set)
        self.objective = objective if objective is not None else ThroughputObjective()
        self.cumulative_throughputs = np.zeros(num_users)

    @abstractmethod
    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select beam and rate assignments for all users.

        Parameters
        ----------
        t : int
            Current time step.

        Returns
        -------
        tuple
            Tuple of (chosen_beams, chosen_rates).
        """
        pass

    @abstractmethod
    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update with feedback."""
        pass

    @abstractmethod
    def reset(self):
        """Reset algorithm to initial state."""
        pass

    def _reset_cumulative(self):
        """Reset cumulative throughputs to zero."""
        self.cumulative_throughputs = np.zeros(self.num_users)

    def _update_cumulative(self, rates: List[int], ack_nack: np.ndarray):
        """Update cumulative throughputs with observed rewards.

        Parameters
        ----------
        rates : list of int
            Rate indices chosen for each user.
        ack_nack : np.ndarray
            Binary feedback (1=ACK, 0=NACK) for each user.
        """
        for u in range(self.num_users):
            self.cumulative_throughputs[u] += self.rate_set[rates[u]] * ack_nack[u]
