"""Satisficing Combinatorial Thompson Sampling (SAT-CTS) algorithm."""

from typing import List, Optional, Tuple

import numpy as np

from obs.algorithms.combinatorial.algorithm import CombinatorialAlgorithm
from obs.algorithms.combinatorial.objectives import Objective


class SATCTSv2Agent(CombinatorialAlgorithm):
    """SAT-CTS v2 with doubling CTS exploration epochs.

    Gate checks in order:
    1. LCB >= target? Play LCB super-arm
    2. MEAN >= target? Play MEAN super-arm
    3. Otherwise: Start/continue CTS exploration round with fresh local
       posteriors and doubling horizon T_ell = 2^ell. After the round
       finishes, re-check the gate.
    """

    def __init__(
        self,
        num_users: int,
        total_beams: int,
        rate_set: np.ndarray,
        target_throughput: float,
        objective: Optional[Objective] = None,
    ):
        super().__init__(num_users, total_beams, rate_set, objective)
        self.target_throughput = target_throughput
        self.last_decision = None
        self.decision_history = []
        self._init_state()

    def _init_state(self):
        """Initialize internal state."""
        shape = (self.num_users, self.total_beams, self.num_rates)
        self.n_plays = np.zeros(shape, dtype=int)
        self.n_success = np.zeros(shape, dtype=int)
        self.epoch = 0
        self.round_remaining = 0
        self.local_alpha = None
        self.local_beta = None

    def _start_exploration_round(self):
        """Start a new CTS exploration round with fresh local posteriors."""
        self.epoch += 1
        self.round_remaining = 2**self.epoch
        shape = (self.num_users, self.total_beams, self.num_rates)
        self.local_alpha = np.ones(shape)
        self.local_beta = np.ones(shape)

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action using SAT-CTS v2 logic."""
        rates = self.rate_set[None, None, :]
        total_threshold = self.target_throughput * self.num_users

        if self.round_remaining > 0:
            psi_ts = np.random.beta(self.local_alpha, self.local_beta)
            ts_values = rates * psi_ts
            self.last_decision = "CTS"
            self.decision_history.append("CTS")
            return self.objective.select_assignment(
                ts_values, self.cumulative_throughputs
            )

        n = np.maximum(1, self.n_plays)
        psi_hat = self.n_success / n
        conf = np.sqrt(1.5 * np.log(max(2, t)) / n)
        psi_lcb = np.maximum(0.0, psi_hat - conf)

        lcb_values = rates * psi_lcb
        lcb_beams, lcb_rates = self.objective.select_assignment(
            lcb_values, self.cumulative_throughputs
        )
        lcb_sum = sum(
            lcb_values[u, lcb_beams[u], lcb_rates[u]] for u in range(self.num_users)
        )
        if lcb_sum >= total_threshold:
            self.last_decision = "LCB"
            self.decision_history.append("LCB")
            return lcb_beams, lcb_rates

        mu_values = rates * psi_hat
        mu_beams, mu_rates = self.objective.select_assignment(
            mu_values, self.cumulative_throughputs
        )
        mu_sum = sum(
            mu_values[u, mu_beams[u], mu_rates[u]] for u in range(self.num_users)
        )
        if mu_sum >= total_threshold:
            self.last_decision = "MU"
            self.decision_history.append("MU")
            return mu_beams, mu_rates

        self._start_exploration_round()
        psi_ts = np.random.beta(self.local_alpha, self.local_beta)
        ts_values = rates * psi_ts
        self.last_decision = "CTS"
        self.decision_history.append("CTS")
        return self.objective.select_assignment(ts_values, self.cumulative_throughputs)

    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update internal state with observed feedback."""
        for u in range(self.num_users):
            b, r = beams[u], rates[u]
            self.n_plays[u, b, r] += 1
            self.n_success[u, b, r] += ack_nack[u]
            if self.round_remaining > 0:
                self.local_alpha[u, b, r] += ack_nack[u]
                self.local_beta[u, b, r] += 1 - ack_nack[u]
        if self.round_remaining > 0:
            self.round_remaining -= 1
        self._update_cumulative(rates, ack_nack)

    def reset(self):
        """Reset to initial state."""
        self._init_state()
        self._reset_cumulative()
        self.last_decision = None
        self.decision_history = []

    def get_decision_counts(self) -> dict:
        """Get counts of each gate decision."""
        from collections import Counter

        counts = Counter(self.decision_history)
        return {
            "LCB": counts.get("LCB", 0),
            "MU": counts.get("MU", 0),
            "CTS": counts.get("CTS", 0),
        }


class SATCTSv2SharedAgent(CombinatorialAlgorithm):
    """SAT-CTS v2 with doubling epochs, shared global posterior.

    2^i doubling schedule controls when the gate is re-checked.
    During CTS phase, samples directly from the global A/B posterior.
    No local copies — everything is shared.
    """

    def __init__(
        self,
        num_users: int,
        total_beams: int,
        rate_set: np.ndarray,
        target_throughput: float,
        objective: Optional[Objective] = None,
    ):
        super().__init__(num_users, total_beams, rate_set, objective)
        self.target_throughput = target_throughput
        self.last_decision = None
        self.decision_history = []
        self._init_state()

    def _init_state(self):
        """Initialize internal state."""
        shape = (self.num_users, self.total_beams, self.num_rates)
        self.n_plays = np.zeros(shape, dtype=int)
        self.n_success = np.zeros(shape, dtype=int)
        self.A = np.ones(shape)
        self.B = np.ones(shape)
        self.epoch = 0
        self.round_remaining = 0

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action."""
        rates = self.rate_set[None, None, :]
        total_threshold = self.target_throughput * self.num_users

        # Inside epoch: CTS with global posterior, skip gate
        if self.round_remaining > 0:
            psi_ts = np.random.beta(self.A, self.B)
            ts_values = rates * psi_ts
            self.last_decision = "CTS"
            self.decision_history.append("CTS")
            return self.objective.select_assignment(
                ts_values, self.cumulative_throughputs
            )

        # Epoch ended: check gate
        n = np.maximum(1, self.n_plays)
        psi_hat = self.n_success / n
        conf = np.sqrt(1.5 * np.log(max(2, t)) / n)
        psi_lcb = np.maximum(0.0, psi_hat - conf)

        lcb_values = rates * psi_lcb
        lcb_beams, lcb_rates = self.objective.select_assignment(
            lcb_values, self.cumulative_throughputs
        )
        lcb_sum = sum(
            lcb_values[u, lcb_beams[u], lcb_rates[u]] for u in range(self.num_users)
        )
        if lcb_sum >= total_threshold:
            self.last_decision = "LCB"
            self.decision_history.append("LCB")
            return lcb_beams, lcb_rates

        mu_values = rates * psi_hat
        mu_beams, mu_rates = self.objective.select_assignment(
            mu_values, self.cumulative_throughputs
        )
        mu_sum = sum(
            mu_values[u, mu_beams[u], mu_rates[u]] for u in range(self.num_users)
        )
        if mu_sum >= total_threshold:
            self.last_decision = "MU"
            self.decision_history.append("MU")
            return mu_beams, mu_rates

        # Start new epoch
        self.epoch += 1
        self.round_remaining = 2**self.epoch
        psi_ts = np.random.beta(self.A, self.B)
        ts_values = rates * psi_ts
        self.last_decision = "CTS"
        self.decision_history.append("CTS")
        return self.objective.select_assignment(ts_values, self.cumulative_throughputs)

    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update internal state with observed feedback."""
        for u in range(self.num_users):
            b, r = beams[u], rates[u]
            self.n_plays[u, b, r] += 1
            self.n_success[u, b, r] += ack_nack[u]
            self.A[u, b, r] = 1 + self.n_success[u, b, r]
            self.B[u, b, r] = 1 + self.n_plays[u, b, r] - self.n_success[u, b, r]
        if self.round_remaining > 0:
            self.round_remaining -= 1
        self._update_cumulative(rates, ack_nack)

    def reset(self):
        """Reset to initial state."""
        self._init_state()
        self._reset_cumulative()
        self.last_decision = None
        self.decision_history = []

    def get_decision_counts(self) -> dict:
        """Get counts of each gate decision."""
        from collections import Counter

        counts = Counter(self.decision_history)
        return {
            "LCB": counts.get("LCB", 0),
            "MU": counts.get("MU", 0),
            "CTS": counts.get("CTS", 0),
        }


class SATCTSv2MemoryAgent(CombinatorialAlgorithm):
    """SAT-CTS v2 with doubling epochs, local posteriors with memory.

    2^i doubling schedule. When a new epoch starts, local posteriors
    are copied from the current global counts (not fresh Beta(1,1)).
    During the epoch, local posteriors keep accumulating.
    """

    def __init__(
        self,
        num_users: int,
        total_beams: int,
        rate_set: np.ndarray,
        target_throughput: float,
        objective: Optional[Objective] = None,
    ):
        super().__init__(num_users, total_beams, rate_set, objective)
        self.target_throughput = target_throughput
        self.last_decision = None
        self.decision_history = []
        self._init_state()

    def _init_state(self):
        """Initialize internal state."""
        shape = (self.num_users, self.total_beams, self.num_rates)
        self.n_plays = np.zeros(shape, dtype=int)
        self.n_success = np.zeros(shape, dtype=int)
        self.epoch = 0
        self.round_remaining = 0
        self.local_alpha = None
        self.local_beta = None

    def _start_exploration_round(self):
        """Start a new CTS exploration round from current global posteriors."""
        self.epoch += 1
        self.round_remaining = 2**self.epoch
        self.local_alpha = 1 + self.n_success.astype(float).copy()
        self.local_beta = 1 + (self.n_plays - self.n_success).astype(float).copy()

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action."""
        rates = self.rate_set[None, None, :]
        total_threshold = self.target_throughput * self.num_users

        if self.round_remaining > 0:
            psi_ts = np.random.beta(self.local_alpha, self.local_beta)
            ts_values = rates * psi_ts
            self.last_decision = "CTS"
            self.decision_history.append("CTS")
            return self.objective.select_assignment(
                ts_values, self.cumulative_throughputs
            )

        n = np.maximum(1, self.n_plays)
        psi_hat = self.n_success / n
        conf = np.sqrt(1.5 * np.log(max(2, t)) / n)
        psi_lcb = np.maximum(0.0, psi_hat - conf)

        lcb_values = rates * psi_lcb
        lcb_beams, lcb_rates = self.objective.select_assignment(
            lcb_values, self.cumulative_throughputs
        )
        lcb_sum = sum(
            lcb_values[u, lcb_beams[u], lcb_rates[u]] for u in range(self.num_users)
        )
        if lcb_sum >= total_threshold:
            self.last_decision = "LCB"
            self.decision_history.append("LCB")
            return lcb_beams, lcb_rates

        mu_values = rates * psi_hat
        mu_beams, mu_rates = self.objective.select_assignment(
            mu_values, self.cumulative_throughputs
        )
        mu_sum = sum(
            mu_values[u, mu_beams[u], mu_rates[u]] for u in range(self.num_users)
        )
        if mu_sum >= total_threshold:
            self.last_decision = "MU"
            self.decision_history.append("MU")
            return mu_beams, mu_rates

        self._start_exploration_round()
        psi_ts = np.random.beta(self.local_alpha, self.local_beta)
        ts_values = rates * psi_ts
        self.last_decision = "CTS"
        self.decision_history.append("CTS")
        return self.objective.select_assignment(ts_values, self.cumulative_throughputs)

    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update internal state with observed feedback."""
        for u in range(self.num_users):
            b, r = beams[u], rates[u]
            self.n_plays[u, b, r] += 1
            self.n_success[u, b, r] += ack_nack[u]
            if self.round_remaining > 0:
                self.local_alpha[u, b, r] += ack_nack[u]
                self.local_beta[u, b, r] += 1 - ack_nack[u]
        if self.round_remaining > 0:
            self.round_remaining -= 1
        self._update_cumulative(rates, ack_nack)

    def reset(self):
        """Reset to initial state."""
        self._init_state()
        self._reset_cumulative()
        self.last_decision = None
        self.decision_history = []

    def get_decision_counts(self) -> dict:
        """Get counts of each gate decision."""
        from collections import Counter

        counts = Counter(self.decision_history)
        return {
            "LCB": counts.get("LCB", 0),
            "MU": counts.get("MU", 0),
            "CTS": counts.get("CTS", 0),
        }


class SATCTSUCBAgent(CombinatorialAlgorithm):
    """Original SAT-CTS with LCB -> MU -> UCB -> TS gate.

    Matches the SATCTSAgent from main_for_sim.py.
    No doubling epochs — gate checked every round.
    """

    def __init__(
        self,
        num_users: int,
        total_beams: int,
        rate_set: np.ndarray,
        target_throughput: float,
        objective: Optional[Objective] = None,
    ):
        super().__init__(num_users, total_beams, rate_set, objective)
        self.target_throughput = target_throughput
        self.last_decision = None
        self.decision_history = []
        self._init_state()

    def _init_state(self):
        """Initialize internal state."""
        shape = (self.num_users, self.total_beams, self.num_rates)
        self.A = np.ones(shape)
        self.B = np.ones(shape)
        self.n_plays = np.zeros(shape, dtype=int)
        self.n_success = np.zeros(shape, dtype=int)

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action using LCB -> MU -> UCB -> TS gate."""
        rates = self.rate_set[None, None, :]
        total_threshold = self.target_throughput * self.num_users

        n = np.maximum(1, self.n_plays)
        psi_hat = self.n_success / n
        conf = np.sqrt(0.5 * np.log(max(2, t)) / n)
        psi_lcb = np.maximum(0.0, psi_hat - conf)
        psi_ucb = psi_hat + conf

        # LCB gate
        lcb_values = rates * psi_lcb
        lcb_beams, lcb_rates = self.objective.select_assignment(
            lcb_values, self.cumulative_throughputs
        )
        lcb_sum = sum(
            lcb_values[u, lcb_beams[u], lcb_rates[u]] for u in range(self.num_users)
        )
        if lcb_sum >= total_threshold:
            self.last_decision = "LCB"
            self.decision_history.append("LCB")
            return lcb_beams, lcb_rates

        # MU gate
        mu_values = rates * psi_hat
        mu_beams, mu_rates = self.objective.select_assignment(
            mu_values, self.cumulative_throughputs
        )
        mu_sum = sum(
            mu_values[u, mu_beams[u], mu_rates[u]] for u in range(self.num_users)
        )
        if mu_sum >= total_threshold:
            self.last_decision = "MU"
            self.decision_history.append("MU")
            return mu_beams, mu_rates

        # UCB gate
        ucb_values = rates * psi_ucb
        ucb_beams, ucb_rates = self.objective.select_assignment(
            ucb_values, self.cumulative_throughputs
        )
        ucb_sum = sum(
            ucb_values[u, ucb_beams[u], ucb_rates[u]] for u in range(self.num_users)
        )
        if ucb_sum >= total_threshold:
            self.last_decision = "UCB"
            self.decision_history.append("UCB")
            return ucb_beams, ucb_rates

        # TS fallback
        psi_ts = np.random.beta(self.A, self.B)
        ts_values = rates * psi_ts
        self.last_decision = "TS"
        self.decision_history.append("TS")
        return self.objective.select_assignment(ts_values, self.cumulative_throughputs)

    def update(self, beams: List[int], rates: List[int], ack_nack: np.ndarray):
        """Update internal state with observed feedback."""
        for u in range(self.num_users):
            b, r = beams[u], rates[u]
            self.n_plays[u, b, r] += 1
            self.n_success[u, b, r] += ack_nack[u]
            self.A[u, b, r] = 1 + self.n_success[u, b, r]
            self.B[u, b, r] = 1 + self.n_plays[u, b, r] - self.n_success[u, b, r]
        self._update_cumulative(rates, ack_nack)

    def reset(self):
        """Reset to initial state."""
        self._init_state()
        self._reset_cumulative()
        self.last_decision = None
        self.decision_history = []

    def get_decision_counts(self) -> dict:
        """Get counts of each gate decision."""
        from collections import Counter

        counts = Counter(self.decision_history)
        return {
            "LCB": counts.get("LCB", 0),
            "MU": counts.get("MU", 0),
            "UCB": counts.get("UCB", 0),
            "TS": counts.get("TS", 0),
        }
