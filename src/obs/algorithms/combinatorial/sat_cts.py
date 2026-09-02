"""Satisficing Combinatorial Thompson Sampling (SAT-CTS) algorithm."""

from typing import List, Optional, Tuple

import numpy as np

from obs.algorithms.combinatorial.algorithm import CombinatorialAlgorithm
from obs.algorithms.combinatorial.objectives import Objective


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
        reset_priors: bool = False,
        init_phase: bool = False,
        init_group_size: int = 1,
    ):
        """Initialize SAT-CTS v2.

        Parameters
        ----------
        reset_priors : bool, optional
            If True, each committed CTS round restarts from fresh ``Beta(1, 1)``
            priors, as assumed by the finite-time analysis (Algorithm 1). The
            LCB/MEAN gate keeps using the globally accumulated counters --
            only the Thompson posterior resets. If False (default) the global
            posterior is retained across rounds, which is the historical
            behaviour of this class; keeping it the default means existing runs
            reproduce byte-identically.
        init_phase : bool, optional
            If True, run the deterministic covering initialization of
            Algorithm 1 (lines 10--14) before the gate is ever evaluated: every
            base arm is played exactly once, which costs
            ``T_0 = total_beams * num_rates`` rounds. This is what the regret
            analysis assumes, since Lemma 5 needs ``n_i >= 1`` for every arm.
            If False (default) no covering phase is run and unplayed arms take
            the convention ``n_i <- max(1, n_i)``, ``psi_hat_i = 0``, which
            makes both indices zero and hence conservative.
        """
        super().__init__(num_users, total_beams, rate_set, objective)
        self.target_throughput = target_throughput
        self.reset_priors = bool(reset_priors)
        self.init_phase = bool(init_phase)
        # Coarse initialization: visit only every `init_group_size`-th beam
        # (the group centre) instead of all of them, cutting T_0 by that
        # factor. NOTE this does NOT give n_i >= 1 for every base arm, so it
        # does not satisfy the premise of the concentration lemma -- it is a
        # practical warm start, not the covering phase the analysis assumes.
        self.init_group_size = max(1, int(init_group_size))
        self._init_beams = np.arange(self.init_group_size // 2,
                                     self.total_beams, self.init_group_size)
        if self.init_phase and len(self._init_beams) < self.num_users:
            # Fewer covering beams than users means no round can give every
            # user a distinct beam; the schedule would silently hand duplicates
            # to the assignment oracle.
            raise ValueError(
                f"init_group_size={self.init_group_size} leaves only "
                f"{len(self._init_beams)} covering beams for "
                f"{self.num_users} users; need at least one beam per user")
        self.T0 = (len(self._init_beams) * self.num_rates
                   if self.init_phase else 0)
        self.last_decision = None
        self.decision_history = []
        self._init_state()

    def _init_assignment(self, j: int) -> Tuple[List[int], List[int]]:
        """Round ``j`` of the covering schedule, ``0 <= j < T_0``.

        Rate index cycles fastest; the beam offset advances every ``num_rates``
        rounds and user ``u`` takes the beam ``u * stride`` positions along from
        the offset. Within a round the beams are distinct, and over the whole
        schedule user ``u`` visits every beam at every rate, which is what
        Lemma 5 needs.

        The stride is what keeps the schedule FEASIBLE. Global beam indices are
        laid out per BS (``bs * K + k``), so consecutive indices belong to the
        same base station: a schedule handing user ``u`` beam ``offset + u``
        parks every user on one BS and violates the per-BS RF-chain cap
        ``|{m : b_m = b}| <= N_RF,b`` for essentially the whole of ``T_0``.
        Spacing the users ``nb // num_users`` apart spreads them evenly over the
        beam index range and therefore evenly over the base stations -- with
        ``M = 15``, ``B = 3`` that is 5 users per BS against ``N_RF = 8``. Arm
        coverage and ``T_0`` are unchanged; only the within-round layout moves.
        """
        r_idx = j % self.num_rates
        offset = j // self.num_rates
        nb = len(self._init_beams)
        stride = max(1, nb // self.num_users)
        beams = [int(self._init_beams[(offset + u * stride) % nb])
                 for u in range(self.num_users)]
        return beams, [r_idx] * self.num_users

    def _init_state(self):
        """Initialize internal state."""
        shape = (self.num_users, self.total_beams, self.num_rates)
        # Shared counters: accumulate over the whole horizon and feed the gate.
        self.n_plays = np.zeros(shape, dtype=int)
        self.n_success = np.zeros(shape, dtype=int)
        # Epoch-local counters: feed the Beta posterior when reset_priors=True.
        self.ep_plays = np.zeros(shape, dtype=int)
        self.ep_success = np.zeros(shape, dtype=int)
        self.A = np.ones(shape)
        self.B = np.ones(shape)
        self.epoch = 0
        self.round_remaining = 0
        self.init_done = 0

    def select_action(self, t: int) -> Tuple[List[int], List[int]]:
        """Select action."""
        rates = self.rate_set[None, None, :]
        total_threshold = self.target_throughput * self.num_users

        # Covering initialization: play the fixed schedule, never touch the gate.
        if self.init_done < self.T0:
            beams, rate_idx = self._init_assignment(self.init_done)
            self.last_decision = "INIT"
            self.decision_history.append("INIT")
            return beams, rate_idx

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
        if self.reset_priors:
            # Fresh Beta(1,1) priors for this committed CTS round (Algorithm 1).
            # The shared n_plays/n_success are deliberately NOT cleared: the gate
            # must keep its accumulated evidence.
            self.ep_plays.fill(0)
            self.ep_success.fill(0)
            self.A.fill(1.0)
            self.B.fill(1.0)
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
            self.ep_plays[u, b, r] += 1
            self.ep_success[u, b, r] += ack_nack[u]
            if self.reset_priors:
                # Posterior sees only evidence from the current committed round.
                s, nn = self.ep_success[u, b, r], self.ep_plays[u, b, r]
            else:
                # Retain: posterior derived from all accumulated evidence.
                s, nn = self.n_success[u, b, r], self.n_plays[u, b, r]
            self.A[u, b, r] = 1 + s
            self.B[u, b, r] = 1 + nn - s
        if self.init_done < self.T0:
            self.init_done += 1
        elif self.round_remaining > 0:
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
            "INIT": counts.get("INIT", 0),
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
