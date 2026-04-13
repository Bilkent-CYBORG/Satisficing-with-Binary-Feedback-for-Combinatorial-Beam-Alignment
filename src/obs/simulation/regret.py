"""Regret classes for different regret types in bandit simulations."""

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
from sklearn.metrics import euclidean_distances

from obs.environment.environment import Environment
from obs.simulation.reward import ContinuousReward

from obs.user.user import User


class Regret(ABC):
    """Base class for regret computation."""

    def __init__(self, reward_log: ContinuousReward, reward_linear: ContinuousReward):
        """Initialize with two reward functions for log and linear scale.

        Parameters
        ----------
        reward_log : ContinuousReward
            ContinuousReward with use_log=True.
        reward_linear : ContinuousReward
            ContinuousReward with use_log=False.
        """
        self.reward_log = reward_log
        self.reward_linear = reward_linear

    @abstractmethod
    def compute(self, user: User, action_idx: int, **kwargs) -> Tuple[float, float]:
        """Compute regret.

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Selected action index.
        **kwargs
            Additional params (x_hat_t, etc.).

        Returns
        -------
        tuple of (float, float)
            Tuple of (log_regret, linear_regret).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this regret type."""
        pass


class StandartRegret(Regret):
    """Standart regret: optimal - selected at true position.

    This is the standard regret measure that computes the difference between
    the optimal reward and the selected action's reward at the user's true position.
    """

    def compute(self, user: User, action_idx: int, **kwargs) -> Tuple[float, float]:
        """Compute standart regret.

        Parameters
        ----------
        user : User
            User instance at true position.
        action_idx : int
            Selected action index.

        Returns
        -------
        tuple of (float, float)
            Tuple of (log_regret, linear_regret).
        """
        user_rss = user.get_rss()
        optimal_idx = int(np.argmax(user_rss))

        # Log regret using reward_log
        optimal_log = self.reward_log.compute(user, optimal_idx)
        selected_log = self.reward_log.compute(user, action_idx)
        log_regret = optimal_log - selected_log

        # Linear regret using reward_linear
        optimal_linear = self.reward_linear.compute(user, optimal_idx)
        selected_linear = self.reward_linear.compute(user, action_idx)
        linear_regret = optimal_linear - selected_linear

        return log_regret, linear_regret

    @property
    def name(self) -> str:
        return "standart"


class RobustSatisficing(Regret):
    """Robust Satisficing regret.

    This regret measure accounts for position uncertainty by computing the expected
    regret over a uniform distribution of positions within an eps-ball around
    the estimated position (x_hat_t).
    """

    def __init__(
        self,
        reward_log: ContinuousReward,
        reward_linear: ContinuousReward,
        environment: Environment,
        eps: float = 20.0,
        tau_log: float = -25.0,
        tau_linear: float = None,
    ):
        """Initialize robust satisficing regret calculator.

        Parameters
        ----------
        reward_log : ContinuousReward
            ContinuousReward with use_log=True.
        reward_linear : ContinuousReward
            ContinuousReward with use_log=False.
        environment : Environment
            Environment instance.
        eps : float, optional
            Radius of the uncertainty ball. Default is 20.0.
        tau_log : float, optional
            Satisficing threshold in log scale (e.g., -25 dB). Default is -25.0.
        tau_linear : float, optional
            Satisficing threshold in linear scale. Default is 10^tau_log.
        """
        super().__init__(reward_log, reward_linear)
        self.environment = environment
        self.eps = eps
        self.tau_log = tau_log
        self.tau_linear = tau_linear if tau_linear is not None else 10**tau_log

    def compute(self, user: User, action_idx: int, **kwargs) -> Tuple[float, float]:
        """Compute robust satisficing regret.

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Selected action index.
        **kwargs
            Must contain 'x_hat_t' (estimated position).

        Returns
        -------
        tuple of (float, float)
            Tuple of (log_regret, linear_regret) - expected over eps-ball.
        """
        x_hat_t = kwargs["x_hat_t"]
        y = self.environment.get_rss().T
        _ = self.environment.get_rss()

        kappa_log, eps_log = self.find_kappa_and_eps(
            C=self.environment.positions(),
            Y=np.log10(y),
            ind_ref_cho=self.environment.get_pos_index(x_hat_t),
            tau=self.tau_log,
            action_idx=action_idx,
        )
        # Log regret
        selected_log = self.reward_log.compute(user, action_idx)
        # if kappa_log = inf and eps_log = -inf just make them 0
        # what to do in this case ask hoca
        if kappa_log == float("inf") and eps_log == float("-inf"):
            kappa_log = 0
            eps_log = 0

        log_regret = max(0.0, self.tau_log - kappa_log * eps_log - selected_log)

        kappa_linear, eps_linear = self.find_kappa_and_eps(
            C=self.environment.positions(),
            Y=y,
            ind_ref_cho=self.environment.get_pos_index(x_hat_t),
            tau=self.tau_linear,
            action_idx=action_idx,
        )

        # Linear regret
        if kappa_linear == float("inf") and eps_linear == float("-inf"):
            kappa_linear = 0
            eps_linear = 0

        selected_linear = self.reward_linear.compute(user, action_idx)
        linear_regret = max(
            0.0, self.tau_linear - kappa_linear * eps_linear - selected_linear
        )

        return log_regret, linear_regret

    def find_kappa_and_eps(
        self,
        C: np.ndarray,
        Y: np.ndarray,
        ind_ref_cho: int,
        tau: float,
        action_idx: int,
        dist_metric: Callable = euclidean_distances,
        pow: float = 1,
    ) -> Tuple[float, float]:
        """Find the action with the highest RS value.

        Parameters
        ----------
        C : np.ndarray
            Shape (#contexts, context_dim).
        Y : np.ndarray
            Shape (#actions, #contexts).
        ind_ref_cho : int
            Index of reference context.
        tau : float
            Threshold value.
        action_idx : int
            Index of selected action.
        dist_metric : callable, optional
            Distance metric. Default is euclidean_distances.
        pow : float, optional
            Power value. Default is 1.

        Returns
        -------
        tuple of (float, float)
            Tuple of (kappa, epsilon) for regret computation.
        """
        X_less_than_tau_index = np.where(Y[:, ind_ref_cho] < tau)[0]
        X_more_than_tau_index = np.where(Y[:, ind_ref_cho] >= tau)[0]
        N_x = Y.shape[0]

        # init kappas and epsilons
        kappa_values = np.inf * np.ones(N_x)
        epsilon_values = np.inf * np.ones(N_x)

        # init kappas
        kappa_values[X_more_than_tau_index] = -np.inf
        kappa_values[X_less_than_tau_index] = np.inf

        # Initialize epsilons based on the condition
        epsilon_values[X_more_than_tau_index] = np.inf
        epsilon_values[X_less_than_tau_index] = -np.inf

        # init kappa and eps distances
        distances_kappa = np.power(
            dist_metric(C[ind_ref_cho].reshape(1, -1), C) + 1e-6, pow
        )
        distances_eps = dist_metric(C[ind_ref_cho].reshape(1, -1), C).flatten()

        for i, index_more in enumerate(X_more_than_tau_index):
            # kappa loop
            kappa_x_temp = (tau - Y[index_more, :]) / distances_kappa.flatten()
            kappa_values[index_more] = np.max(kappa_x_temp)

            # epsilon loop
            C_less_than_tau_index = np.where(Y[index_more, :] < tau)[0]
            if len(C_less_than_tau_index) == 0:
                epsilon_values[index_more] = np.inf
            else:
                epsilon_values[index_more] = np.min(
                    distances_eps[C_less_than_tau_index]
                )

        return min(kappa_values), max(epsilon_values)

    @property
    def name(self) -> str:
        return "robust_satisficing"


class LeninentRegret(Regret):
    """Lenient regret: max(0, tau - reward).

    This regret measure focuses on whether the reward meets a satisficing threshold.
    If reward >= tau, regret is 0 (satisfied).
    If reward < tau, regret is tau - reward (how much below the threshold).
    """

    def __init__(
        self,
        reward_log: ContinuousReward,
        reward_linear: ContinuousReward,
        tau_log: float = -25.0,
        tau_linear: float = None,
    ):
        """Initialize lenient calculator.

        Parameters
        ----------
        reward_log : ContinuousReward
            ContinuousReward with use_log=True.
        reward_linear : ContinuousReward
            ContinuousReward with use_log=False.
        tau_log : float, optional
            Satisficing threshold in log scale (e.g., -25 dB). Default is -25.0.
        tau_linear : float, optional
            Satisficing threshold in linear scale. Default is 10^tau_log.
        """
        super().__init__(reward_log, reward_linear)
        self.tau_log = tau_log
        self.tau_linear = tau_linear if tau_linear is not None else 10**tau_log

    def compute(self, user: User, action_idx: int, **kwargs) -> Tuple[float, float]:
        """Compute lenient regret: min(standard, satisficing).

        Takes the minimum of standard regret and satisficing regret so that
        if tau is unreachable, regret falls back to standard instead of
        penalizing for an impossible target.

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Selected action index.

        Returns
        -------
        tuple of (float, float)
            Tuple of (log_regret, linear_regret).
        """
        user_rss = user.get_rss()
        optimal_idx = int(np.argmax(user_rss))

        # Log scale
        optimal_log = self.reward_log.compute(user, optimal_idx)
        selected_log = self.reward_log.compute(user, action_idx)
        standard_log = optimal_log - selected_log
        sat_log = max(0.0, self.tau_log - selected_log)
        log_regret = min(standard_log, sat_log)

        # Linear scale
        optimal_linear = self.reward_linear.compute(user, optimal_idx)
        selected_linear = self.reward_linear.compute(user, action_idx)
        standard_linear = optimal_linear - selected_linear
        sat_linear = max(0.0, self.tau_linear - selected_linear)
        linear_regret = min(standard_linear, sat_linear)

        return log_regret, linear_regret

    @property
    def name(self) -> str:
        return "lenient"


class ThroughputRegret(ABC):
    """Base class for throughput regret computation.

    Works for both single-user and multi-user (combinatorial) settings.
    Always uses list interface: compute(users, beams, rates).
    For single-user: compute([user], [beam], [rate])
    """

    @abstractmethod
    def compute(self, users: list, beams: list, rates: list) -> float:
        """Compute throughput regret.

        Parameters
        ----------
        users : list
            List of User instances.
        beams : list
            List of beam indices, one per user.
        rates : list
            List of rate indices, one per user.

        Returns
        -------
        float
            Throughput regret value.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this regret type."""
        pass


class ThroughputStandardRegret(ThroughputRegret):
    """Standard throughput regret: optimal - achieved."""

    def __init__(
        self,
        rate_set: np.ndarray,
        optimal_throughput: float,
        psi: np.ndarray,
    ):
        """Initialize standard throughput regret calculator.

        Parameters
        ----------
        rate_set : np.ndarray
            Array of available rates.
        optimal_throughput : float
            Optimal average throughput.
        psi : np.ndarray
            Precomputed success probabilities of shape (num_users, total_beams, num_rates).
        """
        self.rate_set = np.array(rate_set)
        self.optimal = optimal_throughput
        self.psi = psi
        self.num_users = psi.shape[0]

    def compute(self, users: list, beams: list, rates: list) -> float:
        """Compute standard throughput regret.

        Parameters
        ----------
        users : list
            List of User instances.
        beams : list
            List of beam indices, one per user.
        rates : list
            List of rate indices, one per user.

        Returns
        -------
        float
            Regret = optimal_throughput - achieved_throughput.
        """
        num_users = len(users)
        throughput = sum(
            self.rate_set[r] * self.psi[u, b, r]
            for u, (b, r) in enumerate(zip(beams, rates))
        )
        avg_throughput = throughput / num_users
        return max(0.0, self.optimal - avg_throughput)

    @property
    def name(self) -> str:
        return "throughput_standard"


class ThroughputSatisficingRegret(ThroughputRegret):
    """Satisficing throughput regret: min(standard, satisficing).

    If target is unreachable, falls back to standard regret.
    """

    def __init__(
        self,
        rate_set: np.ndarray,
        target_throughput: float,
        optimal_throughput: float,
        psi: np.ndarray,
    ):
        """Initialize satisficing throughput regret calculator.

        Parameters
        ----------
        rate_set : np.ndarray
            Array of available rates.
        target_throughput : float
            Target average throughput per user.
        optimal_throughput : float
            Optimal average throughput for standard regret fallback.
        psi : np.ndarray
            Precomputed success probabilities of shape (num_users, total_beams, num_rates).
        """
        self.rate_set = np.array(rate_set)
        self.target = target_throughput
        self.optimal = optimal_throughput
        self.psi = psi
        self.num_users = psi.shape[0]

    def compute(self, users: list, beams: list, rates: list) -> float:
        """Compute satisficing throughput regret: min(standard, satisficing).

        Parameters
        ----------
        users : list
            List of User instances.
        beams : list
            List of beam indices, one per user.
        rates : list
            List of rate indices, one per user.

        Returns
        -------
        float
            min(standard_regret, satisficing_regret).
        """
        num_users = len(users)
        throughput = sum(
            self.rate_set[r] * self.psi[u, b, r]
            for u, (b, r) in enumerate(zip(beams, rates))
        )
        avg_throughput = throughput / num_users

        sat_regret = max(0.0, self.target - avg_throughput)
        standard_regret = max(0.0, self.optimal - avg_throughput)
        return min(standard_regret, sat_regret)

    @property
    def name(self) -> str:
        return "throughput_satisficing"


# Aliases for backward compatibility
CombinatorialRegret = ThroughputRegret
CombinatorialStandardRegret = ThroughputStandardRegret
CombinatorialSatisficingRegret = ThroughputSatisficingRegret
