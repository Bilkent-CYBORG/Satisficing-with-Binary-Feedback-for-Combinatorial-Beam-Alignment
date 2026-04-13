from abc import ABC, abstractmethod

import numpy as np

from obs.user.user import User


class RewardFunction(ABC):
    """Base class for reward computation."""

    @abstractmethod
    def compute(self, user: User, action_idx: int, **kwargs) -> float:
        """Compute reward for user taking action.

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Index of selected action.
        **kwargs
            Additional arguments.

        Returns
        -------
        float
            Reward value.
        """
        pass


class ContinuousReward(RewardFunction):
    """Continuous reward (RSS/beam power).

    Used by contextual bandits.
    """

    def __init__(
        self,
        use_log: bool = True,
        noise_std: float = 0.1,
        noise_type: str = "heteroskedastic",
    ):
        """Initialize continuous reward function.

        Parameters
        ----------
        use_log : bool, optional
            If True, return log-transformed reward. Default is True.
        noise_std : float, optional
            Base noise standard deviation. Default is 0.1.
        noise_type : str, optional
            "heteroskedastic" (signal-dependent) or "homoskedastic" (constant).
            Default is "heteroskedastic".
        """
        self.use_log = use_log
        self.noise_std = noise_std
        self.noise_type = noise_type

    def compute(self, user: User, action_idx: int, **kwargs) -> float:
        """Return RSS reward at action, optionally log-transformed.

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Index of selected action.

        Returns
        -------
        float
            RSS reward (or log RSS if use_log=True).
        """
        rss = user.get_rss()[action_idx]
        if self.use_log:
            return np.log10(rss)
        return rss

    def compute_noisy(self, user: User, action_idx: int, **kwargs) -> float:
        """Return noisy reward for GP update.

        Noise is added in LINEAR domain (before log transform) for realistic modeling.
        - Heteroskedastic: noise_std proportional to RSS (signal-dependent)
        - Homoskedastic: constant noise_std

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Index of selected action.

        Returns
        -------
        float
            Noisy reward (log-transformed if use_log=True).
        """
        rss = user.get_rss()[action_idx]

        # Add noise in linear domain
        if self.noise_type == "heteroskedastic":
            # Heteroskedastic: noise std proportional to signal strength
            actual_noise_std = self.noise_std * rss
        else:
            # Homoskedastic: constant noise std
            actual_noise_std = self.noise_std

        noisy_rss = rss + np.random.normal(0, actual_noise_std)
        # Ensure positive for log
        noisy_rss = max(noisy_rss, 1e-20)

        if self.use_log:
            return np.log10(noisy_rss)
        return noisy_rss


class BinaryReward(RewardFunction):
    """Binary reward (ACK/NACK based on SNR threshold).

    Used by combinatorial bandits.
    """

    def __init__(self, rate_set: np.ndarray, noise_var: float = 1e-10):
        """Initialize binary reward function.

        Parameters
        ----------
        rate_set : np.ndarray
            Array of rates (bits/symbol), e.g. [6, 8, 10, 12].
        noise_var : float, optional
            Noise variance for SNR computation. Default is 1e-10.
        """
        self.rate_set = np.array(rate_set)
        self.noise_var = noise_var
        # SNR thresholds: gamma_th = 2^R - 1
        self.gamma_th = 2**self.rate_set - 1

    def compute(self, user: User, action_idx: int, rate_idx: int = 0, **kwargs) -> int:
        """Return 1 (ACK) if SNR >= threshold, else 0 (NACK).

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Selected beam index.
        rate_idx : int, optional
            Selected rate index. Default is 0.

        Returns
        -------
        int
            1 for success (ACK), 0 for failure (NACK).
        """
        rss = user.get_observed_rss()[action_idx]
        snr = rss / self.noise_var
        return 1 if snr >= self.gamma_th[rate_idx] else 0

    def get_num_rates(self) -> int:
        """Return number of available rates.

        Returns
        -------
        int
            Number of rates in rate_set.
        """
        return len(self.rate_set)

    def get_rate(self, rate_idx: int) -> float:
        """Return rate value at index.

        Parameters
        ----------
        rate_idx : int
            Rate index.

        Returns
        -------
        float
            Rate value (bits/symbol).
        """
        return self.rate_set[rate_idx]
