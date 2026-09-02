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
    """Binary reward (ACK/NACK).

    Two feedback modes (see `mode`):
      * "bler" (default): ACK ~ Bernoulli(psi), psi = 1 - BLER_r(SNR) at the
        FIXED (noiseless, static-geometry) SNR. Stationary i.i.d. feedback,
        matching the revised system model / finite-time analysis. The logistic
        BLER waterfall is anchored so psi = 0.9 (10% BLER) at the MCS Shannon
        threshold gamma_th = 2^R - 1 (the 3GPP TS 38.214 CQI/MCS operating
        point), with a physical ~bler_width_db (10%-90%) waterfall.
      * "hard": original hard Shannon threshold on the observed (beam-shift
        noisy) RSS -- reproduces the repo's published behavior.
    """

    def __init__(self, rate_set: np.ndarray, noise_var: float = 1e-10,
                 mode: str = "bler", bler_width_db: float = 2.0,
                 bler_anchor_psi: float = 0.9):
        """Initialize binary reward function.

        Parameters
        ----------
        rate_set : np.ndarray
            Array of rates (bits/symbol), e.g. [2.4, 3.9, 5.5, 6.6].
        noise_var : float, optional
            Noise variance for SNR computation. Default is 1e-10.
        mode : str, optional
            "bler" (fixed-SNR Bernoulli block-error) or "hard" (Shannon
            threshold on observed RSS). Default "bler".
        bler_width_db : float, optional
            10%-90% BLER waterfall width in dB. Default 2.0.
        """
        self.rate_set = np.array(rate_set)
        self.noise_var = noise_var
        self.mode = mode
        self.bler_width_db = bler_width_db
        self.bler_anchor_psi = bler_anchor_psi
        # SNR thresholds: gamma_th = 2^R - 1
        self.gamma_th = 2**self.rate_set - 1

    def success_prob(self, snr, rate_idx):
        """psi = 1 - BLER_r(snr): block-success probability at fixed SNR.

        Logistic waterfall in the dB domain. `bler_anchor_psi` is the block-SUCCESS
        probability at the MCS Shannon threshold gamma_th = 2^R - 1:
          * 0.9 -> 10% BLER at threshold (3GPP CQI/MCS operating point); the
            midpoint sits bler_width_db/2 BELOW the threshold.
          * 0.5 -> 50% BLER at threshold; the midpoint sits ON the threshold,
            shifting the waterfall bler_width_db/2 dB right (strictly harder).
        The 10%-90% width is bler_width_db for any anchor. snr may be scalar/array.
        """
        snr_db = 10.0 * np.log10(np.maximum(np.asarray(snr, float), 1e-30))
        thr_db = 10.0 * np.log10(self.gamma_th[rate_idx])
        k = 2.0 * np.log(9.0) / self.bler_width_db
        a = float(self.bler_anchor_psi)
        mid_db = thr_db - np.log(a / (1.0 - a)) / k
        return 1.0 / (1.0 + np.exp(-np.clip(k * (snr_db - mid_db), -30, 30)))

    def compute(self, user: User, action_idx: int, rate_idx: int = 0, **kwargs) -> int:
        """Return ACK(1)/NACK(0).

        Three feedback modes:
          * "bler": Bernoulli draw at the FIXED (noiseless) SNR. Per-slot
            randomness = block-decoding noise only, i.i.d. across slots
            (revised system model; stationary psi).
          * "hard": Shannon threshold on the observed (beam-shift) RSS. Per-slot
            randomness = beam-misalignment only (original repo behavior).
          * "hard_bler": BOTH stacked -- beam-misalignment shifts the SNR each
            slot AND a BLER Bernoulli block-error is drawn at that shifted SNR.
            Most realistic / hardest: misalignment + decoding noise compounded.

        Parameters
        ----------
        user : User
            User instance.
        action_idx : int
            Selected beam index.
        rate_idx : int, optional
            Selected rate index. Default is 0.
        """
        if self.mode == "bler":
            rss = user.get_rss()[action_idx]           # fixed, noiseless
            snr = rss / self.noise_var
            return int(np.random.random() < self.success_prob(snr, rate_idx))
        if self.mode == "hard_bler":
            # beam-shifted (misaligned) SNR this slot, THEN BLER Bernoulli on it
            rss = user.get_observed_rss()[action_idx]
            snr = rss / self.noise_var
            return int(np.random.random() < self.success_prob(snr, rate_idx))
        # "hard": original behavior
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
