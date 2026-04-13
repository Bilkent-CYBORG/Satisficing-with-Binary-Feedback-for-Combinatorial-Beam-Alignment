from abc import ABC, abstractmethod

import numpy as np


class ChannelProvider(ABC):
    """Abstract base class for channel data providers."""

    @abstractmethod
    def get_channels(self) -> np.ndarray:
        """Return all channels.

        Returns
        -------
        np.ndarray
            Channel array. Shape depends on provider:
            - MIMO: (num_bs, num_users, n_ue_ant, n_bs_ant, n_subcarriers)
            - Simple: (num_bs, num_users, N)
        """
        pass

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        """Return positions corresponding to channels.

        Returns
        -------
        np.ndarray
            Position array, typically shape (num_users, 2) or (num_users, 3).
        """
        pass

    @abstractmethod
    def get_position_dim(self) -> int:
        """Return dimension of position.

        Returns
        -------
        int
            Position dimension.
        """
        pass

    @abstractmethod
    def get_tx_pos(self) -> np.ndarray:
        """Return transmitter (base station) positions.

        Returns
        -------
        np.ndarray
            TX position array with shape (num_bs, 3).
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return parameters for cache key generation.

        Returns
        -------
        dict
            Dictionary of all parameters that affect channel generation.
        """
        pass
