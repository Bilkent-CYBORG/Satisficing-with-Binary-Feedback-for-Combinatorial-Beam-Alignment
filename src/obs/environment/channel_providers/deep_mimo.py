from typing import Optional

import deepmimo as dm
import numpy as np

from obs.environment.channel_provider import ChannelProvider


class DeepMIMOProvider(ChannelProvider):
    """Channel provider using DeepMIMO dataset.

    Auto-downloads and loads the scenario on initialization,
    then computes MIMO channels with configurable parameters.
    """

    def __init__(
        self,
        scenario: str,
        num_bs: int,
        # Antenna parameters
        bs_antenna_shape: tuple[int, int] = (1, 1),
        bs_antenna_spacing: float = 0.5,
        ue_antenna_shape: tuple[int, int] = (1, 1),
        ue_antenna_spacing: float = 0.5,
        # OFDM parameters
        subcarriers: int = 1,
        selected_subcarriers: Optional[list[int]] = None,
        # Channel parameters
        freq_domain: bool = False,
        num_paths: int = 1,
    ):
        """Initialize DeepMIMO provider.

        Parameters
        ----------
        scenario : str
            DeepMIMO scenario name (e.g., 'city_3_houston_3p5').
        num_bs : int
            Number of base stations to use.
        bs_antenna_shape : tuple of int, optional
            BS antenna array shape [rows, cols]. Default is (1, 1).
        bs_antenna_spacing : float, optional
            BS antenna element spacing in wavelengths. Default is 0.5.
        ue_antenna_shape : tuple of int, optional
            UE antenna array shape [rows, cols]. Default is (1, 1).
        ue_antenna_spacing : float, optional
            UE antenna element spacing in wavelengths. Default is 0.5.
        subcarriers : int, optional
            Number of OFDM subcarriers. Default is 1.
        selected_subcarriers : list of int, optional
            Which subcarriers to generate. Default is [0].
        freq_domain : bool, optional
            If True, generate frequency domain (OFDM) channels. Default is False.
        num_paths : int, optional
            Number of paths per user. Default is 1.
        """
        self.num_bs = num_bs
        self.scenario = scenario
        self.bs_antenna_shape = bs_antenna_shape
        self.bs_antenna_spacing = bs_antenna_spacing
        self.ue_antenna_shape = ue_antenna_shape
        self.ue_antenna_spacing = ue_antenna_spacing
        self.subcarriers = subcarriers
        self.selected_subcarriers = (
            selected_subcarriers if selected_subcarriers else [0]
        )
        self.freq_domain = freq_domain
        self.num_paths = num_paths
        self.N = bs_antenna_shape[0] * bs_antenna_shape[1]
        self.M = ue_antenna_shape[0] * ue_antenna_shape[1]
        dm.download(scenario)
        self.dataset = dm.load(scenario)

        params = dm.ChannelParameters()
        params.bs_antenna.shape = list(bs_antenna_shape)
        params.bs_antenna.spacing = bs_antenna_spacing
        params.ue_antenna.shape = list(ue_antenna_shape)
        params.ue_antenna.spacing = ue_antenna_spacing
        params.ofdm.subcarriers = subcarriers
        params.ofdm.selected_subcarriers = self.selected_subcarriers
        params.freq_domain = freq_domain
        params.num_paths = num_paths

        # Compute channels
        self.dataset.compute_channels(params)

        # Handle num_bs=1 case: wrap single values into lists for consistent access
        if not isinstance(self.dataset.channels, (list, tuple)):
            self.dataset.channels = [self.dataset.channels]
        if not isinstance(self.dataset.rx_pos, (list, tuple)):
            self.dataset.rx_pos = [self.dataset.rx_pos]
        if not isinstance(self.dataset.power, (list, tuple)):
            self.dataset.power = [self.dataset.power]
        if not isinstance(self.dataset.los, (list, tuple)):
            self.dataset.los = [self.dataset.los]
        if not isinstance(self.dataset.tx_pos, (list, tuple)):
            self.dataset.tx_pos = [self.dataset.tx_pos]
        if not isinstance(self.dataset.tx_ori, (list, tuple)):
            self.dataset.tx_ori = [self.dataset.tx_ori]

    def get_num_bs(self) -> int:
        return self.num_bs

    def get_channels(self) -> np.ndarray:
        """Return channels for all base stations.

        Returns
        -------
        np.ndarray
            Channel array with shape (num_bs, num_users, n_ue_ant, n_bs_ant, n_subcarriers).
        """
        channels_raw = self.dataset.channels

        if len(channels_raw) < self.num_bs:
            raise ValueError(
                f"Requested {self.num_bs} BS but only {len(channels_raw)} available in dataset"
            )
        return np.stack([np.asarray(ch) for ch in channels_raw[: self.num_bs]], axis=0)

    def get_positions(self) -> np.ndarray:
        """Return receiver positions.

        Returns
        -------
        np.ndarray
            Position array with shape (num_users, 3) or (num_users, 2).
        """
        rx_pos_raw = self.dataset.rx_pos

        # return first BS's positions
        return np.asarray(rx_pos_raw[0])

    def get_los(self) -> np.ndarray:
        """Return LOS status for each position.

        Returns
        -------
        np.ndarray
            LOS array with shape (num_users,). Values of -1 indicate invalid channels.
        """
        los_raw = self.dataset.los

        if len(los_raw) < self.num_bs:
            raise ValueError(
                f"Requested {self.num_bs} BS but only {len(los_raw)} available in dataset"
            )

        return np.stack([np.asarray(los) for los in los_raw[: self.num_bs]], axis=0)

    def get_position_dim(self) -> int:
        """Return dimension of position.

        Returns
        -------
        int
            Position dimension.
        """
        return self.get_positions().shape[1]

    def get_tx_pos(self) -> np.ndarray:
        """Return transmitter (base station) positions.

        Returns
        -------
        np.ndarray
            TX position array with shape (num_bs, 3).
        """
        tx_pos_raw = self.dataset.tx_pos

        if len(tx_pos_raw) < self.num_bs:
            raise ValueError(
                f"Requested {self.num_bs} BS but only {len(tx_pos_raw)} available in dataset"
            )
        return np.stack(
            [np.asarray(pos).flatten() for pos in tx_pos_raw[: self.num_bs]], axis=0
        )

    def get_tx_ori(self) -> np.ndarray:
        """Return transmitter (base station) orientations.

        Returns
        -------
        np.ndarray
            TX orientation array with shape (num_bs, ...).
        """
        tx_ori_raw = self.dataset.tx_ori

        if len(tx_ori_raw) < self.num_bs:
            raise ValueError(
                f"Requested {self.num_bs} BS but only {len(tx_ori_raw)} available in dataset"
            )
        return np.stack([np.asarray(ori) for ori in tx_ori_raw[: self.num_bs]], axis=0)

    def get_power(self, bs_idx: Optional[int] = None):
        """Return base station power distribution from the dataset.

        If no valid bs_idx is given, returns the maximum power across all base stations.

        Parameters
        ----------
        bs_idx : int, optional
            Base station index. If None, returns maximum power across all BS.

        Returns
        -------
        DeepMIMOArray
            Power distribution as a DeepMIMOArray.
        """
        p_out = self.dataset.power[0][:, 0].copy()

        if bs_idx is not None:
            if not (0 <= bs_idx < self.num_bs):
                raise ValueError(f"bs_idx={bs_idx} out of range [0, {self.num_bs-1}]")
            p_out[:] = np.asarray(self.dataset.power[bs_idx][:, 0])
            return p_out

        powers = np.stack(
            [
                np.asarray(self.dataset.power[bs][:, 0])
                for bs in range(len(self.dataset.power))
            ],
            axis=0,
        )
        powers[np.isnan(powers)] = -np.inf
        p_out[:] = powers.max(axis=0)

        return p_out

    def get_params(self) -> dict:
        """Return parameters for cache key generation.

        Returns
        -------
        dict
            Dictionary of all parameters that affect channel generation.
        """
        return {
            "scenario": self.scenario,
            "num_bs": self.num_bs,
            "bs_antenna_shape": list(self.bs_antenna_shape),
            "bs_antenna_spacing": self.bs_antenna_spacing,
            "ue_antenna_shape": list(self.ue_antenna_shape),
            "ue_antenna_spacing": self.ue_antenna_spacing,
            "subcarriers": self.subcarriers,
            "selected_subcarriers": self.selected_subcarriers,
            "freq_domain": self.freq_domain,
            "num_paths": self.num_paths,
        }

    def plot(self, bs_idx: Optional[int] = None):
        ax = self.get_power(bs_idx=bs_idx).plot()
        return ax

    def debug_shapes(self):
        print("channels type:", type(self.dataset.channels))
        if isinstance(self.dataset.channels, (list, tuple)):
            print("channels[0] shape:", np.asarray(self.dataset.channels[0]).shape)
        else:
            print("channels shape:", np.asarray(self.dataset.channels).shape)

        rx = self.dataset.rx_pos
        print("rx_pos type:", type(rx))
        print(
            "rx_pos[0] shape:" if isinstance(rx, (list, tuple)) else "rx_pos shape:",
            (
                np.asarray(rx[0]).shape
                if isinstance(rx, (list, tuple))
                else np.asarray(rx).shape
            ),
        )

        pw = self.dataset.power
        print("power len:", len(pw), "power[0] shape:", np.asarray(pw[0]).shape)
