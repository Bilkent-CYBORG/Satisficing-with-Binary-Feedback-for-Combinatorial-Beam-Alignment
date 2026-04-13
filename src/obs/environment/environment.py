from typing import Optional

import deepmimo as dm
import matplotlib.pyplot as plt

import numpy as np

from obs.environment.channel_provider import ChannelProvider
from obs.environment.codebook import HierarchicalCodebook
from obs.utils.utils import encode_action_compact


class Environment:
    """Environment class that wraps a channel provider."""

    def __init__(
        self,
        channel_provider: ChannelProvider,
        K=16,
        K_rx=None,
        p_set_dBm=50,
        eps=1e-18,
        use_hierarchical_codebook=False,
    ):
        """Initialize environment with a channel provider.

        Parameters
        ----------
        channel_provider : ChannelProvider
            Provider that supplies channel and position data.
        K : int, optional
            Number of TX beams (must be power of 2 if use_hierarchical_codebook=True).
            Default is 16.
        K_rx : int, optional
            Number of RX beams. None = auto (M if M > 1, else 1 for MISO).
        p_set_dBm : float, optional
            TX power in dBm. Default is 50.
        eps : float, optional
            Small constant for numerical comparisons. Default is 1e-18.
        use_hierarchical_codebook : bool, optional
            If True, use hierarchical codebook. Default is False.
        """
        self.eps = eps
        self.channel_provider = channel_provider
        self._channels = None
        self._positions = None
        self.N = self.channel_provider.N
        self.M = self.channel_provider.M
        self.p_set_dBm = p_set_dBm
        self.P = 10 ** (self.p_set_dBm / 10) / 1000
        self.sqrt_P = np.sqrt(self.P)
        self._rss = None
        self._actions = None

        self.zoom_flag = False
        self.pos_mask = None
        self.zoom_bounds = None

        # Beam counts
        self.K_tx = K
        self.K_rx = K_rx if K_rx is not None else self.M
        self.K_pairs = self.K_tx * self.K_rx

        # Codebook setup
        self.use_hierarchical_codebook = use_hierarchical_codebook
        self._hierarchical_codebook = None

        if use_hierarchical_codebook:
            self._hierarchical_codebook = self.get_hierarchical_codebook()
            self.tx_codebook = self._hierarchical_codebook.finest_codebook
        else:
            self.tx_codebook = self.generate_codebook(
                self.K_tx, self.channel_provider.bs_antenna_shape, self.N
            )

        self.codebook = self.tx_codebook  # backward compat alias
        self.rx_codebook = (
            self.generate_codebook(
                self.K_rx, self.channel_provider.ue_antenna_shape, self.M
            )
            if self.K_rx > 1
            else None
        )

        # force lazy loading
        self.channels()
        self.positions()
        self.get_actions()
        self.get_rss()

    def channels(self) -> np.ndarray:
        """Get all channels (lazy loaded).

        Returns
        -------
        np.ndarray
            Channel array with shape (num_bs, num_users, ...).
        """
        if self._channels is None:
            self._channels = self.channel_provider.get_channels()

        if self.zoom_flag:
            return self._channels[:, self.pos_mask]

        return self._channels

    def positions(self) -> np.ndarray:
        """Get all positions (lazy loaded).

        Returns
        -------
        np.ndarray
            Position array with shape (num_users, 2) or (num_users, 3).
        """
        if self._positions is None:
            self._positions = self.channel_provider.get_positions()

        # check if zoom is applied
        if self.zoom_flag:
            return self._positions[self.pos_mask]

        return self._positions

    def get_valid_positions(
        self, check_neighbors: bool = True, neighbor_radius: float = 10.0
    ) -> np.ndarray:
        """Get positions that have non-zero RSS values and valid neighbors.

        Filters out positions where all RSS values are zero or near-zero (eps).
        Optionally also checks that nearby positions within neighbor_radius have valid RSS,
        ensuring users can move around after initialization.

        Parameters
        ----------
        check_neighbors : bool, optional
            If True, also verify that nearby positions have valid RSS. Default is True.
        neighbor_radius : float, optional
            Radius to check for valid neighbors (should be >= eps_user). Default is 10.0.

        Returns
        -------
        np.ndarray
            Array of valid positions.
        """
        positions = self.positions()
        rss = self.get_rss()

        # Find positions where max RSS is greater than eps threshold
        valid_mask = np.max(rss, axis=1) > self.eps * 10

        if check_neighbors:
            # For each position, check that it has valid neighbors within radius
            for i, pos in enumerate(positions):
                if not valid_mask[i]:
                    continue

                # Find all positions within neighbor_radius
                distances = np.linalg.norm(positions[:, :2] - pos[:2], axis=1)
                neighbor_indices = np.where(
                    (distances > 0) & (distances <= neighbor_radius)
                )[0]

                if len(neighbor_indices) == 0:
                    # No neighbors found, mark as invalid
                    valid_mask[i] = False
                    continue

                # Check if at least some neighbors have valid RSS
                neighbor_valid = np.max(rss[neighbor_indices], axis=1) > self.eps * 10
                valid_neighbor_count = np.sum(neighbor_valid)

                # Require at least 30% of neighbors to be valid
                if valid_neighbor_count < 0.3 * len(neighbor_indices):
                    valid_mask[i] = False

        valid_positions = positions[valid_mask]

        if len(valid_positions) == 0:
            raise ValueError(
                "No valid positions found with non-zero RSS and valid neighbors!"
            )

        return valid_positions

    def num_bs(self) -> int:
        """Number of base stations.

        Returns
        -------
        int
            Number of base stations.
        """
        return len(self.channels())

    def num_beams(self) -> int:
        """Number of beam pairs per BS (K_tx * K_rx).

        Returns
        -------
        int
            Number of beam pairs per base station.
        """
        return self.K_pairs

    def num_tx_beams(self) -> int:
        """Number of TX beams per BS."""
        return self.K_tx

    def num_rx_beams(self) -> int:
        """Number of RX beams per BS."""
        return self.K_rx

    def get_actions(self) -> np.ndarray:
        """Generate all actions (lazy loaded).

        Returns
        -------
        np.ndarray
            If use_hierarchical_codebook=False:
                Shape (num_positions, num_bs * K, action_dim).
            If use_hierarchical_codebook=True:
                Shape (num_positions, num_bs * (2K-2), action_dim + 2).
                Extra 2 dims at end: [level, beam_idx_in_level].
        """
        if self._actions is None:
            bs_positions = self.channel_provider.get_tx_pos()
            beam_angles = np.array(
                [
                    np.degrees(
                        np.arcsin(np.clip(-1.0 + 2.0 * k / self.K_tx, -1.0, 1.0))
                    )
                    for k in range(self.K_tx)
                ]
            )
            rx_beam_angles = (
                np.array(
                    [
                        np.degrees(
                            np.arcsin(np.clip(-1.0 + 2.0 * k / self.K_rx, -1.0, 1.0))
                        )
                        for k in range(self.K_rx)
                    ]
                )
                if self.K_rx > 1
                else None
            )

            if self.use_hierarchical_codebook:
                hier_cb = self._hierarchical_codebook
                actions = []
                for pos in self.positions():
                    pos_actions = []
                    for bs_idx in range(self.num_bs()):
                        bs_pos = bs_positions[bs_idx]
                        for level in range(hier_cb.H):
                            cb_level = hier_cb.get_codebook(level)
                            for beam_idx in range(len(cb_level)):
                                node = hier_cb.get_node(level, beam_idx)
                                row = encode_action_compact(
                                    pos, bs_pos, node.center_angle
                                )
                                row = np.concatenate([row, [level, beam_idx]])
                                pos_actions.append(row)
                    actions.append(pos_actions)
                self._actions = np.array(actions)
            else:
                actions = []
                for pos in self.positions():
                    pos_actions = []
                    for bs_idx in range(self.num_bs()):
                        bs_pos = bs_positions[bs_idx]
                        for k in range(self.K_tx):
                            if rx_beam_angles is not None:
                                for kr in range(self.K_rx):
                                    row = encode_action_compact(
                                        pos, bs_pos, beam_angles[k], rx_beam_angles[kr]
                                    )
                                    pos_actions.append(row)
                            else:
                                row = encode_action_compact(pos, bs_pos, beam_angles[k])
                                pos_actions.append(row)
                    actions.append(pos_actions)
                self._actions = np.array(actions)

        if self.zoom_flag:
            return self._actions[self.pos_mask]

        return self._actions

    def get_actions_flat(self) -> np.ndarray:
        """Get all actions flattened to 2D.

        Returns
        -------
        np.ndarray
            Shape (num_positions * num_actions_per_pos, action_dim).
        """
        actions = self.get_actions()
        return actions.reshape(-1, actions.shape[-1])

    def get_pos_index(self, target_pos: np.ndarray) -> int:
        """Find the index of the closest position to the target position.

        Parameters
        ----------
        target_pos : np.ndarray
            Target position vector.

        Returns
        -------
        int
            Index of the closest position in the environment.
        """
        differences = self.positions() - target_pos
        distances = np.linalg.norm(differences, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx

    def get_actions_from_pos(self, target_pos: np.ndarray) -> np.ndarray:
        """Get all actions for the target position.

        Parameters
        ----------
        target_pos : np.ndarray
            Target position vector.

        Returns
        -------
        np.ndarray
            Actions array with shape (num_bs * K, pos_dim + num_bs + 2N).
        """
        pos_index = self.get_pos_index(target_pos)
        return self.get_actions()[pos_index]

    def get_actions_from_positions(self, target_positions: np.ndarray) -> np.ndarray:
        """Get all actions for multiple target positions.

        Parameters
        ----------
        target_positions : np.ndarray
            Array of target position vectors with shape (num_positions, pos_dim).

        Returns
        -------
        np.ndarray
            Actions array with shape (num_positions, num_bs * K, pos_dim + num_bs + 2N).
        """
        actions_list = []
        for target_pos in target_positions:
            actions_list.append(self.get_actions_from_pos(target_pos))
        return np.array(actions_list)

    def get_rss_from_pos(self, target_pos: np.ndarray) -> np.ndarray:
        """Get RSS for all actions for the target position.

        Parameters
        ----------
        target_pos : np.ndarray
            Target position vector.

        Returns
        -------
        np.ndarray
            RSS array with shape (num_bs * K,).
        """
        pos_index = self.get_pos_index(target_pos)
        return self.get_rss()[pos_index]

    def get_max_rss_from_pos(self, target_pos: np.ndarray) -> float:
        """Get the maximum RSS for the target position.

        Parameters
        ----------
        target_pos : np.ndarray
            Target position vector.

        Returns
        -------
        float
            Maximum RSS for the target position.
        """
        return self.get_rss_from_pos(target_pos).max()

    def get_max_rss_index_from_pos(self, target_pos: np.ndarray) -> int:
        """Get the index of the maximum RSS for the target position.

        Parameters
        ----------
        target_pos : np.ndarray
            Target position vector.

        Returns
        -------
        int
            Index of the maximum RSS for the target position.
        """
        return np.argmax(self.get_rss_from_pos(target_pos))

    def sample_nearby_position(self, current_pos: np.ndarray, eps: float) -> np.ndarray:
        """Sample a valid position within eps of current position.

        Only moves in xy plane. Uses zoom_bounds if zoom is applied.
        Falls back to current position after max_attempts to avoid infinite loops.

        Parameters
        ----------
        current_pos : np.ndarray
            Current position (x, y) or (x, y, z).
        eps : float
            Maximum distance from current position.

        Returns
        -------
        np.ndarray
            New position array with same shape as input.
        """
        max_attempts = 50
        for _ in range(max_attempts):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, eps)

            # Only move in xy plane
            new_pos = current_pos.copy()
            new_pos[0] += r * np.cos(theta)
            new_pos[1] += r * np.sin(theta)

            # Check zoom bounds if applied
            if self.zoom_flag:
                x_min, x_max, y_min, y_max = self.zoom_bounds
                if not (x_min <= new_pos[0] <= x_max and y_min <= new_pos[1] <= y_max):
                    continue

            # Check if position has valid RSS (not all close to eps)
            # this takes too long idk why like is there a reason ?
            rss = self.get_rss_from_pos(new_pos)
            if not np.allclose(rss, self.eps):
                return new_pos

        # Fallback: stay in place
        return current_pos

    def get_channel_from_pos(
        self, target_pos: np.ndarray, bs_idx: int = 0
    ) -> np.ndarray:
        """Get channel vector for a position to a specific base station.

        Parameters
        ----------
        target_pos : np.ndarray
            Position array (x, y) or (x, y, z).
        bs_idx : int, optional
            Base station index. Default is 0.

        Returns
        -------
        np.ndarray
            Channel array with shape (M, N).
        """
        closest_idx = self.get_pos_index(target_pos)
        ch_bs = self.channels()[bs_idx]
        ch_u = np.array(ch_bs[closest_idx])  # (M, N, n_sub)
        return ch_u.squeeze(-1)  # (M, N)

    def get_rss(self) -> np.ndarray:
        """Compute RSS for all positions and beams using vectorized F @ channel.

        Uses the formula from gp3.py: |F @ channel|.mean(axis=1).mean(axis=-1)

        Returns
        -------
        np.ndarray
            If use_hierarchical_codebook=False:
                Shape (num_positions, num_bs * K).
            If use_hierarchical_codebook=True:
                Shape (num_positions, num_bs * (2K-2)).
        """
        if self._rss is None:
            all_channels = self.channel_provider.get_channels()
            num_pos = len(self.channel_provider.get_positions())
            los = self.channel_provider.get_los()

            if self.use_hierarchical_codebook:
                hier_cb = self._hierarchical_codebook
                total_beams = 2 * self.K_tx - 2
                num_beams_total = self.num_bs() * total_beams
                rss = np.zeros((num_pos, num_beams_total))

                for bs_idx in range(self.num_bs()):
                    ch_bs = all_channels[bs_idx]
                    los_bs = los[bs_idx]
                    valid_mask = los_bs != -1
                    ch_valid = ch_bs[valid_mask]

                    beam_offset = bs_idx * total_beams
                    for level in range(hier_cb.H):
                        F_level = hier_cb.get_codebook(level)
                        beam_gains = (
                            np.abs(F_level @ ch_valid).mean(axis=1).mean(axis=-1)
                        )
                        rss_valid = beam_gains**2 * self.P

                        num_beams_level = F_level.shape[0]
                        start_idx = beam_offset
                        end_idx = beam_offset + num_beams_level
                        rss[valid_mask, start_idx:end_idx] = rss_valid
                        beam_offset += num_beams_level
            else:
                num_beams_total = self.num_bs() * self.K_pairs
                rss = np.zeros((num_pos, num_beams_total))
                F_tx = self.tx_codebook

                for bs_idx in range(self.num_bs()):
                    ch_bs = all_channels[bs_idx]
                    los_bs = los[bs_idx]
                    valid_mask = los_bs != -1
                    ch_valid = ch_bs[valid_mask]

                    if self.rx_codebook is not None:
                        # MIMO: |f_rx^H @ H @ f_tx|^2 * P
                        F_rx = self.rx_codebook
                        # ch_valid: (num_valid, M, N, n_sub)
                        # F_rx: (K_rx, M), F_tx: (K_tx, N)
                        eff_ch = np.einsum("rm, pmns -> rpns", F_rx.conj(), ch_valid)
                        beam_response = np.einsum("tn, rpns -> trps", F_tx, eff_ch)
                        beam_gains = np.abs(beam_response).mean(axis=-1)
                        rss_valid = beam_gains**2 * self.P
                        # (K_tx, K_rx, num_valid) -> (num_valid, K_pairs)
                        rss_valid = rss_valid.reshape(self.K_pairs, -1).T
                    else:
                        # MISO: |F_tx @ h|^2 * P
                        beam_gains = np.abs(F_tx @ ch_valid).mean(axis=1).mean(axis=-1)
                        rss_valid = beam_gains**2 * self.P

                    start_idx = bs_idx * self.K_pairs
                    end_idx = start_idx + self.K_pairs
                    rss[valid_mask, start_idx:end_idx] = rss_valid

            self._rss = rss

        self._rss = np.maximum(self._rss, self.eps)

        if self.zoom_flag:
            return self._rss[self.pos_mask]
        return self._rss

    def generate_codebook(self, K: int, antenna_shape: tuple, N: int) -> np.ndarray:
        """Generate DFT codebook using DeepMIMO steering vectors.

        Parameters
        ----------
        K : int
            Number of beams.
        antenna_shape : tuple
            Antenna array shape for dm.steering_vec.
        N : int
            Number of antenna elements.

        Returns
        -------
        np.ndarray
            Codebook array with shape (K, N).
        """
        codebook = np.zeros((K, N), dtype=np.complex128)

        for k in range(K):
            u = -1.0 + (2.0 * k) / K
            phi_deg = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))

            dm_vec = dm.steering_vec(antenna_shape, phi=float(phi_deg)).squeeze()
            dm_vec = dm_vec / (np.linalg.norm(dm_vec) + 1e-12)

            codebook[k] = dm_vec

        return codebook

    def get_hierarchical_codebook(
        self, num_levels: Optional[int] = None, g: int = 2
    ) -> HierarchicalCodebook:
        """Get hierarchical codebook (lazy loaded).

        Parameters
        ----------
        num_levels : int, optional
            Number of levels. Default is log2(K).
        g : int, optional
            Branching factor. Default is 2 for binary tree.

        Returns
        -------
        HierarchicalCodebook
            HierarchicalCodebook instance.
        """
        return HierarchicalCodebook(
            K=self.K_tx,
            N=self.N,
            antenna_shape=self.channel_provider.bs_antenna_shape,
            num_levels=num_levels,
            g=g,
        )

    def zoom_to_box(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Zoom to rectangular region.

        Parameters
        ----------
        x_min : float
            Minimum x coordinate.
        x_max : float
            Maximum x coordinate.
        y_min : float
            Minimum y coordinate.
        y_max : float
            Maximum y coordinate.

        Raises
        ------
        ValueError
            If zoom bounds contain no positions.
        """
        all_positions = self.channel_provider.get_positions()
        self.pos_mask = (
            (all_positions[:, 0] >= x_min)
            & (all_positions[:, 0] <= x_max)
            & (all_positions[:, 1] >= y_min)
            & (all_positions[:, 1] <= y_max)
        )

        if np.sum(self.pos_mask) == 0:
            raise ValueError("Zoom bounds contain no positions")

        self.zoom_flag = True
        self.zoom_bounds = (x_min, x_max, y_min, y_max)

    def reset_zoom(self):
        """Reset to full position space."""
        self.zoom_flag = False
        self.pos_mask = None
        self.zoom_bounds = None

    def plot_best_beams(self, save_path: str = "best_beams_map.png"):
        """Plot best beams using DeepMIMO's plot_coverage.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. Default is "best_beams_map.png".

        Returns
        -------
        matplotlib.axes.Axes or tuple
            If use_hierarchical_codebook=False:
                Single axis with best beam index.
            If use_hierarchical_codebook=True:
                Figure with two subplots: best level and best beam_idx in that level.
        """
        positions = self.positions()
        rss = self.get_rss()

        if self.use_hierarchical_codebook:
            hier_cb = self._hierarchical_codebook
            best_flat_idx = np.argmax(rss, axis=1)

            # Convert flat index to (level, beam_idx)
            best_levels = np.zeros(len(best_flat_idx))
            best_beam_idxs = np.zeros(len(best_flat_idx))

            for i, flat_idx in enumerate(best_flat_idx):
                level, beam_idx = hier_cb.flat_idx_to_level_beam(flat_idx)
                best_levels[i] = level
                best_beam_idxs[i] = beam_idx

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            dm.plot_coverage(
                positions,
                best_levels,
                title="Best Level",
                cbar_title="Level",
                ax=axes[0],
            )

            dm.plot_coverage(
                positions,
                best_beam_idxs,
                title="Best Beam Index (in level)",
                cbar_title="Beam idx",
                ax=axes[1],
            )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)

            plt.show()
            return fig, axes
        else:
            best_beams = np.argmax(rss, axis=1).astype(float)

            ax = dm.plot_coverage(
                positions, best_beams, title="Best Beams", cbar_title="Best beam index"
            )

            if save_path:
                plt.savefig(save_path, dpi=150)

            plt.show()
            return ax
