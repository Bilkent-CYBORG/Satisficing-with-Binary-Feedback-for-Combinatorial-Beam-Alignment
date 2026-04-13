import matplotlib.pyplot as plt
import numpy as np

from obs.environment.environment import Environment


class User:
    """User with position-based actions.

    Each user has:
    - Position
    - Actions available at their position
    - RSS values at their position
    - Optional channel randomness (time-varying beam shifts)
    """

    def __init__(
        self,
        user_id: int,
        environment: Environment,
        initial_position: np.ndarray,
        eps: float = 0.0,
        channel_randomness: bool = False,
        channel_randomness_std: float = 3.0,
        position_noise: bool = False,
        position_noise_std: float = 5.0,
    ):
        """Initialize user.

        Parameters
        ----------
        user_id : int
            Unique user identifier.
        environment : Environment
            Environment instance.
        initial_position : np.ndarray
            Starting position (x, y).
        eps : float, optional
            Movement radius (0 = static user). Default is 0.0.
        channel_randomness : bool, optional
            If True, apply Gaussian beam index shift to model time-varying channels
            (as in array steering vector randomness). Default is False.
        channel_randomness_std : float, optional
            Standard deviation of the Gaussian shift. Default is 3.0
            (typical shifts of ±3, occasionally ±5 or more).
        position_noise : bool, optional
            If True, get_position() and get_second_to_last_position() can return
            noisy positions to simulate GPS/localization uncertainty. Default is False.
        position_noise_std : float, optional
            Standard deviation of position noise in meters. Default is 5.0.
        """
        self.user_id = user_id
        self.environment = environment
        self.initial_position = initial_position.copy()
        self.position = self.initial_position.copy()
        self.eps = eps
        self.position_history = [self.initial_position.copy()]
        self.channel_randomness = channel_randomness
        self.channel_randomness_std = channel_randomness_std
        self.position_noise = position_noise
        self.position_noise_std = position_noise_std

        # Initialize noisy position tracking
        self._noisy_position = (
            self._generate_noisy_position()
            if position_noise
            else self.initial_position.copy()
        )
        self._noisy_position_history = [self._noisy_position.copy()]

    def get_pos_index(self) -> int:
        """Get closest grid position index for current position.

        Returns
        -------
        int
            Index of the closest position in the environment.
        """
        return self.environment.get_pos_index(self.position)

    def get_actions(self) -> np.ndarray:
        """Get available actions at current position.

        Returns
        -------
        np.ndarray
            Actions array of shape (num_bs * K, action_dim).
        """
        return self.environment.get_actions_from_pos(self.position)

    def get_rss(self) -> np.ndarray:
        """Get RSS for all actions at current position.

        Returns
        -------
        np.ndarray
            RSS array of shape (num_bs * K,).
        """
        return self.environment.get_rss_from_pos(self.position)

    def get_observed_rss(self) -> np.ndarray:
        """Get observed RSS with optional channel randomness.

        If channel_randomness is enabled, applies a Gaussian beam index shift
        per base station to model time-varying channels.

        Returns
        -------
        np.ndarray
            RSS array of shape (num_bs * K,).
        """
        rss = self.get_rss()

        if not self.channel_randomness:
            return rss

        num_bs = self.environment.num_bs()
        K_pairs = self.environment.num_beams()
        K_tx = self.environment.num_tx_beams()
        K_rx = self.environment.num_rx_beams()
        shifted_rss = np.zeros_like(rss)

        for bs_idx in range(num_bs):
            index_shift = int(np.random.randn() * self.channel_randomness_std)
            start_idx = bs_idx * K_pairs
            end_idx = start_idx + K_pairs
            chunk = rss[start_idx:end_idx]

            if K_rx > 1:
                # MIMO: roll along TX dimension only
                chunk_2d = chunk.reshape(K_tx, K_rx)
                chunk_2d = np.roll(chunk_2d, index_shift, axis=0)
                shifted_rss[start_idx:end_idx] = chunk_2d.flatten()
            else:
                shifted_rss[start_idx:end_idx] = np.roll(chunk, index_shift)

        return shifted_rss

    def get_max_rss(self) -> float:
        """Get best possible RSS at current position.

        Returns
        -------
        float
            Maximum RSS for the current position.
        """
        return self.environment.get_max_rss_from_pos(self.position)

    def get_max_rss_index(self) -> int:
        """Get index of best action at current position.

        Returns
        -------
        int
            Index of the best RSS action at the current position.
        """
        return self.environment.get_max_rss_index_from_pos(self.position)

    def move_randomly(self) -> np.ndarray:
        """Move user to a new position within eps of current position.

        For static users (eps=0), position stays the same.

        Returns
        -------
        np.ndarray
            New position array (x, y) or (x, y, z).
        """
        if self.eps == 0:
            return self.position
        self.position = self.environment.sample_nearby_position(self.position, self.eps)
        self.position_history.append(self.position.copy())

        # Update noisy position if noise is enabled
        if self.position_noise:
            self._noisy_position = self._generate_noisy_position()
            self._noisy_position_history.append(self._noisy_position.copy())

        return self.position

    def move_around_center(self) -> np.ndarray:
        """Move user to a new position within eps of initial position.

        Always stays within eps radius of the initial/center position.
        For static users (eps=0), position stays the same.

        Returns
        -------
        np.ndarray
            New position array (x, y) or (x, y, z).
        """
        if self.eps == 0:
            return self.position
        self.position = self.environment.sample_nearby_position(
            self.initial_position, self.eps
        )
        self.position_history.append(self.position.copy())

        # Update noisy position if noise is enabled
        if self.position_noise:
            self._noisy_position = self._generate_noisy_position()
            self._noisy_position_history.append(self._noisy_position.copy())

        return self.position

    def reset(self):
        """Reset user to initial position.

        Returns
        -------
        np.ndarray
            Initial position array (x, y) or (x, y, z).
        """
        self.position = self.initial_position.copy()
        self.position_history = [self.initial_position.copy()]

        # Reset noisy position tracking
        if self.position_noise:
            self._noisy_position = self._generate_noisy_position()
            self._noisy_position_history = [self._noisy_position.copy()]

        return self.position

    def get_channel(self, bs_idx: int = 0) -> np.ndarray:
        """Get channel vector from this user to a specific base station.

        Parameters
        ----------
        bs_idx : int, optional
            Base station index. Default is 0.

        Returns
        -------
        np.ndarray
            Channel vector of shape (N, 1).
        """
        return self.environment.get_channel_from_pos(self.position, bs_idx)

    def get_position(self, noisy: bool = False) -> np.ndarray:
        """Get position of user.

        Parameters
        ----------
        noisy : bool, optional
            If True and position_noise is enabled, return the noisy position.
            Default is False (returns true position).

        Returns
        -------
        np.ndarray
            Position array (x, y) or (x, y, z).
        """
        if noisy and self.position_noise:
            return self._noisy_position
        return self.position

    def get_second_to_last_position(self, noisy: bool = False) -> np.ndarray:
        """Get second to last position of user.

        Parameters
        ----------
        noisy : bool, optional
            If True and position_noise is enabled, return the noisy second-to-last position.
            Default is False (returns true position).

        Returns
        -------
        np.ndarray
            Second to last position array (x, y) or (x, y, z).
        """
        if noisy and self.position_noise:
            if len(self._noisy_position_history) < 2:
                return self._noisy_position
            return self._noisy_position_history[-2]
        if len(self.position_history) < 2:
            return self.position
        return self.position_history[-2]

    def _generate_noisy_position(self) -> np.ndarray:
        """Generate noisy position by adding Gaussian noise to true position.

        Returns
        -------
        np.ndarray
            Noisy position array.
        """
        noise = np.random.randn(len(self.position)) * self.position_noise_std
        return self.position + noise

    def plot_trajectory(self, figsize: tuple = (10, 8), cmap: str = "viridis"):
        """Plot user trajectory with timestep heatmap coloring.

        If position noise is enabled, also plots the observed (noisy) trajectory
        with a different color/marker.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size tuple. Default is (10, 8).
        cmap : str, optional
            Colormap for timestep coloring. Default is 'viridis'.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.
        """
        if len(self.position_history) < 2:
            print("Not enough positions to plot trajectory.")
            return None

        positions = np.array(self.position_history)
        timesteps = np.arange(len(positions))

        fig, ax = plt.subplots(figsize=figsize)

        # Plot true trajectory line
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "b-",
            alpha=0.3,
            linewidth=1,
            label="True path",
        )

        # Scatter true positions with timestep coloring
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=timesteps,
            cmap=cmap,
            s=50,
            edgecolor="blue",
            linewidth=0.5,
        )

        # Plot noisy trajectory if position noise is enabled
        if self.position_noise and len(self._noisy_position_history) >= 2:
            noisy_positions = np.array(self._noisy_position_history)

            # Plot noisy trajectory line
            ax.plot(
                noisy_positions[:, 0],
                noisy_positions[:, 1],
                "r--",
                alpha=0.3,
                linewidth=1,
                label="Observed path",
            )

            # Scatter noisy positions with different marker
            ax.scatter(
                noisy_positions[:, 0],
                noisy_positions[:, 1],
                c=timesteps,
                cmap="Oranges",
                s=30,
                marker="x",
                linewidth=1,
                alpha=0.7,
            )

            # Draw lines connecting true to noisy positions
            for i in range(len(positions)):
                ax.plot(
                    [positions[i, 0], noisy_positions[i, 0]],
                    [positions[i, 1], noisy_positions[i, 1]],
                    "gray",
                    alpha=0.2,
                    linewidth=0.5,
                )

        # Mark start and end (true positions)
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            c="green",
            s=150,
            marker="o",
            edgecolor="black",
            linewidth=2,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            c="red",
            s=150,
            marker="X",
            edgecolor="black",
            linewidth=2,
            label="End",
            zorder=5,
        )

        # Colorbar for timestep
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Timestep")

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        title = f"User {self.user_id} Trajectory"
        if self.position_noise:
            title += f" (noise std={self.position_noise_std}m)"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_beams(
        self,
        figsize: tuple = (12, 4),
        use_log: bool = True,
        highlight_arm: int = None,
        save_path: str = None,
    ):
        """Plot RSS values for all beams at current position.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size tuple. Default is (12, 4).
        use_log : bool, optional
            If True, plot in dB scale (10*log10). Default is True.
        highlight_arm : int, optional
            Arm index to highlight (e.g., recommended arm). Default is None.
        save_path : str, optional
            Path to save figure. Default is None.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.
        """
        rss = self.get_rss()
        K = self.environment.num_beams()
        optimal_arm = self.get_max_rss_index()

        # Handle hierarchical codebook - extract finest level only
        if self.environment.use_hierarchical_codebook:
            hier_cb = self.environment.get_hierarchical_codebook()
            finest_level = hier_cb.H - 1
            finest_rss = []
            for i in range(K):
                flat_idx = hier_cb.level_beam_to_flat_idx(finest_level, i)
                finest_rss.append(rss[flat_idx])
            rss = np.array(finest_rss)
            # Optimal arm needs to be mapped to finest level too
            level, beam_idx = hier_cb.flat_idx_to_level_beam(optimal_arm)
            if level == finest_level:
                optimal_arm = beam_idx
            else:
                # Find finest level child
                current_level, current_idx = level, beam_idx
                while current_level < finest_level:
                    node = hier_cb.get_node(current_level, current_idx)
                    current_idx = node.zoom_in_idxs[len(node.zoom_in_idxs) // 2]
                    current_level += 1
                optimal_arm = current_idx

        if use_log:
            plot_rss = 10 * np.log10(rss)
            ylabel = "RSS (dB)"
        else:
            plot_rss = rss
            ylabel = "RSS"

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(plot_rss)), plot_rss, alpha=0.7, color="steelblue")

        # Mark optimal arm
        ax.axvline(
            x=optimal_arm,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Optimal ({optimal_arm})",
        )

        # Mark highlighted arm if provided
        if highlight_arm is not None and highlight_arm != optimal_arm:
            ax.axvline(
                x=highlight_arm,
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"Selected ({highlight_arm})",
            )

        ax.set_xlabel("Beam Index")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"User {self.user_id} Beam RSS at position ({self.position[0]:.1f}, {self.position[1]:.1f})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig
