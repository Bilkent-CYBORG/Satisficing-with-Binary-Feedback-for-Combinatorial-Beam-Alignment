"""Combinatorial simulation for multi-user beam assignment experiments."""

from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from obs.algorithms.combinatorial.algorithm import CombinatorialAlgorithm
from obs.algorithms.combinatorial.objectives import compute_jain_index
from obs.simulation.regret import CombinatorialSatisficingRegret
from obs.simulation.reward import BinaryReward
from obs.simulation.simulation import Simulation

from obs.user.user import User


class CombinatorialSimulation(Simulation):
    """Simulation for combinatorial bandits (multi-user beam assignment).

    - Multiple users moving with small epsilon
    - CUCB, CTS, SAT-CTS algorithms
    - Binary (ACK/NACK) feedback based on SNR threshold
    """

    def __init__(
        self,
        environment,
        num_users: int = 3,
        user_positions: np.ndarray = None,
        rate_set: np.ndarray = None,
        noise_var: float = 1e-10,
        target_throughput: float = 8.0,
        eps_user: float = 0.0,
        channel_randomness: bool = True,
        channel_randomness_std: float = 5.0,
    ):
        """Initialize combinatorial simulation.

        Parameters
        ----------
        environment : Environment
            Environment instance.
        num_users : int, optional
            Number of users. Default is 3. Ignored if user_positions is provided.
        user_positions : np.ndarray, optional
            Specific positions for users, shape (num_users, 2) or (num_users, 3).
            If None, users spawn at random valid positions.
        rate_set : np.ndarray, optional
            Array of available rates (bits/symbol). Default is [8.0, 10.0, 12.0].
        noise_var : float, optional
            Noise variance for SNR computation. Default is 1e-10.
        target_throughput : float, optional
            Target average throughput per user. Default is 8.0.
        eps_user : float, optional
            User movement radius. Default is 0.0 (static).
        channel_randomness : bool, optional
            If True, apply random beam shift. Default is True.
        channel_randomness_std : float, optional
            Standard deviation for beam index shift. Default is 5.0.
        """
        self.user_positions = (
            np.array(user_positions) if user_positions is not None else None
        )
        self.num_users = (
            len(self.user_positions) if self.user_positions is not None else num_users
        )
        self.eps_user = eps_user
        self.channel_randomness = channel_randomness
        self.channel_randomness_std = channel_randomness_std

        self.rate_set = (
            np.array(rate_set) if rate_set is not None else np.array([8.0, 10.0, 12.0])
        )
        self.num_rates = len(self.rate_set)
        self.noise_var = noise_var
        self.target_throughput = target_throughput

        # SNR thresholds for each rate
        self.gamma_th = 2**self.rate_set - 1

        # Binary reward function
        super().__init__(environment, BinaryReward(self.rate_set, noise_var))

    def create_users(self) -> List[User]:
        """Create users at specified or random valid positions.

        Returns
        -------
        list of User
            List of User instances.
        """
        if self.user_positions is not None:
            # Snap provided positions to closest environment positions
            env_positions = self.environment.positions()
            positions = []
            for pos in self.user_positions:
                dists = np.linalg.norm(env_positions[:, : len(pos)] - pos, axis=1)
                closest_pos = env_positions[np.argmin(dists)]
                positions.append(closest_pos)
        else:
            valid_positions = self.environment.get_valid_positions(
                check_neighbors=False
            )
            if len(valid_positions) < self.num_users:
                raise ValueError(
                    f"Not enough valid positions ({len(valid_positions)}) "
                    f"for {self.num_users} users."
                )
            indices = np.random.choice(
                len(valid_positions), size=self.num_users, replace=False
            )
            positions = [valid_positions[i] for i in indices]

        users = []
        for u in range(self.num_users):
            users.append(
                User(
                    user_id=u,
                    environment=self.environment,
                    initial_position=positions[u],
                    eps=self.eps_user,
                    channel_randomness=self.channel_randomness,
                    channel_randomness_std=self.channel_randomness_std,
                )
            )
        return users

    def _estimate_psi(self, users: List[User]):
        """Estimate success probabilities for all (user, beam, rate) triplets.

        Uses sigmoid approximation based on SNR margin, same as source.

        Parameters
        ----------
        users : list of User
            List of User instances.
        """
        total_beams = users[0].get_rss().shape[0]
        self.total_beams = total_beams
        self.psi = np.zeros((self.num_users, total_beams, self.num_rates))

        for u, user in enumerate(users):
            rss = user.get_rss()
            snr = rss / self.noise_var

            thr = self.gamma_th[None, :]
            snr_expanded = snr[:, None]

            margin = (snr_expanded - thr) / np.sqrt(np.maximum(thr, 1.0))
            margin_clipped = np.clip(0.5 * margin, -30, 30)
            self.psi[u] = 1.0 / (1.0 + np.exp(-margin_clipped))

        # Compute optimal throughput for standard regret
        self._compute_optimal_throughput(users)

    def _compute_optimal_throughput(self, users: List[User]):
        """Compute optimal throughput using Hungarian algorithm.

        Parameters
        ----------
        users : list of User
            List of User instances.
        """
        from scipy.optimize import linear_sum_assignment

        throughput_matrix = self.rate_set[None, None, :] * self.psi
        best_rates = throughput_matrix.argmax(axis=2)
        scores = np.take_along_axis(
            throughput_matrix, best_rates[..., None], axis=2
        ).squeeze(-1)

        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)

        total_throughput = 0.0
        for u_idx, b_idx in zip(row_ind, col_ind):
            r_idx = best_rates[u_idx, b_idx]
            total_throughput += self.rate_set[r_idx] * self.psi[u_idx, b_idx, r_idx]

        self.optimal_throughput = total_throughput / len(users)

    def play(self, users: List[User], beams: List[int], rates: List[int]) -> np.ndarray:
        """Execute beam-rate assignment and get feedback.

        Parameters
        ----------
        users : list of User
            List of User instances.
        beams : list of int
            List of beam indices, one per user.
        rates : list of int
            List of rate indices, one per user.

        Returns
        -------
        np.ndarray
            ack_nack array (1=ACK, 0=NACK) for each user.
        """
        ack_nack = np.zeros(self.num_users, dtype=int)

        for u, user in enumerate(users):
            ack_nack[u] = self.reward_function.compute(
                user, beams[u], rate_idx=rates[u]
            )

        return ack_nack

    def run(
        self,
        algorithms: Dict[str, CombinatorialAlgorithm],
        T: int,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run combinatorial bandit experiment.

        Parameters
        ----------
        algorithms : dict
            Dict of {"name": CombinatorialAlgorithm}.
        T : int
            Number of rounds.
        verbose : bool, optional
            Print progress. Default is True.

        Returns
        -------
        dict
            Results dictionary with histories and statistics.
        """
        # Create users and estimate psi
        users = self.create_users()
        self._estimate_psi(users)

        # Create regret calculator
        self.regret_calculator = CombinatorialSatisficingRegret(
            rate_set=self.rate_set,
            target_throughput=self.target_throughput,
            optimal_throughput=self.optimal_throughput,
            psi=self.psi,
        )

        # Initialize histories
        histories = {}
        for name, algo in algorithms.items():
            histories[name] = {
                "throughputs": [],
                "ack_rates": [],
                "regrets": [],
                "per_user_throughputs": [],
                "jain_indices": [],
                "log_utilities": [],
            }
            algo._reset_cumulative()

        if verbose:
            print(f"Running simulation for T={T} rounds")
            print(
                f"  {self.num_users} users, {self.total_beams} beams, {self.num_rates} rates"
            )
            print(f"  Target throughput: {self.target_throughput}")
            for u, user in enumerate(users):
                print(f"  User {u}: position = {user.position[:2]}")

        for t in tqdm(range(1, T + 1), desc="Simulation"):
            for name, algo in algorithms.items():
                beams, rates = algo.select_action(t)

                ack_nack = self.play(users, beams, rates)

                throughput = np.mean(ack_nack * self.rate_set[rates])

                regret = self.regret_calculator.compute(users, beams, rates)

                algo.update(beams, rates, ack_nack)

                per_user_throughput = ack_nack * self.rate_set[rates]
                jain_idx = compute_jain_index(algo.cumulative_throughputs)
                log_utility = np.sum(np.log(algo.cumulative_throughputs + 1e-10))

                histories[name]["throughputs"].append(throughput)
                histories[name]["ack_rates"].append(np.mean(ack_nack))
                histories[name]["regrets"].append(regret)
                histories[name]["per_user_throughputs"].append(
                    per_user_throughput.copy()
                )
                histories[name]["jain_indices"].append(jain_idx)
                histories[name]["log_utilities"].append(log_utility)

            # Move users for next round
            for user in users:
                user.move_around_center()

        # Compute cumulative regrets and per-user cumulative throughputs
        for name, algo in algorithms.items():
            cum_reg = np.cumsum(histories[name]["regrets"])
            histories[name]["cum_regrets"] = np.concatenate([[0], cum_reg])
            histories[name]["cumulative_per_user"] = algo.cumulative_throughputs.copy()
            histories[name]["per_user_throughputs"] = np.array(
                histories[name]["per_user_throughputs"]
            )

        if verbose:
            print("Simulation complete!")
            for name in algorithms:
                final_reg = histories[name]["cum_regrets"][-1]
                print(f"  {name}: cumulative regret = {final_reg:.2f}")

        return {
            "histories": histories,
            "T": T,
            "target_throughput": self.target_throughput,
        }

    def run_multiple(
        self,
        algorithm_factory: Callable[[], Dict[str, CombinatorialAlgorithm]],
        n_experiments: int,
        T: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run multiple experiments and aggregate results."""
        sample_algos = algorithm_factory()
        algo_names = list(sample_algos.keys())

        all_cum_regrets = {name: [] for name in algo_names}
        all_jain_indices = {name: [] for name in algo_names}
        all_log_utilities = {name: [] for name in algo_names}
        all_throughputs = {name: [] for name in algo_names}
        all_ack_rates = {name: [] for name in algo_names}
        all_cumulative_per_user = {name: [] for name in algo_names}

        for exp in range(n_experiments):
            print(f"\nExperiment {exp + 1}/{n_experiments}")

            algorithms = algorithm_factory()
            for algo in algorithms.values():
                algo.reset()

            result = self.run(algorithms, T, verbose=False, **kwargs)

            for name in algo_names:
                hist = result["histories"][name]
                all_cum_regrets[name].append(hist["cum_regrets"])
                all_jain_indices[name].append(hist["jain_indices"])
                all_log_utilities[name].append(hist["log_utilities"])
                all_throughputs[name].append(hist["throughputs"])
                all_ack_rates[name].append(hist["ack_rates"])
                all_cumulative_per_user[name].append(hist["cumulative_per_user"])

        # Aggregate
        aggregated = {}
        for name in algo_names:
            regrets = np.array(all_cum_regrets[name])
            jain_indices = np.array(all_jain_indices[name])
            log_utilities = np.array(all_log_utilities[name])
            throughputs = np.array(all_throughputs[name])
            ack_rates = np.array(all_ack_rates[name])
            cumulative_per_user = np.array(all_cumulative_per_user[name])

            aggregated[name] = {
                "mean": np.mean(regrets, axis=0),
                "std": np.std(regrets, axis=0),
                "jain_mean": np.mean(jain_indices, axis=0),
                "jain_std": np.std(jain_indices, axis=0),
                "log_utility_mean": np.mean(log_utilities, axis=0),
                "log_utility_std": np.std(log_utilities, axis=0),
                "throughput_mean": np.mean(throughputs, axis=0),
                "throughput_std": np.std(throughputs, axis=0),
                "ack_rate_mean": np.mean(ack_rates, axis=0),
                "ack_rate_std": np.std(ack_rates, axis=0),
                "cumulative_per_user_mean": np.mean(cumulative_per_user, axis=0),
                "cumulative_per_user_std": np.std(cumulative_per_user, axis=0),
            }

        return {
            "aggregated": aggregated,
            "n_experiments": n_experiments,
            "T": T,
            "target_throughput": self.target_throughput,
        }

    _color_cycle = ["#7cb9a0", "#e8a0a0", "#8db8d9", "#e8c48a", "#c4a0d9", "#b0b0b0"]
    _marker_cycle = ["s", "o", "^", "D", "v", "p"]
    _linestyle_cycle = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), "-"]

    _rc_defaults = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }

    def _get_style(self, names):
        """Return color/marker/linestyle dicts for the given algorithm names."""
        colors = {
            name: self._color_cycle[i % len(self._color_cycle)]
            for i, name in enumerate(names)
        }
        markers = {
            name: self._marker_cycle[i % len(self._marker_cycle)]
            for i, name in enumerate(names)
        }
        linestyles = {
            name: self._linestyle_cycle[i % len(self._linestyle_cycle)]
            for i, name in enumerate(names)
        }
        return colors, markers, linestyles

    def _style_ax(self, ax):
        """Apply shared axis styling."""
        ax.tick_params(direction="in", which="both")
        ax.grid(True, which="major", linestyle="--", linewidth=0.4, color="#cccccc")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot simulation results."""
        plt.rcParams.update(self._rc_defaults)
        histories = results["histories"]
        T = results["T"]
        colors, markers, linestyles = self._get_style(list(histories.keys()))
        marker_every = max(1, T // 12)

        fig, ax = plt.subplots(figsize=(5, 3.5))

        x = np.arange(0, T + 1)
        for name in histories:
            ax.plot(
                x,
                histories[name]["cum_regrets"],
                color=colors.get(name),
                label=name,
                linewidth=1.5,
                linestyle=linestyles.get(name, "--"),
                marker=markers.get(name, "s"),
                markevery=marker_every,
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.0,
            )

        ax.set_xlabel("Time Slot")
        ax.set_ylabel("Cumulative Satisficing Regret")
        ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=10)
        self._style_ax(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig

    def plot_fairness(self, results: Dict[str, Any], save_path: str = None):
        """Plot fairness metrics (Jain's index and log-utility over time).

        Parameters
        ----------
        results : dict
            Results dictionary from run() or run_multiple().
        save_path : str, optional
            Path to save the figure. Default is None.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.
        """
        plt.rcParams.update(self._rc_defaults)
        is_aggregated = "aggregated" in results
        T = results["T"]

        if is_aggregated:
            data = results["aggregated"]
            algo_names = list(data.keys())
        else:
            data = results["histories"]
            algo_names = list(data.keys())

        colors, markers, linestyles = self._get_style(algo_names)
        marker_every = max(1, T // 12)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Compute optimal fairness metrics using oracle (psi)
        optimal_per_user = np.max(self.psi * self.rate_set[None, None, :], axis=(1, 2))
        optimal_jain_per_round = []
        optimal_log_utility_per_round = []
        for t in range(1, T + 1):
            cumulative_optimal = optimal_per_user * t
            optimal_jain_per_round.append(compute_jain_index(cumulative_optimal))
            optimal_log_utility_per_round.append(
                np.sum(np.log(cumulative_optimal + 1e-10))
            )

        x = np.arange(1, T + 1)

        # Plot 1: Jain's Fairness Index over time
        ax1 = axes[0]
        for name in algo_names:
            color = colors.get(name)
            marker = markers.get(name, "s")
            ls = linestyles.get(name, "--")
            if is_aggregated:
                mean = data[name]["jain_mean"]
                std = data[name]["jain_std"]
                ax1.plot(
                    x,
                    mean,
                    color=color,
                    label=name,
                    linewidth=1.5,
                    linestyle=ls,
                    marker=marker,
                    markevery=marker_every,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                )
                ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            else:
                ax1.plot(
                    x,
                    data[name]["jain_indices"],
                    color=color,
                    label=name,
                    linewidth=1.5,
                    linestyle=ls,
                    marker=marker,
                    markevery=marker_every,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                )

        ax1.plot(x, optimal_jain_per_round, "k--", label="Optimal", linewidth=1.5)
        ax1.set_xlabel("Time Slot")
        ax1.set_ylabel("Jain's Fairness Index")
        ax1.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=10)
        self._style_ax(ax1)
        ax1.set_ylim(0, 1.05)

        # Plot 2: Log-Utility (Proportional Fairness) over time
        ax2 = axes[1]
        for name in algo_names:
            color = colors.get(name)
            marker = markers.get(name, "s")
            ls = linestyles.get(name, "--")
            if is_aggregated:
                mean = data[name]["log_utility_mean"]
                std = data[name]["log_utility_std"]
                ax2.plot(
                    x,
                    mean,
                    color=color,
                    label=name,
                    linewidth=1.5,
                    linestyle=ls,
                    marker=marker,
                    markevery=marker_every,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                )
                ax2.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            else:
                ax2.plot(
                    x,
                    data[name]["log_utilities"],
                    color=color,
                    label=name,
                    linewidth=1.5,
                    linestyle=ls,
                    marker=marker,
                    markevery=marker_every,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                )

        ax2.plot(
            x, optimal_log_utility_per_round, "k--", label="Optimal", linewidth=1.5
        )
        ax2.set_xlabel("Time Slot")
        ax2.set_ylabel("Sum of Log Utilities")
        ax2.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=10)
        self._style_ax(ax2)

        # Plot 3: Final cumulative throughput distribution per user
        ax3 = axes[2]
        x_users = np.arange(self.num_users)
        width = 0.8 / len(algo_names)

        for i, name in enumerate(algo_names):
            color = colors.get(name)
            if is_aggregated:
                cumulative = data[name]["cumulative_per_user_mean"]
                err = data[name]["cumulative_per_user_std"]
                ax3.bar(
                    x_users + i * width,
                    cumulative,
                    width,
                    label=name,
                    color=color,
                    yerr=err,
                    capsize=2,
                    edgecolor="black",
                    linewidth=0.5,
                )
            else:
                cumulative = data[name]["cumulative_per_user"]
                ax3.bar(
                    x_users + i * width,
                    cumulative,
                    width,
                    label=name,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax3.set_xlabel("User")
        ax3.set_ylabel("Cumulative Throughput")
        ax3.set_xticks(x_users + width * (len(algo_names) - 1) / 2)
        ax3.set_xticklabels([f"U{i}" for i in range(self.num_users)])
        ax3.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=10)
        self._style_ax(ax3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig

    def print_fairness_summary(self, results: Dict[str, Any]):
        """Print a summary of fairness metrics.

        Parameters
        ----------
        results : dict
            Results dictionary from run().
        """
        histories = results["histories"]
        _ = results["T"]

        print("\n" + "=" * 60)
        print("FAIRNESS SUMMARY")
        print("=" * 60)

        for name in histories:
            jain_indices = histories[name]["jain_indices"]
            cumulative = histories[name]["cumulative_per_user"]

            print(f"\n{name}:")
            print(f"  Final Jain's Index: {jain_indices[-1]:.4f}")
            print(f"  Mean Jain's Index:  {np.mean(jain_indices):.4f}")
            print(f"  Total Throughput:   {np.sum(cumulative):.2f}")
            print(f"  Per-user cumulative: {cumulative}")
            print(f"  Std of cumulative:   {np.std(cumulative):.2f}")

    def plot_aggregated_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot aggregated results from multiple experiments with error bars.

        Parameters
        ----------
        results : dict
            Results dictionary from run_multiple().
        save_path : str, optional
            Path to save the figure. Default is None.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.
        """
        plt.rcParams.update(self._rc_defaults)
        aggregated = results["aggregated"]
        T = results["T"]
        colors, markers, linestyles = self._get_style(list(aggregated.keys()))
        marker_every = max(1, T // 12)

        fig, ax = plt.subplots(figsize=(5, 3.5))

        x = np.arange(0, T + 1)

        for name in aggregated:
            mean = aggregated[name]["mean"]
            std = aggregated[name]["std"]
            color = colors.get(name)
            marker = markers.get(name, "s")

            ax.plot(
                x,
                mean,
                color=color,
                label=name,
                linewidth=1.4,
                linestyle=linestyles.get(name, "--"),
                marker=marker,
                markevery=marker_every,
                markersize=5,
                markerfacecolor="none",
                markeredgewidth=1.0,
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_xlabel("Time Slot")
        ax.set_ylabel("Cumulative Satisficing Regret")
        ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=10)
        self._style_ax(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig
