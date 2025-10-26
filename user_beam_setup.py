import math
import numpy as np
import scipy.io
import torch
import os
from typing import List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class BeamConfig:
    """Common parameters shared by a set of users / experiments."""

    num_base_stations: int = 3  # How many BSs exist in the scenario
    arms_per_bs: int = 120  # Arms per BS (K per BS)
    N: int = 64  # Antenna elements in ULA
    p_set: float = 50  # Transmit power in dBm
    satisficing_rank: int = 4  # Rank for satisficing threshold
    base_path: str = "./"  # Where channel files live (current directory)


def get_default_test_configurations() -> List[BeamConfig]:
    """Return a list of baseline configurations (mimics notebook helper)."""
    return [
        BeamConfig(num_base_stations=3, arms_per_bs=120, N=64, p_set=50, satisficing_rank=4),
        # Add further variations here if needed
    ]


def f_codebook(K, N):
    """Generate codebook for K arms with N elements each"""
    codebook = []
    num = torch.arange(N, dtype=torch.float64)
    for k in range(K):
        x = 1j * math.pi * (-1 + (2 * k) / K)
        exp_vector = torch.exp(x * num)
        f_k = (1 / math.sqrt(N)) * exp_vector.to(torch.cdouble)
        codebook.append(f_k)
    return codebook


def load_h_file(base_station_id: int, user_id: int, base_path: str = "./") -> torch.Tensor:
    filename = f"h_U{user_id + 1}_B{base_station_id + 1}.mat"
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Channel file not found: {filepath}")

    mat_data = scipy.io.loadmat(filepath)

    # Assuming new files keep the Quadriga key
    if "h_quadriga" in mat_data:
        h = mat_data["h_quadriga"]
    elif "h" in mat_data:
        h = mat_data["h"]
    elif f"h_U{user_id + 1}_B{base_station_id + 1}" in mat_data:
        h = mat_data[f"h_U{user_id + 1}_B{base_station_id + 1}"]
    elif f"unnamed" in mat_data:
        h = mat_data[f"unnamed"]
    elif f"unnamed1" in mat_data:
        h = mat_data[f"unnamed1"]
    else:
        raise KeyError(f"No 'h_quadriga' or 'h' variable in {filename}")

    # Ensure column vector
    if h.shape[1] != 1:
        h = h.T

    return torch.tensor(h, dtype=torch.cdouble)


class User:
    def __init__(self, user_id: int, config: BeamConfig):
        """
        Create a user with ID and initialise beam/ reward information.
        The user is assumed to have access to ALL beams from ALL base-stations.

        Args:
            user_id:           Unique identifier of the user.
            num_base_stations: Total number of base stations in the scenario.
            arms_per_bs:       Number of beamforming arms per base station (default: 120).
        """
        self.user_id = user_id
        self.config = config
        self.num_base_stations = config.num_base_stations
        self.arms_per_bs = config.arms_per_bs

        # Total number of arms across all base-stations
        self.total_arms = self.num_base_stations * self.arms_per_bs

        # Beam array as a torch tensor of indices (int64)
        self.beam_array = torch.arange(self.total_arms, dtype=torch.long)

        self._bs_codebooks: List[List[torch.Tensor]] = [
            f_codebook(self.arms_per_bs, self.config.N) for _ in range(self.num_base_stations)
        ]

        self.arm_vectors: torch.Tensor = torch.stack(
            [vec for cb in self._bs_codebooks for vec in cb], dim=0
        )  # complex128 (cdouble)

        # Load channel files once at initialization (avoids repeated disk I/O)
        self._channel_files = {}
        for bs_id in range(self.num_base_stations):
            h = load_h_file(bs_id, self.user_id, self.config.base_path)
            self._channel_files[bs_id] = h

    def get_beam_array(self) -> torch.Tensor:
        """Return tensor of global arm indices"""
        return self.beam_array

    def get_bs_and_local_index(self, global_arm_idx: int) -> tuple[int, int]:
        """Return (base_station_id, local_arm_idx) for a given global arm index."""
        if global_arm_idx < 0 or global_arm_idx >= self.total_arms:
            raise ValueError(f"Global arm index {global_arm_idx} out of valid range 0-{self.total_arms - 1}")

        bs_id = global_arm_idx // self.arms_per_bs
        local_idx = global_arm_idx % self.arms_per_bs
        return (bs_id, local_idx)

    def compute_beam_rewards(self, time_slot: int = 0) -> torch.Tensor:
        """
        Compute rewards for all beams in this user's beam array.

        Args:
            time_slot: Time slot for generating time-varying channels

        Returns:
            Reward vector for all beams across all base stations
        """
        return self._compute_rewards_for_timeslot(time_slot)

    def _compute_rewards_for_timeslot(self, time_slot: int) -> torch.Tensor:
        """Internal method to compute rewards for a specific time slot"""
        # Pre-allocate reward list
        rewards: List[float] = []

        # Signal and noise setup (same as notebook)
        P = 10 ** (self.config.p_set / 10) / 1000  # Convert dBm to Watts
        sqrt_P = torch.sqrt(torch.tensor(P, dtype=torch.cdouble))

        # Seed for reproducible index shifts
        torch.manual_seed(time_slot + self.user_id * 1000)

        # Iterate over each base-station, use channel file from memory, compute its 120 rewards
        for bs_id in range(self.num_base_stations):
            # Use channel file from memory instead of loading from disk
            h = self._channel_files[bs_id]

            # Generate Gaussian random shift for this BS (mean=0, std=3)
            index_shift = int(torch.randn(1).item() * 3)

            bs_rewards = []
            for local_idx in range(self.arms_per_bs):
                # Apply circular shift to beam index
                shifted_idx = (local_idx + index_shift) % self.arms_per_bs
                f_k = self._bs_codebooks[bs_id][shifted_idx].reshape(-1, 1)

                y = sqrt_P * h.conj().T @ f_k
                r = y.conj().T @ y
                bs_rewards.append(abs(r).item())

            rewards.extend(bs_rewards)

        return torch.tensor(rewards, dtype=torch.double)


def create_users(num_users: int, config: BeamConfig) -> List[User]:
    """
    Create users and assign each to a base station.

    Args:
        num_users: Number of users
        num_base_stations: Number of base stations
        arms_per_bs: Arms per base station

    Returns:
        List of User objects
    """
    users = []
    for user_id in range(num_users):
        user = User(user_id, config)
        users.append(user)
    return users
