"""
Example script for running combinatorial bandit simulation.
Compares SAT-CTS, SAT-CTS-W (workshop), CTS, and CUCB.
15 users with fixed positions.
"""

import json
import os
from datetime import datetime

import numpy as np
from obs.algorithms.combinatorial import (
    CTSAgent,
    CUCBAgent,
    SATCTSUCBAgent,
    SATCTSv2SharedAgent,
)
from obs.environment.channel_providers.deep_mimo import DeepMIMOProvider
from obs.environment.environment import Environment
from obs.simulation.combinatorial_simulation import CombinatorialSimulation


def main():
    T = 10000
    n_experiments = 3
    channel_randomness = True
    channel_randomness_std = 3.0
    target_throughput = 8.0

    rate_set = np.array([6.0, 8.0, 12.0])
    BW_HZ = 50e6
    NF_DB = 5.0
    N0_DBM_PER_HZ = -174.0
    N_dBm = N0_DBM_PER_HZ + 10 * np.log10(BW_HZ) + NF_DB
    noise_var = 10 ** ((N_dBm - 30) / 10.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Setup environment
    print("[1] Setting up environment...")
    provider = DeepMIMOProvider(
        scenario="city_3_houston_28",
        num_bs=3,
        bs_antenna_shape=(64, 1),
        ue_antenna_shape=(1, 1),
        subcarriers=1,
        num_paths=1,
    )
    env = Environment(channel_provider=provider, K=120, p_set_dBm=50)
    total_beams = env.get_actions().shape[1]
    valid_pos = env.get_valid_positions(check_neighbors=False)
    print(f"  Total beams: {total_beams}, valid positions: {len(valid_pos)}")

    user_positions_15 = np.array(
        [
            [79.1288, 21.0108],
            [21.1288, 17.0108],
            [7.12878, 48.0108],
            [-76.8712, 53.0108],
            [63.1288, 71.0108],
            [98.1288, -66.9892],
            [7.12878, -5.98922],
            [81.1288, 57.0108],
            [-58.8712, 57.0108],
            [-16.8712, 2.01078],
            [48.1288, -70.9892],
            [-32.8712, -42.9892],
            [-70.8712, -59.9892],
            [20.1288, -58.9892],
            [-58.8712, 47.0108],
        ]
    )

    experiments = [
        (15, user_positions_15),
    ]

    for num_users, user_positions in experiments:
        print("\n" + "=" * 60)
        print(f"Running with {num_users} users")
        print("=" * 60)

        sim = CombinatorialSimulation(
            environment=env,
            user_positions=user_positions,
            rate_set=rate_set,
            noise_var=noise_var,
            target_throughput=target_throughput,
            eps_user=0.0,
            channel_randomness=channel_randomness,
            channel_randomness_std=channel_randomness_std,
        )

        def algorithm_factory(nu=num_users, tb=total_beams):
            return {
                "SAT-CTS": SATCTSv2SharedAgent(nu, tb, rate_set, target_throughput),
                "SAT-CTS-W": SATCTSUCBAgent(nu, tb, rate_set, target_throughput),
                "CTS": CTSAgent(nu, tb, rate_set),
                "CUCB": CUCBAgent(nu, tb, rate_set),
            }

        results = sim.run_multiple(
            algorithm_factory=algorithm_factory,
            n_experiments=n_experiments,
            T=T,
        )

        print(f"\nResults ({num_users} users):")
        save_data = {
            "config": {
                "T": T,
                "n_experiments": n_experiments,
                "num_users": num_users,
                "target_throughput": target_throughput,
                "rate_set": rate_set.tolist(),
                "noise_var": noise_var,
                "user_positions": user_positions.tolist(),
                "total_beams": total_beams,
            },
            "results": {},
        }

        for name in results["aggregated"]:
            agg = results["aggregated"][name]
            mean = agg["mean"][-1]
            std = agg["std"][-1]
            print(f"  {name}: {mean:.2f} +/- {std:.2f}")
            save_data["results"][name] = {
                "mean_cum_regret": agg["mean"].tolist(),
                "std_cum_regret": agg["std"].tolist(),
                "jain_mean": agg["jain_mean"].tolist(),
                "jain_std": agg["jain_std"].tolist(),
                "log_utility_mean": agg["log_utility_mean"].tolist(),
                "log_utility_std": agg["log_utility_std"].tolist(),
                "throughput_mean": agg["throughput_mean"].tolist(),
                "throughput_std": agg["throughput_std"].tolist(),
                "final_mean": float(mean),
                "final_std": float(std),
            }

        out_base = os.path.join(
            results_dir, f"combinatorial_{num_users}users_{timestamp}"
        )
        with open(f"{out_base}.json", "w") as f:
            json.dump(save_data, f, indent=2)
        sim.plot_aggregated_results(
            results,
            save_path=f"{out_base}_regret.png",
        )
        sim.plot_fairness(
            results,
            save_path=f"{out_base}_fairness.png",
        )

    print("\nAll done!")


if __name__ == "__main__":
    main()
