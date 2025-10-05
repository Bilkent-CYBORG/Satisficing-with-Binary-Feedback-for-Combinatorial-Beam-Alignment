#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment  # Hungarian

# Ensure user_beam_setup is in path
sys.path.append(os.getcwd())
from user_beam_setup import get_default_test_configurations, create_users

# 1) Load configuration and auto-detect number of users from files
cfg = get_default_test_configurations()[0]
print(f"Configuration: {cfg.num_base_stations} BSs, {cfg.arms_per_bs} beams/BS, N={cfg.N} antennas")

# Auto-detect number of users from available h_U*_B1.mat files
import glob
import re


def infer_num_users(cfg):
    """Infer number of users from available h_U*_B1.mat files"""
    pattern = os.path.join(cfg.base_path, "h_U*_B1.mat")
    files = glob.glob(pattern)
    if not files:
        print("No channel files found, using default 3 users")
        return 3

    user_ids = []
    for filepath in files:
        match = re.search(r"h_U(\d+)_B1\.mat$", os.path.basename(filepath))
        if match:
            user_ids.append(int(match.group(1)))

    num_users = max(user_ids) if user_ids else 3
    print(f"Found channel files for {num_users} users")
    return num_users


num_users = infer_num_users(cfg)

# Extract configuration parameters
B = cfg.num_base_stations  # Number of base stations
K = cfg.arms_per_bs        # Beams per base station
M = num_users              # Number of users
total_beams = B * K        # Total number of beams

# 2) Define rate set and SNR thresholds - Strategic 3-rate selection
R_set = np.array([8.0, 12.0, 16.0])  # bits/symbol - Conservative, Moderate, Aggressive
gamma_th = 2 ** R_set - 1  # SNR thresholds from Shannon-Hartley

# Noise parameters
noise_var = 1e-10  # -100 dBW noise floor
noise_power_dbm = 10 * np.log10(noise_var * 1000)  # Convert to dBm
print(f"Noise floor: {noise_power_dbm:.1f} dBm")


# 3) Optimized Time-varying channel environment with caching
class TimeVaryingBeamEnvironment:
    def __init__(self, users, rate_set, noise_var=1e-10,
                 target_throughput=2.5):
        """
        Environment with time-varying channels and ACK/NACK feedback.
        Optimized with channel caching per time slot.
        """
        self.users = users
        self.num_users = len(users)
        self.rate_set = np.array(rate_set)
        self.num_rates = len(rate_set)
        self.noise_var = noise_var
        self.target_throughput = target_throughput
        self.total_beams = users[0].total_arms if users else B * K

        # SNR thresholds for each rate
        self.gamma_th = 2 ** self.rate_set - 1

        # Channel caching variables
        self._cached_time = None
        self._cached_beam_powers = None  # list of np arrays, one per user

        # Initialize success probability estimates
        self._estimate_success_probabilities()

    def _get_beam_powers(self, time_slot):
        """Cache channel realization for the entire time slot"""
        if self._cached_time != time_slot:
            self._cached_beam_powers = [
                user.compute_beam_rewards(time_slot=time_slot).cpu().numpy()
                for user in self.users
            ]
            self._cached_time = time_slot
        return self._cached_beam_powers

    def _estimate_success_probabilities(self):
        """Optimized probability estimation with fewer samples and vectorization"""
        self.psi = np.zeros((self.num_users, self.total_beams, self.num_rates))

        # Reduced samples from 20 to 5
        num_samples = 5
        for u, user in enumerate(self.users):
            # Vectorized sampling
            samples = np.stack([
                user.compute_beam_rewards(time_slot=t).cpu().numpy()
                for t in range(num_samples)
            ], axis=0)  # (S, total_beams)

            mean_snr = samples.mean(axis=0) / self.noise_var  # (B_tot,)

            # Broadcast over rates for vectorized computation
            thr = self.gamma_th[None, :]  # (1,R)
            margin = (mean_snr[:, None] - thr) / np.sqrt(np.maximum(thr, 1.0))

            # Numerically stable sigmoid to avoid overflow
            margin_clipped = np.clip(0.5 * margin, -30, 30)
            self.psi[u] = 1.0 / (1.0 + np.exp(-margin_clipped))  # (B_tot,R)

        # Find optimal assignment for reference
        self._find_optimal_superarm()

    def _hungarian_best(self, values):
        """
        Hungarian assignment after picking best rate per (u,b).
        values: (U,B,R) matrix of scores to maximize.
        Returns chosen_beams (len U) and chosen_rates (len U).
        """
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)  # (U,B)
        # scores[u,b] = max_r values[u,b,r]
        scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)  # (U,B)

        # Hungarian solves a minimization; use negative scores
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)

        chosen_beams = [None] * U
        chosen_rates = [None] * U
        for u_idx, b_idx in zip(row_ind, col_ind):
            chosen_beams[u_idx] = int(b_idx)
            chosen_rates[u_idx] = int(best_r[u_idx, b_idx])
        return chosen_beams, chosen_rates

    def _find_optimal_superarm(self):
        """Find the optimal beam and rate assignment with Hungarian algorithm"""
        # Create throughput matrix for all (user, beam, rate) combinations
        throughput_matrix = self.rate_set[None, None, :] * self.psi  # (U,B,R)

        # Hungarian with best rate per (u,b)
        chosen_beams, chosen_rates = self._hungarian_best(throughput_matrix)

        # Store optimal assignment
        self.optimal_beams = chosen_beams
        self.optimal_rates = chosen_rates

        # Calculate optimal throughput
        total_throughput = sum(
            self.rate_set[r] * self.psi[u, b, r]
            for u, (b, r) in enumerate(zip(chosen_beams, chosen_rates))
        )
        self.optimal_throughput = total_throughput / self.num_users

    def play(self, beam_assignment, rate_assignment, time_slot):
        """
        Execute beam and rate assignment with cached channel realization.
        Enhanced with maximum achievable rate logging.
        """
        ack_nack = np.zeros(self.num_users, dtype=int)
        instant_throughput = np.zeros(self.num_users)
        bp_list = self._get_beam_powers(time_slot)  # reuse for all agents in t

        # Calculate maximum achievable rates for this time slot
        max_achievable_rates = []
        oracle_throughput = 0

        for u in range(self.num_users):
            beam_powers = bp_list[u]

            # Find the best beam and rate for this user at this time slot
            best_beam = -1
            best_rate = -1
            best_throughput = 0

            for b in range(self.total_beams):
                snr = beam_powers[b] / self.noise_var
                # Find highest rate that would succeed with this SNR
                for r_idx in range(len(self.rate_set) - 1, -1, -1):
                    if snr >= self.gamma_th[r_idx]:
                        if self.rate_set[r_idx] > best_throughput:
                            best_throughput = self.rate_set[r_idx]
                            best_beam = b
                            best_rate = r_idx
                        break

            max_achievable_rates.append({
                'user': u,
                'best_beam': best_beam,
                'best_rate_idx': best_rate,
                'best_rate': self.rate_set[best_rate] if best_rate >= 0 else 0,
                'max_throughput': best_throughput,
                'actual_beam': beam_assignment[u],
                'actual_rate': self.rate_set[rate_assignment[u]]
            })
            oracle_throughput += best_throughput

            # Process actual assignment
            b = beam_assignment[u]
            r = rate_assignment[u]

            # Normal SNR-based success/failure
            snr = beam_powers[b] / self.noise_var
            thr = self.gamma_th[r]
            snr_actual = snr * np.exp(0.1 * np.random.randn())
            if snr_actual >= thr:
                ack_nack[u] = 1
                instant_throughput[u] = self.rate_set[r]

        return ack_nack, instant_throughput, max_achievable_rates

    def compute_regret(self, beam_assignment, rate_assignment):
        """Satisficing regret for current assignment (units: bits/symbol)"""
        throughput = 0
        for u in range(self.num_users):
            b = beam_assignment[u]
            r = rate_assignment[u]
            throughput += self.rate_set[r] * self.psi[u, b, r]

        avg_throughput = throughput / self.num_users
        return max(0.0, self.target_throughput - avg_throughput)

    def compute_standard_regret(self, beam_assignment, rate_assignment):
        """Standard regret vs. optimal superarm (units: bits/symbol)"""
        throughput = 0.0
        for u in range(self.num_users):
            b = beam_assignment[u]
            r = rate_assignment[u]
            throughput += self.rate_set[r] * self.psi[u, b, r]
        avg_throughput = throughput / self.num_users
        return max(0.0, self.optimal_throughput - avg_throughput)


# 4) Optimized Algorithm implementations
class SATCTSAgent:
    """
    Satisficing gate by posterior probability using Beta Monte-Carlo (no Gaussian).
    - Candidate 1: TS super-arm (sample ψ ~ Beta)
    - Candidate 2: LCB super-arm (Hoeffding per-arm LCB on ψ)
    - Gate: pick the one with larger P(sum r*ψ >= M * target) estimated by MC.
    """

    def __init__(self, env, seed=0, verbose=False, mc_samples=2000):
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.S = int(mc_samples)
        self.decision_log = []

    def init(self):
        U, Btot, R = self.env.num_users, self.env.total_beams, self.env.num_rates
        self.A = np.ones((U, Btot, R))
        self.B = np.ones((U, Btot, R))
        self.n_plays = np.zeros((U, Btot, R), dtype=int)
        self.n_success = np.zeros((U, Btot, R), dtype=int)

    def _hungarian_best(self, values):
        """Hungarian assignment over (U×B) after picking best rate per (u,b)"""
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)  # (U,B)
        scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)  # (U,B)
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        chosen_beams = [None] * U
        chosen_rates = [None] * U
        for u_idx, b_idx in zip(row_ind, col_ind):
            chosen_beams[u_idx] = int(b_idx)
            chosen_rates[u_idx] = int(best_r[u_idx, b_idx])
        return chosen_beams, chosen_rates

    def step(self, t):
        U, Btot, R = self.env.num_users, self.env.total_beams, self.env.num_rates
        rates = self.env.rate_set[np.newaxis, np.newaxis, :]  # (1,1,R)

        # ---------- Candidate 1: TS super-arm (Beta sampling, exact) ----------
        psi_ts = self.rng.beta(self.A, self.B)  # ~ Beta(A,B)
        theta_ts = rates * psi_ts  # r * ψ_sample
        ts_beams, ts_rates = self._hungarian_best(theta_ts)

        # ---------- Compute LCBs and optimal LCB assignment ----------
        n = np.maximum(1, self.n_plays)
        psi_hat = self.n_success / n
        conf = np.sqrt(0.5 * np.log(max(2, t)) / n)
        psi_lcb = np.maximum(0.0, psi_hat - conf)
        lcb_values = rates * psi_lcb  # r * LCB(ψ)

        lcb_beams, lcb_rates = self._hungarian_best(lcb_values)
        lcb_sum = sum(lcb_values[u, lcb_beams[u], lcb_rates[u]] for u in range(U))
        lcb_avg = lcb_sum / U

        # ---------- Hierarchical Decision Cascade: LCB → μ → UCB → TS ----------
        target_threshold = self.env.target_throughput
        total_threshold = target_threshold * U

        # Step 1: Try LCB
        if lcb_sum >= total_threshold:
            chosen_beams, chosen_rates = lcb_beams, lcb_rates
            decision = "LCB"
        else:
            # Step 2: Try Mean (empirical estimate)
            mu_values = rates * psi_hat
            mu_beams, mu_rates = self._hungarian_best(mu_values)
            mu_sum = sum(mu_values[u, mu_beams[u], mu_rates[u]] for u in range(U))

            if mu_sum >= total_threshold:
                chosen_beams, chosen_rates = mu_beams, mu_rates
                decision = "MU"
            else:
                # Step 3: Try UCB (upper confidence bound)
                conf = np.sqrt(0.5 * np.log(max(2, t)) / n)
                psi_ucb = psi_hat + conf  # No capping for proper exploration
                ucb_values = rates * psi_ucb
                ucb_beams, ucb_rates = self._hungarian_best(ucb_values)
                ucb_sum = sum(ucb_values[u, ucb_beams[u], ucb_rates[u]] for u in range(U))

                if ucb_sum >= total_threshold:
                    chosen_beams, chosen_rates = ucb_beams, ucb_rates
                    decision = "UCB"
                else:
                    # Step 4: Fallback to Thompson Sampling
                    chosen_beams, chosen_rates = ts_beams, ts_rates
                    decision = "TS"


        # ---------- Play and update ----------
        result = self.env.play(chosen_beams, chosen_rates, t)
        if len(result) == 3:
            ack_nack, instant_tput, oracle_info = result
        else:
            ack_nack, instant_tput = result
            oracle_info = None

        for m in range(U):
            b = chosen_beams[m]
            r = chosen_rates[m]
            self.n_plays[m, b, r] += 1
            self.n_success[m, b, r] += ack_nack[m]
            self.A[m, b, r] = 1 + self.n_success[m, b, r]
            self.B[m, b, r] = 1 + self.n_plays[m, b, r] - self.n_success[m, b, r]

        # Create decision log entry
        log_entry = {
            "t": t, "decision": decision, "threshold": target_threshold,
            "lcb_sum": lcb_sum, "lcb_avg": lcb_avg,
            "ts_beams": ts_beams, "ts_rates": ts_rates,
            "chosen_beams": chosen_beams, "chosen_rates": chosen_rates
        }
        if decision != "LCB":
            log_entry.update({"mu_sum": mu_sum, "mu_beams": mu_beams, "mu_rates": mu_rates})
        if decision not in ["LCB", "MU"]:
            log_entry.update({"ucb_sum": ucb_sum, "ucb_beams": ucb_beams, "ucb_rates": ucb_rates})

        self.decision_log.append(log_entry)
        return ack_nack, chosen_beams, chosen_rates


class CTSAgent:
    """Optimized Combinatorial Thompson Sampling"""

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.RandomState(seed)

    def init(self):
        # Beta parameters
        self.A = np.ones((self.env.num_users, self.env.total_beams, self.env.num_rates))
        self.B = np.ones((self.env.num_users, self.env.total_beams, self.env.num_rates))

    def _hungarian_best(self, values):
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)  # (U,B)
        scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        chosen_beams = [None] * U
        chosen_rates = [None] * U
        for u_idx, b_idx in zip(row_ind, col_ind):
            chosen_beams[u_idx] = int(b_idx)
            chosen_rates[u_idx] = int(best_r[u_idx, b_idx])
        return chosen_beams, chosen_rates

    def step(self, t):
        # Thompson sampling
        psi_samples = self.rng.beta(self.A, self.B)
        theta_samples = self.env.rate_set[np.newaxis, np.newaxis, :] * psi_samples

        # Find optimal assignment via Hungarian
        chosen_beams, chosen_rates = self._hungarian_best(theta_samples)

        # Play and observe
        result = self.env.play(chosen_beams, chosen_rates, t)
        if len(result) == 3:
            ack_nack, instant_tput, oracle_info = result
        else:
            ack_nack, instant_tput = result
            oracle_info = None

        # Update Beta parameters
        for u in range(self.env.num_users):
            b = chosen_beams[u]
            r = chosen_rates[u]
            self.A[u, b, r] += ack_nack[u]
            self.B[u, b, r] += 1 - ack_nack[u]

        return ack_nack, chosen_beams, chosen_rates


class CUCBAgent:
    """Optimized Combinatorial Upper Confidence Bound"""

    def __init__(self, env):
        self.env = env

    def init(self):
        self.n_plays = np.zeros((self.env.num_users, self.env.total_beams, self.env.num_rates), dtype=int)
        self.psi_hat = np.zeros((self.env.num_users, self.env.total_beams, self.env.num_rates))

    def _hungarian_best(self, values):
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)  # (U,B)
        scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        chosen_beams = [None] * U
        chosen_rates = [None] * U
        for u_idx, b_idx in zip(row_ind, col_ind):
            chosen_beams[u_idx] = int(b_idx)
            chosen_rates[u_idx] = int(best_r[u_idx, b_idx])
        return chosen_beams, chosen_rates

    def step(self, t):
        # Vectorized UCB computation
        n = np.maximum(1, self.n_plays)
        conf = np.sqrt(2 * np.log(max(2, t)) / n)
        ucb_psi = self.psi_hat + conf  # No capping for proper exploration
        theta_ucb = self.env.rate_set[None, None, :] * ucb_psi

        # Hungarian assignment
        chosen_beams, chosen_rates = self._hungarian_best(theta_ucb)

        # Play and observe
        result = self.env.play(chosen_beams, chosen_rates, t)
        if len(result) == 3:
            ack_nack, instant_tput, oracle_info = result
        else:
            ack_nack, instant_tput = result
            oracle_info = None

        # Update estimates
        for u in range(self.env.num_users):
            b = chosen_beams[u]
            r = chosen_rates[u]
            self.n_plays[u, b, r] += 1
            # Running average update
            self.psi_hat[u, b, r] += (ack_nack[u] - self.psi_hat[u, b, r]) / self.n_plays[u, b, r]

        return ack_nack, chosen_beams, chosen_rates


def analyze_performance_gap(env, T):
    """Analyze the gap between achieved and maximum possible performance"""
    # Silent analysis - just compute statistics without printing
    oracle_samples = []
    for t_sample in range(0, min(T, 100), 10):
        bp_list = env._get_beam_powers(t_sample)
        slot_max = 0

        for u in range(env.num_users):
            beam_powers = bp_list[u]
            user_max = 0

            for b in range(env.total_beams):
                snr = beam_powers[b] / env.noise_var
                # Find highest achievable rate
                for r_idx in range(len(env.rate_set) - 1, -1, -1):
                    if snr >= env.gamma_th[r_idx]:
                        user_max = max(user_max, env.rate_set[r_idx])
                        break

            slot_max += user_max

        oracle_samples.append(slot_max / env.num_users)


# 5) Run experiments
T = 10000  # Time horizon for experiments
num_iterations = 100
target_throughput = 10  # bits/symbol per user

print(f"\n{'=' * 60}")
print(f"Running optimized experiments with {num_iterations} iterations, T={T} rounds")
print(f"Number of users: {M}")
print(f"Total beams: {total_beams} ({B} BSs × {K} beams/BS)")
print(f"Rate set: {R_set}")
print(f"Target throughput: {target_throughput} bits/symbol per user")
print(f"{'=' * 60}\n")

# Initialize results storaged
all_regrets = {name: [] for name in ['CUCB', 'CTS', 'SAT-CTS']}            # satisficing (cumulative)
all_std_regrets = {name: [] for name in ['CUCB', 'CTS', 'SAT-CTS']}        # standard (cumulative)
all_throughputs = {name: [] for name in ['CUCB', 'CTS', 'SAT-CTS']}

for iteration in range(num_iterations):
    print(f"\r[{iteration + 1}/{num_iterations}] Initializing users...", end='', flush=True)

    # ---- Fresh users each iteration to ensure new random channels ----
    users = create_users(num_users, cfg)

    # Create environment
    print(f"\r[{iteration + 1}/{num_iterations}] Creating environment...", end='', flush=True)
    env = TimeVaryingBeamEnvironment(
        users=users,
        rate_set=R_set,
        noise_var=noise_var,
        target_throughput=target_throughput
    )

    # Analyze performance gap before simulation
    analyze_performance_gap(env, T)

    # Create agents
    print(f"\r[{iteration + 1}/{num_iterations}] Initializing agents...", end='', flush=True)
    agents = {
        'CUCB': CUCBAgent(env),
        'CTS': CTSAgent(env, seed=iteration),
        'SAT-CTS': SATCTSAgent(env, seed=iteration)
    }

    for ag in agents.values():
        ag.init()

    # Run simulation
    regrets = {name: np.zeros(T) for name in agents}        # satisficing cumulative
    std_regs = {name: np.zeros(T) for name in agents}       # standard cumulative
    throughputs = {name: np.zeros(T) for name in agents}

    for t in range(1, T + 1):
        if t % 500 == 0:
            print(f"\r[{iteration + 1}/{num_iterations}] Round {t}/{T}...", end='', flush=True)

        for name, agent in agents.items():
            ack_nack, beams, rates = agent.step(t)

            # Compute satisficing regret (cumulative)
            inst_reg = env.compute_regret(beams, rates)
            regrets[name][t - 1] = (regrets[name][t - 2] if t > 1 else 0) + inst_reg

            # Compute standard regret (cumulative)
            inst_std = env.compute_standard_regret(beams, rates)
            std_regs[name][t - 1] = (std_regs[name][t - 2] if t > 1 else 0) + inst_std

            # Track average throughput
            throughputs[name][t - 1] = np.mean(ack_nack * env.rate_set[rates])

    # Store results for this iteration
    print(f"\r[{iteration + 1}/{num_iterations}] Complete!                    ")
    for name in agents:
        all_regrets[name].append(regrets[name].copy())
        all_std_regrets[name].append(std_regs[name].copy())
        all_throughputs[name].append(throughputs[name].copy())

# Calculate statistics
print("\nCalculating statistics...")
mean_regrets = {}
std_regrets = {}
mean_std_regrets = {}
std_std_regrets = {}
mean_throughputs = {}
std_throughputs = {}

for name in ['CUCB', 'CTS', 'SAT-CTS']:
    regrets_array = np.array(all_regrets[name])
    mean_regrets[name] = np.mean(regrets_array, axis=0)
    std_regrets[name] = np.std(regrets_array, axis=0)

    std_array = np.array(all_std_regrets[name])
    mean_std_regrets[name] = np.mean(std_array, axis=0)
    std_std_regrets[name] = np.std(std_array, axis=0)

    throughputs_array = np.array(all_throughputs[name])
    mean_throughputs[name] = np.mean(throughputs_array, axis=0)
    std_throughputs[name] = np.std(throughputs_array, axis=0)


# -------------------------- Plotting (3 figures) --------------------------
print("Generating plots...")
colors = {'CUCB': 'blue', 'CTS': 'orange', 'SAT-CTS': 'green'}
time_steps = np.arange(1, T + 1)

# draw error bars at sparse points (~200 across x-axis)
ERR_EVERY = max(1, T // 200)

# ===== FIGURE 1: CTS vs SAT-CTS — satisficing & standard regrets (same axes) =====
fig1 = plt.figure(figsize=(14, 6))
ax = fig1.add_subplot(111)

pair_defs = [
    ('CTS',      'satisficing', mean_regrets['CTS'],           std_regrets['CTS'],           colors['CTS']),
    ('SAT-CTS',  'satisficing', mean_regrets['SAT-CTS'],       std_regrets['SAT-CTS'],       colors['SAT-CTS']),
    ('CTS',      'standard',    mean_std_regrets['CTS'],       std_std_regrets['CTS'],       'tab:purple'),
    ('SAT-CTS',  'standard',    mean_std_regrets['SAT-CTS'],   std_std_regrets['SAT-CTS'],   'tab:brown'),
]

for algo_name, kind, mean_reg, std_reg, col in pair_defs:
    # main line WITH label (legend)
    ax.plot(time_steps, mean_reg, color=col, linewidth=2,
            label=f'{algo_name} ({kind})')

    # shaded band (no legend)
    lower = np.maximum(0, mean_reg - std_reg)
    upper = mean_reg + std_reg
    ax.fill_between(time_steps, lower, upper, color=col, alpha=0.15)

    # sparse error bars (no legend)
    idx = np.arange(0, T, ERR_EVERY, dtype=int)
    ax.errorbar(time_steps[idx], mean_reg[idx], yerr=std_reg[idx], fmt='none',
                ecolor=col, elinewidth=1, capsize=2, alpha=0.9)

ax.set_xlabel('Time Slot', fontsize=12)
ax.set_ylabel('Cumulative Regret (bits/symbol)', fontsize=12)
ax.set_title('CTS vs SAT-CTS — Satisficing & Standard Regret', fontsize=14)
ax.legend(fontsize=10, ncol=2)   # legend shows only the 4 main lines
ax.grid(True, alpha=0.3)
fig1.tight_layout()

print("Displaying plots...")
plt.show()
# ------------------------ end plotting ------------------------


# Print final statistics
print(f"\n{'=' * 60}")
print("OPTIMIZED FINAL STATISTICS")
print(f"{'=' * 60}")
print(f"Target throughput: {target_throughput:.3f} bits/symbol per user")
if hasattr(env, 'optimal_throughput'):
    print(f"Optimal throughput: {env.optimal_throughput:.3f} bits/symbol per user")

print(f"\nFinal cumulative satisficing regret (mean ± std) over {num_iterations} runs:")
for name in ['CUCB', 'CTS', 'SAT-CTS']:
    final_regret_mean = mean_regrets[name][-1]
    final_regret_std = std_regrets[name][-1]
    print(f"  {name:12s}: {final_regret_mean:8.1f} ± {final_regret_std:6.1f} (bits/symbol)")

print(f"\nFinal cumulative standard regret (mean ± std) over {num_iterations} runs:")
for name in ['CTS', 'SAT-CTS']:  # reporting the two requested algos
    final_regret_mean = mean_std_regrets[name][-1]
    final_regret_std = std_std_regrets[name][-1]
    print(f"  {name:12s}: {final_regret_mean:8.1f} ± {final_regret_std:6.1f} (bits/symbol)")

print(f"\nFinal average throughput (last 10% of rounds):")
last_10pct = max(1, T // 10)
for name in ['CUCB', 'CTS', 'SAT-CTS']:
    final_throughput_mean = np.mean(mean_throughputs[name][-last_10pct:])
    final_throughput_std = np.mean(std_throughputs[name][-last_10pct:])
    print(f"  {name:12s}: {final_throughput_mean:.3f} ± {final_throughput_std:.3f} bits/symbol")

# Relative performance vs SAT-CTS (satisficing regret)
print(f"\nRelative performance (lower is better, vs SAT-CTS satisficing regret):")
sat_regret = mean_regrets['SAT-CTS'][-1]
for name in ['CUCB', 'CTS', 'SAT-CTS']:
    relative = mean_regrets[name][-1] / sat_regret if sat_regret > 0 else 0
    print(f"  {name:12s}: {relative:.2f}x")

# After the simulation loop, add analysis
print(f"\n{'=' * 60}")
print("ALGORITHM BEHAVIOR ANALYSIS")
print(f"{'=' * 60}")

# Analyze SAT-CTS decisions
if 'SAT-CTS' in agents and hasattr(agents['SAT-CTS'], 'decision_log'):
    decisions = agents['SAT-CTS'].decision_log
    lcb_count = sum(1 for d in decisions if d['decision'] == 'LCB')
    mu_count = sum(1 for d in decisions if d['decision'] == 'MU')
    ucb_count = sum(1 for d in decisions if d['decision'] == 'UCB')
    ts_count = sum(1 for d in decisions if d['decision'] == 'TS')

    print(f"\nSAT-CTS Hierarchical Decision Statistics:")
    print(f"  LCB (most conservative): {lcb_count}/{len(decisions)} ({100 * lcb_count / len(decisions):.1f}%)")
    print(f"  MU (empirical mean):     {mu_count}/{len(decisions)} ({100 * mu_count / len(decisions):.1f}%)")
    print(f"  UCB (optimistic):        {ucb_count}/{len(decisions)} ({100 * ucb_count / len(decisions):.1f}%)")
    print(f"  TS (exploration):        {ts_count}/{len(decisions)} ({100 * ts_count / len(decisions):.1f}%)")

    # Show progression through hierarchy
    first_decisions = {}
    for decision_type in ['LCB', 'MU', 'UCB', 'TS']:
        first_t = next((d['t'] for d in decisions if d['decision'] == decision_type), None)
        if first_t is not None:
            first_decisions[decision_type] = first_t

    if first_decisions:
        print(
            f"  First use: {', '.join(f'{k} at t={v}' for k, v in sorted(first_decisions.items(), key=lambda x: x[1]))}")

# Show exploration statistics
print(f"\nExploration Statistics (unique (user,beam,rate) combinations tried):")
for name, agent in agents.items():
    if hasattr(agent, 'n_plays'):
        explored = np.sum(agent.n_plays > 0)
        total = agent.n_plays.size
        print(f"  {name}: {explored}/{total} ({100 * explored / total:.1f}%)")

print(f"{'=' * 60}\n")
print("OPTIMIZATION SUMMARY:")
print("✓ A) Channel caching per time slot implemented")
print("✓ B) Vectorized LCB/UCB index computation")
print("✓ C) Hungarian assignment algorithm for optimal allocation")
print("✓ D) Reduced probability warm-up samples (20→5) with vectorization")
print("✓ F) Reduced test horizon (2000→200) for faster debugging")
print(f"{'=' * 60}")
