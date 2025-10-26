#!/usr/bin/env python3
print("Starting simulation script...")
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

print("Imports complete. Loading configuration...")

sys.path.append(os.getcwd())
from user_beam_setup import get_default_test_configurations, create_users

cfg = get_default_test_configurations()[0]
print(f"Configuration loaded. Base path: {cfg.base_path}")

import glob
import re


def infer_num_users(cfg):
    """Infer number of users from available h_U*_B1.mat files under channel_data/"""
    base_dir = os.path.join(os.getcwd(), "channel_data")
    pattern = os.path.join(base_dir, "h_U*_B1.mat")
    files = glob.glob(pattern)
    if not files:
        return 3
    user_ids = []
    for filepath in files:
        match = re.search(r"h_U(\d+)_B1\.mat$", os.path.basename(filepath))
        if match:
            user_id = int(match.group(1))
            user_ids.append(user_id)
    num_users = max(user_ids) if user_ids else 3
    return num_users


num_users = infer_num_users(cfg)
print(f"Detected {num_users} users from channel files")

B = cfg.num_base_stations
K = cfg.arms_per_bs
M = num_users
total_beams = B * K
print(f"Configuration: {B} base stations, {K} beams per BS, {total_beams} total beams")

R_set  = np.array([6., 8., 10., 12.])
gamma_th = 2 ** R_set - 1

BW_HZ = 50e6
NF_DB = 5.0
N0_DBM_PER_HZ = -174.0
N_dBm = N0_DBM_PER_HZ + 10*np.log10(BW_HZ) + NF_DB
noise_var = 10 ** ((N_dBm - 30) / 10.0)


class TimeVaryingBeamEnvironment:
    def __init__(self, users, rate_set, noise_var=1e-10,
                 target_throughput=2.5):
        self.users = users
        self.num_users = len(users)
        self.rate_set = np.array(rate_set)
        self.num_rates = len(rate_set)
        self.noise_var = noise_var
        self.target_throughput = target_throughput
        self.total_beams = users[0].total_arms if users else B * K
        self.gamma_th = 2 ** self.rate_set - 1
        self._estimate_success_probabilities()

    def _get_beam_powers(self, time_slot):
        return [
            user.compute_beam_rewards(time_slot=time_slot).cpu().numpy()
            for user in self.users
        ]

    def _estimate_success_probabilities(self):
        self.psi = np.zeros((self.num_users, self.total_beams, self.num_rates))
        num_samples = 5
        for u, user in enumerate(self.users):
            samples = np.stack([
                user.compute_beam_rewards(time_slot=t).cpu().numpy()
                for t in range(num_samples)
            ], axis=0)
            mean_snr = samples.mean(axis=0) / self.noise_var
            thr = self.gamma_th[None, :]
            margin = (mean_snr[:, None] - thr) / np.sqrt(np.maximum(thr, 1.0))
            margin_clipped = np.clip(0.5 * margin, -30, 30)
            self.psi[u] = 1.0 / (1.0 + np.exp(-margin_clipped))
        self._find_optimal_superarm()

    def _hungarian_best(self, values):
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)
        scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        chosen_beams = [None] * U
        chosen_rates = [None] * U
        for u_idx, b_idx in zip(row_ind, col_ind):
            chosen_beams[u_idx] = int(b_idx)
            chosen_rates[u_idx] = int(best_r[u_idx, b_idx])
        return chosen_beams, chosen_rates

    def _find_optimal_superarm(self):
        throughput_matrix = self.rate_set[None, None, :] * self.psi
        chosen_beams, chosen_rates = self._hungarian_best(throughput_matrix)
        self.optimal_beams = chosen_beams
        self.optimal_rates = chosen_rates
        total_throughput = sum(
            self.rate_set[r] * self.psi[u, b, r]
            for u, (b, r) in enumerate(zip(chosen_beams, chosen_rates))
        )
        self.optimal_throughput = total_throughput / self.num_users

    def play(self, beam_assignment, rate_assignment, time_slot):
        ack_nack = np.zeros(self.num_users, dtype=int)
        instant_throughput = np.zeros(self.num_users)
        bp_list = self._get_beam_powers(time_slot)
        max_achievable_rates = []
        oracle_throughput = 0
        for u in range(self.num_users):
            beam_powers = bp_list[u]
            best_beam = -1
            best_rate = -1
            best_throughput = 0
            for b in range(self.total_beams):
                snr = beam_powers[b] / self.noise_var
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
            b = beam_assignment[u]
            r = rate_assignment[u]
            true_rss = beam_powers[b]
            snr = true_rss / self.noise_var
            thr = self.gamma_th[r]
            if snr >= thr:
                ack_nack[u] = 1
                instant_throughput[u] = self.rate_set[r]
        return ack_nack, instant_throughput, max_achievable_rates

    def compute_regret(self, beam_assignment, rate_assignment):
        throughput = 0
        for u in range(self.num_users):
            b = beam_assignment[u]
            r = rate_assignment[u]
            throughput += self.rate_set[r] * self.psi[u, b, r]
        avg_throughput = throughput / self.num_users
        return max(0.0, self.target_throughput - avg_throughput)

    def compute_standard_regret(self, beam_assignment, rate_assignment):
        throughput = 0.0
        for u in range(self.num_users):
            b = beam_assignment[u]
            r = rate_assignment[u]
            throughput += self.rate_set[r] * self.psi[u, b, r]
        avg_throughput = throughput / self.num_users
        return max(0.0, self.optimal_throughput - avg_throughput)


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
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)
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
        U, Btot, R = self.env.num_users, self.env.total_beams, self.env.num_rates
        rates = self.env.rate_set[np.newaxis, np.newaxis, :]
        psi_ts = self.rng.beta(self.A, self.B)
        theta_ts = rates * psi_ts
        ts_beams, ts_rates = self._hungarian_best(theta_ts)
        n = np.maximum(1, self.n_plays)
        psi_hat = self.n_success / n
        conf = np.sqrt(0.5 * np.log(max(2, t)) / n)
        psi_lcb = np.maximum(0.0, psi_hat - conf)
        lcb_values = rates * psi_lcb
        lcb_beams, lcb_rates = self._hungarian_best(lcb_values)
        lcb_sum = sum(lcb_values[u, lcb_beams[u], lcb_rates[u]] for u in range(U))
        lcb_avg = lcb_sum / U
        target_threshold = self.env.target_throughput
        total_threshold = target_threshold * U
        if lcb_sum >= total_threshold:
            chosen_beams, chosen_rates = lcb_beams, lcb_rates
            decision = "LCB"
        else:
            mu_values = rates * psi_hat
            mu_beams, mu_rates = self._hungarian_best(mu_values)
            mu_sum = sum(mu_values[u, mu_beams[u], mu_rates[u]] for u in range(U))
            if mu_sum >= total_threshold:
                chosen_beams, chosen_rates = mu_beams, mu_rates
                decision = "MU"
            else:
                conf = np.sqrt(0.5 * np.log(max(2, t)) / n)
                psi_ucb = psi_hat + conf
                ucb_values = rates * psi_ucb
                ucb_beams, ucb_rates = self._hungarian_best(ucb_values)
                ucb_sum = sum(ucb_values[u, ucb_beams[u], ucb_rates[u]] for u in range(U))
                if ucb_sum >= total_threshold:
                    chosen_beams, chosen_rates = ucb_beams, ucb_rates
                    decision = "UCB"
                else:
                    chosen_beams, chosen_rates = ts_beams, ts_rates
                    decision = "TS"
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
    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.RandomState(seed)

    def init(self):
        self.A = np.ones((self.env.num_users, self.env.total_beams, self.env.num_rates))
        self.B = np.ones((self.env.num_users, self.env.total_beams, self.env.num_rates))

    def _hungarian_best(self, values):
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)
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
        psi_samples = self.rng.beta(self.A, self.B)
        theta_samples = self.env.rate_set[np.newaxis, np.newaxis, :] * psi_samples
        chosen_beams, chosen_rates = self._hungarian_best(theta_samples)
        result = self.env.play(chosen_beams, chosen_rates, t)
        if len(result) == 3:
            ack_nack, instant_tput, oracle_info = result
        else:
            ack_nack, instant_tput = result
            oracle_info = None
        for u in range(self.env.num_users):
            b = chosen_beams[u]
            r = chosen_rates[u]
            self.A[u, b, r] += ack_nack[u]
            self.B[u, b, r] += 1 - ack_nack[u]
        return ack_nack, chosen_beams, chosen_rates


class CUCBAgent:
    def __init__(self, env):
        self.env = env

    def init(self):
        self.n_plays = np.zeros((self.env.num_users, self.env.total_beams, self.env.num_rates), dtype=int)
        self.psi_hat = np.zeros((self.env.num_users, self.env.total_beams, self.env.num_rates))

    def _hungarian_best(self, values):
        U, Btot, R = values.shape
        best_r = values.argmax(axis=2)
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
        n = np.maximum(1, self.n_plays)
        conf = np.sqrt(2 * np.log(max(2, t)) / n)
        ucb_psi = self.psi_hat + conf
        theta_ucb = self.env.rate_set[None, None, :] * ucb_psi
        chosen_beams, chosen_rates = self._hungarian_best(theta_ucb)
        result = self.env.play(chosen_beams, chosen_rates, t)
        if len(result) == 3:
            ack_nack, instant_tput, oracle_info = result
        else:
            ack_nack, instant_tput = result
            oracle_info = None
        for u in range(self.env.num_users):
            b = chosen_beams[u]
            r = chosen_rates[u]
            self.n_plays[u, b, r] += 1
            self.psi_hat[u, b, r] += (ack_nack[u] - self.psi_hat[u, b, r]) / self.n_plays[u, b, r]
        return ack_nack, chosen_beams, chosen_rates


def analyze_performance_gap(env, T):
    oracle_samples = []
    for t_sample in range(0, min(T, 100), 10):
        bp_list = env._get_beam_powers(t_sample)
        slot_max = 0
        for u in range(env.num_users):
            beam_powers = bp_list[u]
            user_max = 0
            for b in range(env.total_beams):
                snr = beam_powers[b] / env.noise_var
                for r_idx in range(len(env.rate_set) - 1, -1, -1):
                    if snr >= env.gamma_th[r_idx]:
                        user_max = max(user_max, env.rate_set[r_idx])
                        break
            slot_max += user_max
        oracle_samples.append(slot_max / env.num_users)


T = 10000
num_iterations = 100
target_throughput = 8

print(f"\nSimulation parameters: T={T}, iterations={num_iterations}, target_throughput={target_throughput}")
print("Running REALIZABLE experiment (target=8)\n")

all_regrets = {name: [] for name in ['CUCB', 'CTS', 'SAT-CTS']}
all_std_regrets = {name: [] for name in ['CUCB', 'CTS', 'SAT-CTS']}
all_throughputs = {name: [] for name in ['CUCB', 'CTS', 'SAT-CTS']}

for iteration in range(num_iterations):
    print(f"\n{'='*60}")
    print(f"Running iteration {iteration + 1}/{num_iterations}")
    print(f"{'='*60}")
    users = create_users(num_users, cfg)
    env = TimeVaryingBeamEnvironment(
        users=users,
        rate_set=R_set,
        noise_var=noise_var,
        target_throughput=target_throughput
    )
    analyze_performance_gap(env, T)
    agents = {
        'CUCB': CUCBAgent(env),
        'CTS': CTSAgent(env, seed=iteration),
        'SAT-CTS': SATCTSAgent(env, seed=iteration)
    }
    for ag in agents.values():
        ag.init()
    regrets = {name: np.zeros(T) for name in agents}
    std_regs = {name: np.zeros(T) for name in agents}
    throughputs = {name: np.zeros(T) for name in agents}
    for t in range(1, T + 1):
        if t % 100 == 0 or t == 1:
            print(f"  Time step: {t}/{T} ({100*t/T:.1f}%)")
        for name, agent in agents.items():
            ack_nack, beams, rates = agent.step(t)
            inst_reg = env.compute_regret(beams, rates)
            regrets[name][t - 1] = (regrets[name][t - 2] if t > 1 else 0) + inst_reg
            inst_std = env.compute_standard_regret(beams, rates)
            std_regs[name][t - 1] = (std_regs[name][t - 2] if t > 1 else 0) + inst_std
            throughputs[name][t - 1] = np.mean(ack_nack * env.rate_set[rates])
    for name in agents:
        all_regrets[name].append(regrets[name].copy())
        all_std_regrets[name].append(std_regs[name].copy())
        all_throughputs[name].append(throughputs[name].copy())

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

print("\n" + "="*70)
print("EXPERIMENT COMPLETED! Generating graphs...")
print("="*70)

colors = {'CUCB': 'blue', 'CTS': 'orange', 'SAT-CTS': 'green'}
time_steps = np.arange(1, T + 1)
ERR_EVERY = max(1, T // 200)

print("\nGenerating Figure 1: Realizable case (target=8) - Satisficing regret")
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111)
for algo_name in ['CUCB', 'CTS', 'SAT-CTS']:
    mean_reg = mean_regrets[algo_name]
    std_reg = std_regrets[algo_name]
    col = colors[algo_name]
    ax1.plot(time_steps, mean_reg, color=col, linewidth=2, label=algo_name)
    lower = np.maximum(0, mean_reg - std_reg)
    upper = mean_reg + std_reg
    ax1.fill_between(time_steps, lower, upper, color=col, alpha=0.15)
    idx = np.arange(0, T, ERR_EVERY, dtype=int)
    ax1.errorbar(time_steps[idx], mean_reg[idx], yerr=std_reg[idx], fmt='none',
                 ecolor=col, elinewidth=1, capsize=2, alpha=0.9)
ax1.set_xlabel('Time Slot', fontsize=12)
ax1.set_ylabel('Cumulative Satisficing Regret (bits/symbol)', fontsize=12)
ax1.set_title('Realizable Case (target=8) — Satisficing Regret', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('realizable_satisficing.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved as 'realizable_satisficing.png'")

print("\nGenerating Figure 2: CTS vs SAT-CTS - Both regrets (realizable case)")
fig2 = plt.figure(figsize=(12, 6))
ax2 = fig2.add_subplot(111)
pair_defs = [
    ('CTS',      'satisficing', mean_regrets['CTS'],           std_regrets['CTS'],           colors['CTS']),
    ('SAT-CTS',  'satisficing', mean_regrets['SAT-CTS'],       std_regrets['SAT-CTS'],       colors['SAT-CTS']),
    ('CTS',      'standard',    mean_std_regrets['CTS'],       std_std_regrets['CTS'],       'tab:purple'),
    ('SAT-CTS',  'standard',    mean_std_regrets['SAT-CTS'],   std_std_regrets['SAT-CTS'],   'tab:brown'),
]
for algo_name, kind, mean_reg, std_reg, col in pair_defs:
    ax2.plot(time_steps, mean_reg, color=col, linewidth=2,
             label=f'{algo_name} ({kind})')
    lower = np.maximum(0, mean_reg - std_reg)
    upper = mean_reg + std_reg
    ax2.fill_between(time_steps, lower, upper, color=col, alpha=0.15)
    idx = np.arange(0, T, ERR_EVERY, dtype=int)
    ax2.errorbar(time_steps[idx], mean_reg[idx], yerr=std_reg[idx], fmt='none',
                 ecolor=col, elinewidth=1, capsize=2, alpha=0.9)
ax2.set_xlabel('Time Slot', fontsize=12)
ax2.set_ylabel('Cumulative Regret (bits/symbol)', fontsize=12)
ax2.set_title('CTS vs SAT-CTS — Satisficing & Standard Regret (target=8)', fontsize=14)
ax2.legend(fontsize=10, ncol=2)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('cts_vs_satcts_both_regrets.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved as 'cts_vs_satcts_both_regrets.png'")

print("\n" + "="*70)
print("ALL FIGURES SAVED SUCCESSFULLY!")
print("="*70)

