"""The round loop: one channel realization per slot, shared by every method.

`crn=True` implements the common-random-numbers protocol. A single channel
draw per slot is handed to every method, so a difference between two methods
can never come from channel luck. The decoding coin is necessarily per-method,
since each method selects a different beam-rate pair; it lives on a SEPARATE
generator so the channel stream does not shift when the method set changes.
Feedback corruption gets a third generator for the same reason.
"""
import time

import numpy as np

from obs.algorithms.combinatorial.objectives import (
    CapacitatedThroughputObjective,
    SharedCapacitatedThroughputObjective,
    SharedThroughputObjective,
)
from obs.config import (
    DM_BLOCK_DB,
    DM_BEAM_EXCLUSIVE,
    DM_FEEDBACK_P,
    DM_INTERFERENCE,
    rf_chains,
)
from obs.simulation.bler import success_prob
from obs.simulation.ground_truth import estimate_psi, optimal_throughput
from obs.simulation.methods import NAMES, make_agents
from obs.user.beam_user import build_users

_INTERFERENCE_ON_NOTE = (
    "q_b orthogonal sub-channels REMOVED when interference is on: the same "
    "4-PRB sub-channel is spatially reused across all scheduled beams, so "
    "co-scheduled UEs are co-channel. n_cu = 624 and the measured BLER table "
    "are unchanged. Interference is summed over every other scheduled beam at "
    "full P_tx (no power control, no coordination).")
_INTERFERENCE_OFF_NOTE = (
    "q_b = 8 orthogonal sub-channels: co-scheduled UEs never share a "
    "time-frequency resource, so inter-user interference is zero by "
    "construction and gamma is an SNR.")
_BENCHMARK_NOTE = (
    "psi and g* are computed WITHOUT interference. With interference on, the "
    "per-slot reward is coupled across users (the reward is no longer a linear "
    "function of independent base arms), so g* is an upper bound and the "
    "reported regret is conservative. Blockage IS folded into psi, so g* stays "
    "the stationary optimum.")


def run(num_users=15, num_bs=3, N=64, K=120, T=10000, n_exp=5,
        rate_set=None, target=4.6,
        crn=True, seed=0, methods=None,
        save_selections=True, selection_stride=1, metric_stride=10,
        n_paths=3, n_rf=None, interference=None, rho=None, p01=None, p10=None,
        feedback_p=None, beam_exclusive=None):
    """Run the selected methods on the ray-traced channel.

    Three regret definitions are recorded per slot, because they diverge in the
    non-realizable regime and only one of them matches the paper:
      - `reg`      max(0, target - achieved)   <- the paper's satisficing regret
      - `reg_only` the same quantity, kept under its historical name
      - `reg_std`  max(0, g* - achieved)       <- standard regret

    Per-UE cumulative throughput is recorded every `metric_stride` rounds as
    well as at T, so fairness can be plotted as a trajectory rather than only
    as an endpoint. Under CRN the UE indices are stable across seeds and
    methods -- user 7 is the same user everywhere -- so averaging a per-UE
    curve over seeds is meaningful.
    """
    from obs.config import NR_RATE_SET
    rate_set = np.array(NR_RATE_SET) if rate_set is None else rate_set
    methods = list(methods) if methods else list(NAMES)
    n_rf = rf_chains(num_users, num_bs) if n_rf is None else n_rf
    feedback_p = DM_FEEDBACK_P if feedback_p is None else float(feedback_p)
    beam_exclusive = (DM_BEAM_EXCLUSIVE if beam_exclusive is None
                      else int(beam_exclusive))
    interference = DM_INTERFERENCE if interference is None else interference
    rng = np.random.default_rng(seed)
    tb = num_bs * K
    if beam_exclusive and num_users > tb:
        raise ValueError(
            f"{num_users} users but only {tb} beams (B*K); every user needs a "
            f"distinct beam")

    users, geo_meta = build_users(num_users, num_bs, N, K, rng, n_paths=n_paths,
                                  rho=rho, p01=p01, p10=p10)
    psi = estimate_psi(users, rate_set)

    beam_to_bs = np.repeat(np.arange(num_bs), K)
    if n_rf and n_rf > 0:
        cap = np.full(num_bs, int(n_rf))
        objective = (CapacitatedThroughputObjective(beam_to_bs, cap)
                     if beam_exclusive
                     else SharedCapacitatedThroughputObjective(beam_to_bs, cap))
    else:
        cap = None
        objective = None if beam_exclusive else SharedThroughputObjective()
    gstar = optimal_throughput(psi, rate_set, beam_to_bs, cap,
                               beam_exclusive=bool(beam_exclusive))

    Z = lambda: {n: np.zeros((n_exp, T)) for n in methods}
    reg, reg_only, reg_std = Z(), Z(), Z()
    jain, sumlog = Z(), Z()
    sel = {n: [] for n in methods}
    phase = {n: [] for n in methods}
    sel_time = {n: 0.0 for n in methods}
    upd_time = {n: 0.0 for n in methods}
    per_ue_G = {n: [] for n in methods}
    frac_pass = {n: [] for n in methods}
    ue_outage_frac = {n: [] for n in methods}
    ue_outage_max = {n: [] for n in methods}
    per_ue_curve = {n: [] for n in methods}
    switches = {n: [] for n in methods}

    for e in range(n_exp):
        agents = make_agents(methods, num_users, tb, rate_set, target, objective)
        np.random.seed((seed * 7919 + e) % (2**32))
        coin = np.random.default_rng((seed + 1) * 1_000_003 + e)
        fbcoin = np.random.default_rng((seed + 1) * 7_700_017 + e)
        Gm = {n: np.zeros(num_users) for n in methods}
        n_pass = {n: 0 for n in methods}
        n_out = {n: np.zeros(num_users, dtype=np.int64) for n in methods}
        run_out = {n: np.zeros(num_users, dtype=np.int64) for n in methods}
        max_out = {n: np.zeros(num_users, dtype=np.int64) for n in methods}
        cum = {n: 0.0 for n in methods}
        cum_only = {n: 0.0 for n in methods}
        cum_std = {n: 0.0 for n in methods}
        e_sel = {n: [] for n in methods}
        e_phase = {n: [] for n in methods}
        n_sw = {n: 0 for n in methods}
        prev = {n: None for n in methods}
        ue_curve = {n: [] for n in methods}

        for t in range(1, T + 1):
            snr_lin = np.stack([u.block_snr() for u in users])
            snr_db = 10 * np.log10(np.maximum(snr_lin, 1e-12))

            for n in methods:
                _t0 = time.perf_counter()
                beams, rates = agents[n].select_action(t)
                sel_time[n] += time.perf_counter() - _t0
                beams = list(map(int, beams))
                rates = list(map(int, rates))
                ack = np.empty(num_users, dtype=int)
                pu = np.empty(num_users)
                exp_tput = 0.0
                if interference:
                    tot = snr_lin[np.arange(num_users)][:, beams].sum(axis=1)
                    sig = snr_lin[np.arange(num_users), beams]
                    sinr_db = 10 * np.log10(
                        np.maximum(sig / (1.0 + tot - sig), 1e-12))
                else:
                    sinr_db = None

                for uu in range(num_users):
                    b, r = beams[uu], rates[uu]
                    g_db = snr_db[uu, b] if sinr_db is None else sinr_db[uu]
                    p = success_prob(g_db, rate_set[r])
                    ack[uu] = int(coin.random() < p)
                    pu[uu] = rate_set[r] * ack[uu]
                    exp_tput += rate_set[r] * psi[uu, b, r]
                if feedback_p < 1.0:
                    flip = fbcoin.random(num_users) >= feedback_p
                    ack_obs = np.where(flip, 1 - ack, ack)
                else:
                    ack_obs = ack
                _t0 = time.perf_counter()
                agents[n].update(beams, rates, ack_obs)
                upd_time[n] += time.perf_counter() - _t0

                ach = exp_tput / num_users
                r_only = max(0.0, target - ach)
                r_std = max(0.0, gstar - ach)
                cum_only[n] += r_only
                cum_std[n] += r_std
                cum[n] += r_only
                reg[n][e, t - 1] = cum[n]
                reg_only[n][e, t - 1] = cum_only[n]
                reg_std[n][e, t - 1] = cum_std[n]

                Gm[n] += pu
                n_pass[n] += int((Gm[n] >= target * t).sum())
                miss = ack == 0
                n_out[n] += miss
                run_out[n] = (run_out[n] + 1) * miss
                np.maximum(max_out[n], run_out[n], out=max_out[n])
                s1, s2 = Gm[n].sum(), (Gm[n] ** 2).sum()
                jain[n][e, t - 1] = (s1 * s1) / (num_users * s2) if s2 > 0 else 1.0 / num_users
                sumlog[n][e, t - 1] = np.log(np.maximum(Gm[n], 1e-12)).sum()

                cur = (tuple(beams), tuple(rates))
                if prev[n] is not None and cur != prev[n]:
                    n_sw[n] += 1
                prev[n] = cur

                if save_selections and (t - 1) % selection_stride == 0:
                    e_sel[n].append([beams, rates, ack.tolist()])
                    e_phase[n].append(getattr(agents[n], "last_decision", None))

                if (t - 1) % metric_stride == 0 or t == T:
                    ue_curve[n].append(Gm[n].copy())

        for n in methods:
            frac_pass[n].append(n_pass[n] / float(T * num_users))
            ue_outage_frac[n].append((n_out[n] / T).tolist())
            ue_outage_max[n].append(max_out[n].tolist())
            per_ue_G[n].append(Gm[n].tolist())
            per_ue_curve[n].append(np.asarray(ue_curve[n]))
            switches[n].append(n_sw[n])
            if save_selections:
                sel[n].append(e_sel[n])
                phase[n].append(e_phase[n])

    return {"names": methods, "reg": reg, "reg_only": reg_only, "reg_std": reg_std,
            "jain": jain, "sumlog": sumlog, "sel": sel, "phase": phase,
            "per_ue_G": per_ue_G,
            "ue_outage_frac": ue_outage_frac, "ue_outage_max": ue_outage_max,
            "frac_pass_timeavg": frac_pass,
            "per_ue_curve": {n: np.stack(per_ue_curve[n]) for n in methods},
            "metric_stride": metric_stride,
            "switches": switches, "psi": psi,
            "gstar": float(gstar), "rate_set": rate_set.tolist(),
            "target": target, "T": T, "n_exp": n_exp, "crn": bool(crn),
            "num_users": num_users, "num_bs": num_bs, "N": N, "K": K,
            "selection_stride": selection_stride,
            "sel_time_s": sel_time, "upd_time_s": upd_time,
            "n_paths": n_paths, "geometry": geo_meta,
            "n_rf": (int(n_rf) if n_rf else 0),
            "feedback_p": float(feedback_p),
            "beam_exclusive": bool(beam_exclusive),
            "interference": bool(interference),
            "interference_note": (_INTERFERENCE_ON_NOTE if interference
                                  else _INTERFERENCE_OFF_NOTE),
            "rho": float(users[0].rho),
            "block_p01": float(users[0].p01), "block_p10": float(users[0].p10),
            "block_db": (DM_BLOCK_DB if users[0].p01 > 0 else None),
            "block_steady_frac": (float(users[0].p01 /
                                        (users[0].p01 + users[0].p10))
                                  if users[0].p01 > 0 else 0.0),
            "regret_benchmark_note": _BENCHMARK_NOTE,
            "rf_cap_stats": ({"oracle_calls": objective.n_calls,
                              "capped_calls": objective.n_capped,
                              "capped_frac": (objective.n_capped
                                              / max(1, objective.n_calls))}
                             if objective is not None else None)}
