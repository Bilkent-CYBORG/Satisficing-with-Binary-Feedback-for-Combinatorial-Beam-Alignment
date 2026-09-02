#!/usr/bin/env python3
"""DeepMIMO ray-traced beam/rate satisficing comparison.

One tau per invocation so the ladder can run as independent processes -- that is
the only parallel split that preserves CRN, since all methods must share one
channel stream per slot.

Records, per method:
  satisficing regret (paper's min(standard, satisficing)), standard regret,
  average throughput, time to reach the threshold, per-UE throughput, Jain,
  switches, and decision/update wall time.

Also writes the full link budget, geometry placement report and git provenance
into every metrics file, so a run is reproducible from its own output.

Env: TAU (required), T, N_EXP, SEED, OUT_DIR, L_PATHS, DM_MIN_PATHS, PTX_DBM,
     METHODS (comma-separated subset of dm_sim.NAMES; default all).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_sim as R

TAU = float(os.environ["TAU"])
T = int(os.environ.get("T", 10000))
N_EXP = int(os.environ.get("N_EXP", 15))
SEED = int(os.environ.get("SEED", 0))
LP = int(os.environ.get("L_PATHS", 3))
OUT = os.environ.get("OUT_DIR", "results/deepmimo_compare")
STRIDE = 10
HOLD = 100
M = int(os.environ.get("M_USERS", 15))
B = int(os.environ.get("N_BS", 3))
K = int(os.environ.get("N_BEAMS", 120))
METHODS = [m.strip() for m in
           os.environ.get("METHODS", ",".join(R.NAMES)).split(",") if m.strip()]
NANT = int(os.environ.get("N_ANT", 64))
NRF = int(os.environ.get("N_RF", 8))


def time_to_threshold(reg_only_cum, tau, hold=HOLD):
    """First round after which per-round satisficing regret stays 0 for `hold`.

    reg_only_cum is the CUMULATIVE max(0, tau - achieved).  Its increment is the
    per-round shortfall, so a run of zero increments means the threshold is being
    met continuously.  Returns np.nan if it never happens.
    """
    inc = np.diff(np.concatenate([[0.0], reg_only_cum]))
    met = inc <= 1e-12
    if not met.any() or met.size < hold:
        return np.nan
    ok = np.convolve(met.astype(int), np.ones(hold, int), "valid") == hold
    idx = np.flatnonzero(ok)
    return float(idx[0] + 1) if idx.size else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = f"tau{TAU:.2f}"
    t0 = time.time()
    print(f"[cfg] {R.MODEL_NAME}  scenario={R.DM_SCENARIO}  "
          f"P_tx={R.DM_PTX_DBM} dBm  N_RF={NRF}  tau={TAU}  T={T}  n_exp={N_EXP}  "
          f"seed={SEED}  L={LP}", flush=True)

    res = R.run(num_users=M, num_bs=B, N=NANT, K=K, T=T, n_exp=N_EXP,
                rate_set=np.array(R.NR_RATE_SET), target=TAU,
                n_paths=LP, crn=True, seed=SEED, metric_stride=STRIDE, n_rf=NRF,
                methods=METHODS, save_selections=False)
    wall = time.time() - t0
    g = res["gstar"]
    gm = res["geometry"]
    psi = res["psi"]
    print(f"[geo] {gm['links']} links, {gm['links_with_zero_paths']} dead, "
          f"{gm['links_below_L']} with < L={LP} paths; snap <= "
          f"{gm['snap_distance_max_m']:.1f} m", flush=True)
    rc = res["rf_cap_stats"]
    if rc:
        print(f"[rf]  cap fired on {rc['capped_calls']}/{rc['oracle_calls']} "
              f"oracle calls ({100*rc['capped_frac']:.2f}%)", flush=True)
    print(f"[done] g*={g:.4f}  wall={wall/60:.1f} min", flush=True)

    out = {"tau": TAU, "T": T, "n_exp": N_EXP, "seed": SEED, "L": LP,
           "gstar": g, "wall_s": wall,
           "realizable": bool(TAU <= g), "stride": STRIDE,
           "config": {**R.link_budget_config(LP), **R.git_provenance(),
                      "num_users": M, "num_bs": B, "K": K, "N_antennas": NANT,
                      "total_beams": B * K, "N_RF_per_bs": res["n_rf"],
                      "rf_cap_stats": res["rf_cap_stats"],
                      "uncapped_methods": res["uncapped_methods"],
                      "rf_cap_note": (
                          "per-BS RF-chain cap |{m : b_m = b}| <= N_RF enforced "
                          "exactly (Hungarian first, transportation LP only when "
                          "the cap binds). Methods in uncapped_methods run "
                          "WITHOUT the cap, i.e. on a larger feasible set."),
                      "base_arms": M * B * K * len(R.NR_RATE_SET),
                      "metric_stride": STRIDE,
                      "geometry": gm,
                      "arm_split_pct": {
                          "dead_le_0.05": float(100 * (psi <= .05).mean()),
                          "informative": float(
                              100 * ((psi > .05) & (psi < .95)).mean()),
                          "always_on_ge_0.95": float(100 * (psi >= .95).mean())}},
           "methods": {}}

    for m in res["names"]:
        rs_ = res["reg"][m]
        rd = res["reg_std"][m]
        ro = res["reg_only"][m]
        ach = g - np.diff(np.concatenate([np.zeros((N_EXP, 1)), rd], axis=1),
                          axis=1)
        hit = np.array([time_to_threshold(ro[e], TAU) for e in range(N_EXP)])
        G = np.asarray(res["per_ue_G"][m])
        J = np.asarray(res["jain"][m])
        SL = np.asarray(res["sumlog"][m])
        C = np.asarray(res["per_ue_curve"][m])
        out["methods"][m] = {
            "reg_mean": rs_.mean(0)[::STRIDE].tolist(),
            "reg_sd": rs_.std(0)[::STRIDE].tolist(),
            "reg_std_mean": rd.mean(0)[::STRIDE].tolist(),
            "reg_std_sd": rd.std(0)[::STRIDE].tolist(),
            "reg_final_mean": float(rs_[:, -1].mean()),
            "reg_final_sd": float(rs_[:, -1].std()),
            "reg_std_final_mean": float(rd[:, -1].mean()),
            "reg_std_final_sd": float(rd[:, -1].std()),
            "reg_only_final_mean": float(ro[:, -1].mean()),
            "avg_tput_mean": float(ach.mean()),
            "avg_tput_sd": float(ach.mean(1).std()),
            "final_tput_mean": float(ach[:, -T // 10:].mean()),
            "tput_frac_of_gstar": float(ach.mean() / g),
            "t_hit_mean": float(np.nanmean(hit)) if np.isfinite(hit).any() else None,
            "t_hit_sd": float(np.nanstd(hit)) if np.isfinite(hit).any() else None,
            "t_hit_frac_runs": float(np.isfinite(hit).mean()),
            "jain_mean": J.mean(0)[::STRIDE].tolist(),
            "jain_sd": J.std(0)[::STRIDE].tolist(),
            "ue_outage_frac_mean": float(np.asarray(res["ue_outage_frac"][m]).mean()),
            "ue_outage_frac_worst": float(np.asarray(res["ue_outage_frac"][m]).max(axis=1).mean()),
            "ue_outage_max_slots_mean": float(np.asarray(res["ue_outage_max"][m]).mean()),
            "ue_outage_max_slots_worst": float(np.asarray(res["ue_outage_max"][m]).max(axis=1).mean()),
            "jain_final_mean": float(J[:, -1].mean()),
            "jain_final_sd": float(J[:, -1].std()),
            "jain_per_seed_final": J[:, -1].tolist(),
            "sumlog_mean": SL.mean(0)[::STRIDE].tolist(),
            "sumlog_sd": SL.std(0)[::STRIDE].tolist(),
            "sumlog_final_mean": float(SL[:, -1].mean()),
            "sumlog_final_sd": float(SL[:, -1].std()),
            "meanlog_final_mean": float(SL[:, -1].mean() / M),
            "sumlog_per_seed_final": SL[:, -1].tolist(),
            "per_ue_G_mean": (G.mean(0) / T).tolist(),
            "per_ue_G_per_seed": (G / T).tolist(),
            "per_ue_curve_mean": (C.mean(0) / T).tolist(),
            "min_ue_avg_throughput": float((G / T).min(axis=1).mean()),
            "max_ue_avg_throughput": float((G / T).max(axis=1).mean()),
            "frac_pass_timeavg_mean": float(np.mean(res["frac_pass_timeavg"][m])),
            "frac_pass_timeavg_sd": float(np.std(res["frac_pass_timeavg"][m])),
            "frac_ue_meeting_target": float(
                np.mean((G / T) >= TAU)),
            "n_starved_ue_mean": float(
                ((G / T) < 0.5 * TAU).sum(axis=1).mean()),
            "worst_ue_curve_mean": (C.mean(0) / T).min(axis=1).tolist(),
            "switches_mean": float(np.mean(res["switches"][m])),
            "sel_time_s": float(res["sel_time_s"][m]),
            "upd_time_s": float(res["upd_time_s"][m]),
            "us_per_decision": float(1e6 * res["sel_time_s"][m] / (T * N_EXP)),
        }

    path = os.path.join(OUT, f"metrics_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {path}", flush=True)

    hdr = (f"{'method':22s} {'sat regret':>13} {'std regret':>13} "
           f"{'avg tput':>10} {'T_hit':>9} {'Jain':>6} {'mean-log':>9} "
           f"{'worst UE':>9} {'us/dec':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for m in res["names"]:
        d = out["methods"][m]
        th = "never" if d["t_hit_mean"] is None else f"{d['t_hit_mean']:.0f}"
        print(f"{m:22s} {d['reg_final_mean']:9.1f}+-{d['reg_final_sd']:<4.0f} "
              f"{d['reg_std_final_mean']:9.1f}+-{d['reg_std_final_sd']:<4.0f} "
              f"{d['avg_tput_mean']:10.4f} {th:>9} "
              f"{d['jain_final_mean']:6.3f} {d['meanlog_final_mean']:9.3f} "
              f"{d['min_ue_avg_throughput']:9.3f} "
              f"{d['us_per_decision']:8.1f}")


if __name__ == "__main__":
    main()
