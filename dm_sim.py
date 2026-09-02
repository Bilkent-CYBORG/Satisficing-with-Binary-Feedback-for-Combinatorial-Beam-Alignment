#!/usr/bin/env python3
"""Command line entry point for the ray-traced beam-and-rate simulator.

ONE channel model, no setup switch. The chain is:

  DeepMIMO `city_3_houston_28` (28 GHz ray tracing)
      |  per (UE, BS): the L strongest paths' AoDs theta_l and average
      |  powers sigma_l^2.  FIXED for the whole horizon.
      v
  h_{m,b}(t) = sqrt(N) sum_l beta_l(t) a(cos theta_l),
               beta_l(t) = sigma_l eps_l(t),  eps ~ CN(0,1) i.i.d. per round
      |  El Ayach's sparse geometric channel with zero-mean complex Gaussian
      |  path gains; sigma_l^2 read from the ray tracer instead of assumed equal.
      v
  gamma_{m,(b,k)}(t) = (P_tx / sigma^2_noise) |h^H f_k|^2   ~ Exponential
      v
  measured Sionna 5G-NR LDPC BLER (nr_bler_table_v2.json, n_cu = 624)
      v
  psi = P(ACK) in closed form  ->  g*  ->  satisficing regret

The model itself lives in the `obs` package, one concern per module:

  obs.config                       scenario, link budget, every env knob
  obs.environment.deepmimo_geometry  ray-traced angles/powers, DFT codebook
  obs.user.beam_user               the per-round fading channel
  obs.simulation.bler              measured NR-LDPC BLER -> P(ACK)
  obs.simulation.ground_truth      closed-form psi and the oracle g*
  obs.simulation.methods           which agents exist and how they are built
  obs.simulation.dm_simulation     the round loop (CRN, regret, QoS metrics)
  obs.simulation.provenance        the reproduction contract

Usage: python dm_sim.py            (all configuration is by environment)
Env:   DM_SCENARIO, PTX_DBM, NF_DB, DM_MIN_PATHS, BLER_TABLE, FEEDBACK_P,
       N_RF, RHO, BLOCK_*, INTERFERENCE, SIM_* (see __main__).
"""
import gzip
import json
import os

import numpy as np

from obs.config import (
    DM_NOISE_DBM, DM_N_RF, DM_PTX_DBM, DM_SCENARIO, MODEL_NAME, NR_RATE_SET,
)
from obs.simulation.bler import (
    BLER_TABLE_PATH, measured_bler, measured_success_prob, success_prob,
)
from obs.simulation.dm_simulation import run
from obs.simulation.ground_truth import estimate_psi, optimal_throughput
from obs.simulation.methods import NAMES, NAMES_G, STYLE
from obs.simulation.provenance import git_provenance, link_budget_config
from obs.user.beam_user import BeamUser, build_users


def main():
    from datetime import datetime, timezone

    T      = int(os.environ.get("SIM_T", 10000))
    n_exp  = int(os.environ.get("SIM_ITERS", 5))
    target = float(os.environ.get("SIM_TARGET", 4.55))
    seed   = int(os.environ.get("SIM_SEED", 0))
    M      = int(os.environ.get("SIM_USERS", 15))
    nbs    = int(os.environ.get("SIM_BS", 3))
    Kb     = int(os.environ.get("SIM_K", 120))
    Nant   = int(os.environ.get("SIM_N_ANT", 64))
    sstr   = int(os.environ.get("SELECTION_STRIDE", 1))
    npaths = int(os.environ.get("SIM_PATHS", 3))
    savesel = os.environ.get("SAVE_SELECTIONS", "1") not in ("0", "false", "no")
    meth   = os.environ.get("SIM_METHODS", ",".join(NAMES))
    methods = [m.strip() for m in meth.split(",") if m.strip()]
    tag    = os.environ.get("TAG", "dm")
    outdir = os.environ.get("OUT_DIR", "results")

    print(f"[cfg] {MODEL_NAME} scenario={DM_SCENARIO} T={T} iters={n_exp} "
          f"target={target} L={npaths} "
          f"P_tx={DM_PTX_DBM} dBm noise={DM_NOISE_DBM:.2f} dBm N_RF={DM_N_RF} "
          f"bler={os.path.basename(BLER_TABLE_PATH)} methods={methods}",
          flush=True)

    res = run(num_users=M, num_bs=nbs, N=Nant, K=Kb, T=T, n_exp=n_exp,
              target=target, seed=seed, methods=methods,
              save_selections=savesel, selection_stride=sstr,
              n_paths=npaths)

    gm = res["geometry"]
    print(f"[geo] {gm['links']} links, {gm['links_with_zero_paths']} dead, "
          f"{gm['links_below_L']} with < L={npaths} paths; snap <= "
          f"{gm['snap_distance_max_m']:.1f} m; LoS {gm['los_links']}/{gm['links']}; "
          f"kept power median {100*gm['kept_power_fraction_median']:.1f}%",
          flush=True)
    print(f"g* = {res['gstar']:.4f}   (target {target} => "
          f"{'realizable' if target < res['gstar'] else 'NON-realizable'})",
          flush=True)

    psi = res["psi"]
    print(f"[arms] dead(<=0.05) {100*(psi<=.05).mean():.2f}%  "
          f"informative {100*((psi>.05)&(psi<.95)).mean():.2f}%  "
          f"always-on(>=0.95) {100*(psi>=.95).mean():.2f}%", flush=True)

    cfg = {"T": T, "n_experiments": n_exp, "target_throughput": target,
           "gstar": res["gstar"], "seed": seed,
           "num_users": M, "num_bs": nbs, "K": Kb, "N_antennas": Nant,
           "total_beams": nbs * Kb, "base_arms": M * nbs * Kb * len(NR_RATE_SET),
           "crn": res["crn"], "selection_stride": res["selection_stride"],
           "N_RF_per_bs": res["n_rf"], "rf_cap_stats": res["rf_cap_stats"],
           "arm_split_pct": {
               "dead_le_0.05": float(100 * (psi <= .05).mean()),
               "informative": float(100 * ((psi > .05) & (psi < .95)).mean()),
               "always_on_ge_0.95": float(100 * (psi >= .95).mean())},
           **link_budget_config(npaths),
           "geometry": res["geometry"],
           **git_provenance(),
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    out = {"config": cfg, "results": {}}
    for n in res["names"]:
        r, ro, rs = res["reg"][n], res["reg_only"][n], res["reg_std"][n]
        j, sl = res["jain"][n], res["sumlog"][n]
        G = np.asarray(res["per_ue_G"][n], float)
        ph = [p for run_ph in res["phase"][n] for p in run_ph if p is not None]
        out["results"][n] = {
            "mean_cum_regret": r.mean(0).tolist(), "std_cum_regret": r.std(0).tolist(),
            "jain_mean": j.mean(0).tolist(), "jain_std": j.std(0).tolist(),
            "sumlog_mean": sl.mean(0).tolist(), "sumlog_std": sl.std(0).tolist(),
            "final_mean": float(r.mean(0)[-1]), "final_std": float(r.std(0)[-1]),
            "final_jain": float(j.mean(0)[-1]), "final_sumlog": float(sl.mean(0)[-1]),
            "per_seed_cum_regret": r.tolist(),
            "per_seed_sat_only_cum_regret": ro.tolist(),
            "per_seed_std_cum_regret": rs.tolist(),
            "per_seed_jain": j.tolist(),
            "per_seed_sumlog": sl.tolist(),
            "per_ue_cum_throughput": res["per_ue_G"][n],
            "n_arm_switches": res["switches"][n],
            "slots_per_phase": {k: ph.count(k) for k in sorted(set(ph))},
            "frac_ue_meeting_target": [float(np.mean((g / T) >= target)) for g in G],
            "min_ue_avg_throughput": [float(np.min(g / T)) for g in G],
            "final_sat_only_mean": float(ro.mean(0)[-1]),
            "final_std_regret_mean": float(rs.mean(0)[-1]),
        }
        print(f"  {n:22s} regret={r.mean(0)[-1]:9.1f} +/- {r.std(0)[-1]:7.1f}  "
              f"(sat_only={ro.mean(0)[-1]:9.1f}, std={rs.mean(0)[-1]:8.1f})  "
              f"jain={j.mean(0)[-1]:.3f}  sumlog={sl.mean(0)[-1]:.1f}", flush=True)

    out["ground_truth"] = {"gstar": res["gstar"], "tau_r": target,
                           "psi_shape": list(res["psi"].shape),
                           "psi": res["psi"].tolist()}
    if savesel:
        out["selections"] = {n: res["sel"][n] for n in res["names"]}
        out["phase"] = {n: res["phase"][n] for n in res["names"]}
        out["selection_encoding"] = "per seed -> per stored slot -> [beams, rates, acks]"

    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, f"{tag}_T{T}_tau{target}.json.gz")
    with gzip.open(jpath, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"JSON: {jpath} ({os.path.getsize(jpath)/1e6:.2f} MB gz)", flush=True)
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
