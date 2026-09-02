"""The reproduction contract written into every results file.

A run is reproducible from its own output: the link budget, the UE placement
report, the identity of the BLER table, and the git commit plus dirty flag all
travel with the numbers.
"""
import os
import subprocess
import sys

import numpy as np

from obs.config import (
    DM_B_SC,
    DM_FEEDBACK_P,
    DM_MCS_INDICES,
    DM_MIN_PATHS,
    DM_MIN_UE_SEP_M,
    DM_N_CU,
    DM_N_PRB,
    DM_NF_DB,
    DM_NOISE_DBM,
    DM_PTX_DBM,
    DM_Q_B,
    DM_REQUIRE_LOS,
    DM_SCENARIO,
    DM_SCS_KHZ,
    DM_TBS_BITS,
    MODEL_NAME,
    NR_RATE_SET,
)
from obs.simulation.bler import BLER_TABLE_PATH


def link_budget_config(n_paths=3):
    """Everything needed to reproduce the SNR scale, as a dict for config.json."""
    return {
        "model": MODEL_NAME,
        "scenario": DM_SCENARIO,
        "carrier_GHz": 28.0,
        "antenna_spacing_lambda": 0.5,
        "n_paths_kept": n_paths,
        "min_paths_per_link": DM_MIN_PATHS,
        "feedback_p": DM_FEEDBACK_P,
        "feedback_note": ("reliability of the ACK/NACK channel; the learner sees "
                          "the flipped bit with probability 1-p, while throughput "
                          "and regret are accounted on the true outcome"),
        "require_los_links": DM_REQUIRE_LOS,
        "min_ue_separation_m": DM_MIN_UE_SEP_M,
        "placement_note": ("require_los_links > 0 is a DEPLOYMENT ASSUMPTION: "
                           "it excludes grid points blocked from every BS, "
                           "which are exactly the users that starve, so it "
                           "improves the per-UE fairness numbers by "
                           "construction and must be reported alongside them"),
        "path_selection": "L strongest by ray-traced power, per (UE, BS) link",
        "sigma2_source": ("DeepMIMO `power` field, dBW received at 0 dBW "
                          "transmitted, i.e. numerically the path gain in dB; "
                          "sigma_l^2 = 10^(power/10) is dimensionless"),
        "P_tx_dBm": DM_PTX_DBM,
        "noise_figure_dB": DM_NF_DB,
        "noise_floor_dBm": float(DM_NOISE_DBM),
        "noise_floor_formula": "-174 + 10log10(B_sc) + NF",
        "snr_scale_dB": float(DM_PTX_DBM - DM_NOISE_DBM),
        "assumed_not_measured": ["P_tx_dBm", "noise_figure_dB"],
        "assumed_note": ("P_tx and NF enter as ONE additive constant applied "
                         "identically to every arm; they set where the arm-SNR "
                         "distribution sits against the BLER waterfalls, hence "
                         "g* and the dead/informative/always-on split, but they "
                         "do not alter the geometry"),
        "channel_law": ("h(t) = sqrt(N) sum_l sigma_l eps_l(t) a(cos theta_l), "
                        "eps ~ CN(0,1) i.i.d. over l, links and rounds"),
        "channel_reference": ("El Ayach et al. 2014 -- sparse geometric channel "
                             "with zero-mean complex Gaussian path gains on "
                             "fixed AoDs; sigma_l^2 from the ray tracer instead "
                             "of assumed equal"),
        "normalisation": ("sqrt(N), NOT sqrt(N/L): beta already carry the "
                          "measured sigma_l^2, so 1/L would attenuate every "
                          "link by 10log10(L) and tie the link budget to how "
                          "many paths were kept. E||h||^2 = N sum_l sigma_l^2"),
        "fixed_over_horizon": ["L", "theta_l", "sigma_l^2", "UE positions",
                               "codebook"],
        "redrawn_per_round": ["eps_l(t)"],
        "snr_distribution": ("gamma ~ Exponential(mean gbar) on every beam; "
                             "sd of 10log10(gamma) is exactly "
                             "(10/ln10)(pi/sqrt6) = 5.570 dB for every arm"),
        "psi_method": ("closed form, INT [1-BLER_r(gbar x)] e^-x dx on a log "
                       "grid in x; no Monte Carlo, so g* carries no MC noise"),
        "n_cu": DM_N_CU, "q_b": DM_Q_B, "N_PRB_per_subchannel": DM_N_PRB,
        "B_sc_MHz": DM_B_SC / 1e6, "scs_kHz": DM_SCS_KHZ,
        "numerology_mu": 3, "slot_us": 125.0,
        "mcs_indices": DM_MCS_INDICES, "tbs_bits": DM_TBS_BITS,
        "rate_set": list(NR_RATE_SET),
        "bler_table": os.path.basename(BLER_TABLE_PATH),
        "bler_source": ("MEASURED Sionna 5G-NR LDPC, counted transport-block "
                        "errors; no fit and no anchor"),
        "bler_interp": ("log10(BLER) vs SNR[dB]; clamp to 1 below grid, "
                        "extrapolate last log-slope above, floor 1e-12"),
        "flat_fading_caveat": ("DeepMIMO `delay` is discarded and each 5.76 MHz "
                               "sub-channel treated as frequency-flat. This "
                               "OVERSTATES per-block SNR variance, so the bandit "
                               "faces more noise than reality would give it"),
        "regret_definition": "min(standard, satisficing)",
        "regret_definition_note": (
            "cum_regret uses min(max(0,g*-ach), max(0,tau-ach)), the paper's "
            "definition. sat_only_* is the pure satisficing form, which DIVERGES "
            "when tau > g* (accrues >= tau-g* per slot). std_* is standard regret."),
        "crn_note": ("one channel realization per slot shared by all methods; "
                     "decoding coins on a separate stream so the channel does "
                     "not shift when the method set changes"),
        "agent_rng_seeding": "np.random.seed((seed*7919+e) % 2**32) per experiment e",
    }


def git_provenance():
    """Commit, dirty flag and interpreter versions, or Nones outside a checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty,
            "python": sys.version.split()[0], "numpy": np.__version__}
