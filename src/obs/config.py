"""Scenario, link budget and experiment knobs for the ray-traced simulator.

Everything here is read from the environment ONCE, at import, so a run's
configuration is fixed before any random number is drawn and can be written out
verbatim as the reproduction contract (see `obs.simulation.provenance`).

Only P_tx and the noise figure are assumed rather than measured. They enter as
one additive constant applied identically to every arm, so they set where the
arm-SNR distribution sits against the BLER waterfalls -- hence g* and the
dead/informative/always-on split -- but they never alter the geometry.
"""
import os

import numpy as np

MODEL_NAME = "deepmimo_el_ayach_zero_mean_beta"

DM_SCENARIO = os.environ.get("DM_SCENARIO", "city_3_houston_28")
DM_PTX_DBM = float(os.environ.get("PTX_DBM", 30.0))
DM_NF_DB = float(os.environ.get("NF_DB", 7.0))
DM_N_PRB, DM_SCS_KHZ = 4, 120.0
DM_B_SC = DM_N_PRB * 12 * DM_SCS_KHZ * 1e3
DM_NOISE_DBM = -174 + 10 * np.log10(DM_B_SC) + DM_NF_DB
DM_N_CU = 624
DM_Q_B = 8
DM_MCS_INDICES = [9, 15, 21, 24]
DM_TBS_BITS = [1544, 2472, 3496, 4096]
NR_RATE_SET = [2.4062, 3.9023, 5.5547, 6.5703]

DM_MIN_PATHS = int(os.environ.get("DM_MIN_PATHS", 1))

DM_REQUIRE_LOS = int(os.environ.get("DM_REQUIRE_LOS", 0))

DM_MIN_UE_SEP_M = float(os.environ.get("DM_MIN_UE_SEP_M", 0.0))

DM_EXTRA_UE_SEED = int(os.environ.get("DM_EXTRA_UE_SEED", 20260813))

DM_CANON = np.array([
    [79.1288, 21.0108], [21.1288, 17.0108], [7.12878, 48.0108],
    [-76.8712, 53.0108], [63.1288, 71.0108], [98.1288, -66.9892],
    [7.12878, -5.98922], [81.1288, 57.0108], [-58.8712, 57.0108],
    [-16.8712, 2.01078], [48.1288, -70.9892], [-32.8712, -42.9892],
    [-70.8712, -59.9892], [20.1288, -58.9892], [-58.8712, 47.0108]])

DM_RHO = float(os.environ.get("RHO", 0.0))
DM_P01 = float(os.environ.get("BLOCK_P01", 0.0))
DM_P10 = float(os.environ.get("BLOCK_P10", 0.005))
DM_BLOCK_DB = float(os.environ.get("BLOCK_DB", 20.0))
DM_INTERFERENCE = os.environ.get("INTERFERENCE", "0") not in ("0", "", "no")

_N_RF_ENV = os.environ.get("N_RF")


def rf_chains(num_users, num_bs):
    """Per-BS RF chains for a configuration.

    A fixed value cannot serve every configuration: the assignment needs
    M <= sum_b N_RF,b, so N_RF = 8 with B = 3 caps the system at 24 UEs. The
    rule ceil(1.5 * M / B) scales with the deployment and reproduces the
    nominal 8 exactly at M = 15, B = 3. The 1.5 factor matters: minimum
    feasibility alone, ceil(M / B), sits below the unconstrained per-BS loads
    at the optimum, so the cap would dictate the base-station split instead of
    constraining it. Setting N_RF overrides the rule; 0 disables the cap.
    """
    if _N_RF_ENV is not None:
        return int(_N_RF_ENV)
    return int(np.ceil(1.5 * num_users / num_bs))

DM_FEEDBACK_P = float(os.environ.get("FEEDBACK_P", 1.0))

