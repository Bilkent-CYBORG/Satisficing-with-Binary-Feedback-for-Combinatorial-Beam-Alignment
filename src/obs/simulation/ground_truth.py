"""Ground-truth psi and the clairvoyant benchmark g*.

Both are computed WITHOUT Monte Carlo. That matters because every regret curve
is measured against g*: sampling noise in the benchmark would show up as noise
in every method's regret, indistinguishable from the thing being measured.
"""
import numpy as np

from obs.config import DM_BLOCK_DB
from obs.simulation.bler import success_prob


def estimate_psi(users, rate_set, p_block=None, block_db=None):
    """Ground-truth psi, closed form -- no Monte Carlo.

    gamma is EXACTLY exponential (h ~ CN(0,R) => |h^H f|^2 ~ Exp), so writing
    gamma = gbar*x with x ~ Exp(1),

        psi = INT_0^inf [1 - BLER_r(gbar*x)] e^-x dx.

    Integrated on a LOG grid in x because the waterfall sits at
    x = gamma_th/gbar, which is ~1e-4 for a strong arm: a linear grid (or
    scipy.integrate.quad) steps straight over that sliver and returns psi = 1
    exactly.
    """
    tb = users[0].num_bs * users[0].K
    R = len(rate_set)
    psi = np.zeros((len(users), tb, R))
    xs = np.logspace(-9, 2.0, 24001)
    w = np.exp(-xs)
    if p_block is None:
        p_block = (users[0].p01 / (users[0].p01 + users[0].p10)
                   if users[0].p01 > 0 else 0.0)
    att = 10 ** (-(DM_BLOCK_DB if block_db is None else block_db) / 10.0)
    for u, usr in enumerate(users):
        gbar = usr.mean_snr
        for k in range(tb):
            if gbar[k] <= 0:
                continue
            for r in range(R):
                v = np.trapz(success_prob(10 * np.log10(gbar[k] * xs),
                                          rate_set[r]) * w, xs)
                if p_block > 0:
                    vb = np.trapz(success_prob(10 * np.log10(gbar[k] * att * xs),
                                               rate_set[r]) * w, xs)
                    v = (1.0 - p_block) * v + p_block * vb
                psi[u, k, r] = v
    return psi


def optimal_throughput(psi, rate_set, beam_to_bs=None, cap=None):
    """Clairvoyant per-UE throughput under the assignment constraint.

    If ``beam_to_bs``/``cap`` are given, the benchmark obeys the SAME per-BS
    RF-chain cap the algorithms do. Otherwise regret would be measured against
    an infeasible reference. (Measured on this geometry the cap does not bind
    at the optimum -- load [5,3,7] against N_RF=8 -- so g* is unchanged; the
    argument still has to be made rather than assumed.)
    """
    from scipy.optimize import linear_sum_assignment
    val = rate_set[None, None, :] * psi
    best_r = val.argmax(axis=2)
    score = np.take_along_axis(val, best_r[..., None], axis=2)[..., 0]
    if cap is not None and len(cap):
        _, ci = linear_sum_assignment(-score)
        load = np.bincount(np.asarray(beam_to_bs)[ci], minlength=len(cap))
        if (load > np.asarray(cap)).any():
            from obs.utils.utils import capacitated_best
            ci, _ = capacitated_best(val, np.asarray(beam_to_bs),
                                     np.asarray(cap))
            ci = np.asarray(ci)
    else:
        _, ci = linear_sum_assignment(-score)
    return sum(rate_set[best_r[u, ci[u]]] * psi[u, ci[u], best_r[u, ci[u]]]
               for u in range(len(psi))) / len(psi)
