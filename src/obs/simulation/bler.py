"""Measured 5G-NR LDPC block-error rate, and the ACK probability it implies.

Counted transport-block errors from gen_nr_bler_table.py (Sionna 1.2.1,
TS 38.212 LDPC + rate matching, 20-iteration BP, exact APP demapping, complex
AWGN, n_cu = 624). No fit, no anchor, no analytic waterfall -- in particular no
Shannon threshold anywhere in the chain.
"""
import json
import os

import numpy as np

_BLER_GRID = None
_DEFAULT_TABLE = "nr_bler_table_v2.json"


def _default_table_path():
    """Locate the shipped BLER table without assuming the working directory.

    BLER_TABLE wins if set. Otherwise the current directory is tried first (so
    a run in a checkout picks up that checkout's table), then the repository
    root inferred from this file's location, which is what an editable install
    resolves to.
    """
    env = os.environ.get("BLER_TABLE")
    if env:
        return env
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(here))))
    for cand in (os.path.join(os.getcwd(), _DEFAULT_TABLE),
                 os.path.join(repo_root, _DEFAULT_TABLE)):
        if os.path.exists(cand):
            return cand
    return os.path.join(repo_root, _DEFAULT_TABLE)


BLER_TABLE_PATH = _default_table_path()


def _load_bler_grid(path=None):
    """{r_bits: (snr_db asc, log10 BLER asc-in-snr)} from the measured table."""
    with open(path or BLER_TABLE_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    grid = {}
    for m in doc["mcs"].values():
        pts = sorted(m["points"], key=lambda p: p["snr_db"])
        snr = np.array([p["snr_db"] for p in pts], float)
        bler = np.clip(np.array([p["bler"] for p in pts], float), 1e-12, 1.0)
        grid[round(float(m["spectral_efficiency"]), 4)] = (snr, np.log10(bler))
    return grid


def measured_bler(snr_db, r_bits, grid=None):
    """BLER by log-domain interpolation of the measured table.

    BLER spans six decades, so log10(BLER) is interpolated against SNR in dB --
    linear interpolation of BLER itself would be meaningless. Below the grid the
    result is clamped to 1 (nothing decodes); above it the last log-slope is
    extrapolated with a 1e-12 floor, rather than pinning to the last point,
    which would understate the reliability of very strong arms.
    """
    global _BLER_GRID
    if grid is None:
        if _BLER_GRID is None:
            _BLER_GRID = _load_bler_grid()
        grid = _BLER_GRID
    key = round(float(r_bits), 4)
    if key not in grid:
        raise KeyError(f"no measured BLER curve for r={r_bits} "
                       f"(have {sorted(grid)})")
    snr, logb = grid[key]
    x = np.atleast_1d(np.asarray(snr_db, float))
    out = np.interp(x, snr, logb)
    out = np.where(x < snr[0], 0.0, out)
    if len(snr) >= 2 and snr[-1] > snr[-2]:
        slope = (logb[-1] - logb[-2]) / (snr[-1] - snr[-2])
        hi = x > snr[-1]
        out = np.where(hi, logb[-1] + slope * (x - snr[-1]), out)
    b = np.clip(10.0 ** out, 1e-12, 1.0)
    return b if np.ndim(snr_db) else float(b[0])


def measured_success_prob(snr_db, r_bits, grid=None):
    """psi = 1 - BLER from the measured table."""
    return 1.0 - measured_bler(snr_db, r_bits, grid)


success_prob = measured_success_prob
