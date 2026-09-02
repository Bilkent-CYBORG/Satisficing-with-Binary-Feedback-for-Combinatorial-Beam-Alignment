"""Ray-traced geometry and the DFT beam codebook.

The ray tracer supplies the LARGE-SCALE part of the channel and nothing else:
per (UE, BS) link, the angles of departure and average powers of the L
strongest paths. Those are fixed for the whole horizon. Small-scale fading is
added on top of them by `obs.user.beam_user.BeamUser`.
"""
import numpy as np

from obs.config import (
    DM_CANON,
    DM_EXTRA_UE_SEED,
    DM_MIN_PATHS,
    DM_MIN_UE_SEP_M,
    DM_REQUIRE_LOS,
    DM_SCENARIO,
)

_DM_CACHE = {}


def dft_codebook(N, K):
    """K DFT beams uniform in cos(theta) over [-1,1); unit-norm columns.

    Unit norm means a beam supplies directivity, not extra power. The `pi` in
    the exponent IS 2*pi*d/lambda at d = lambda/2.
    """
    cos_grid = np.linspace(-1, 1, K, endpoint=False)
    return np.exp(1j * np.pi * np.arange(N)[:, None] * cos_grid[None, :]) / np.sqrt(N)


def load_deepmimo_geometry(num_users, num_bs, n_paths, min_paths=None,
                           require_los=None, min_sep_m=None):
    """Ray-traced (cos_theta, sigma_l^2) per (UE, BS), with no dead links.

    Returns ``(geo, meta)`` where ``geo[(u, bs)] = (cos_theta (L,), sigma2 (L,))``
    with sigma2 LINEAR and L = min(n_paths, paths available on that link).

    Placement: DeepMIMO offers ~31k grid positions of which ~25k are active, so
    a UE with no path to some BS is a placement artefact, not a property of the
    scene. Each nominal position in ``DM_CANON`` is therefore snapped to the
    nearest grid point that has >= ``min_paths`` paths from EVERY BS. Snapping is
    greedy and exclusive, so two UEs never land on the same grid point.

    ``meta`` records what the snap cost (distances, per-link path counts, LoS
    mix) so the placement can be reported rather than assumed.

    Cached: the load is slow and the geometry is a property of the scene, not of
    the run, so it is identical across seeds and experiments.
    """
    min_paths = DM_MIN_PATHS if min_paths is None else int(min_paths)
    require_los = DM_REQUIRE_LOS if require_los is None else int(require_los)
    min_sep_m = DM_MIN_UE_SEP_M if min_sep_m is None else float(min_sep_m)
    key = (num_users, num_bs, n_paths, min_paths, DM_SCENARIO,
           require_los, min_sep_m)
    if key in _DM_CACHE:
        return _DM_CACHE[key]

    import deepmimo as dm
    d = dm.load(DM_SCENARIO)
    P = np.asarray(d.power)
    A = np.asarray(d.aod_az)
    NPth = np.asarray(d.num_paths)
    RX = np.asarray(d.rx_pos)
    LOS = np.asarray(d.los)

    live = np.all(NPth[:num_bs] >= min_paths, axis=0)
    if require_los > 0:
        live &= (LOS[:num_bs] == 1).sum(axis=0) >= require_los
    cand = np.flatnonzero(live)
    if cand.size < num_users:
        raise RuntimeError(
            f"only {cand.size} grid points have >= {min_paths} paths from all "
            f"{num_bs} BSs and >= {require_los} LoS links; cannot place "
            f"{num_users} UEs")
    xy = RX[0][:, :2]

    def far_enough(g, chosen):
        """Separation test against the points already placed."""
        if min_sep_m <= 0.0 or not chosen:
            return True
        return bool(np.min(np.linalg.norm(
            xy[np.asarray(chosen)] - xy[g], axis=1)) >= min_sep_m)

    idx, snap_m, taken = [], [], set()
    for c in DM_CANON[:num_users]:
        order = np.argsort(np.linalg.norm(xy[cand] - c, axis=1))
        placed = False
        for o in order:
            g = int(cand[o])
            if g not in taken and far_enough(g, idx):
                taken.add(g)
                idx.append(g)
                snap_m.append(float(np.linalg.norm(xy[g] - c)))
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"no candidate for nominal position {c} satisfies "
                f"min_sep_m={min_sep_m}; relax it or use fewer UEs")

    if num_users > len(DM_CANON):
        pool = np.array([g for g in cand if g not in taken])
        perm = np.random.default_rng(DM_EXTRA_UE_SEED).permutation(pool)
        need = num_users - len(DM_CANON)
        extra = []
        for g in perm:
            if far_enough(int(g), idx + extra):
                extra.append(int(g))
                if len(extra) == need:
                    break
        if len(extra) < need:
            raise RuntimeError(
                f"only {len(extra)} of {need} extra UEs satisfy "
                f"min_sep_m={min_sep_m}")
        for g in extra:
            taken.add(int(g))
            idx.append(int(g))
            snap_m.append(float("nan"))

    geo = {}
    for ui, u in enumerate(idx):
        for bs in range(num_bs):
            n = int(NPth[bs, u])
            sel = np.argsort(-P[bs, u, :n])[:min(n_paths, n)]
            geo[(ui, bs)] = (np.cos(np.deg2rad(A[bs, u, sel])),
                             10 ** (P[bs, u, sel] / 10.0))

    counts = [len(geo[(u, b)][0]) for u in range(num_users) for b in range(num_bs)]
    kept_frac = []
    for ui, u in enumerate(idx):
        for bs in range(num_bs):
            n = int(NPth[bs, u])
            all_p = 10 ** (P[bs, u, :n] / 10.0)
            kept_frac.append(float(geo[(ui, bs)][1].sum() / all_p.sum()))
    meta = {
        "scenario": DM_SCENARIO,
        "grid_positions": int(RX.shape[1]),
        "live_grid_positions": int(cand.size),
        "min_paths_required": min_paths,
        "require_los_links": require_los,
        "min_ue_separation_m": min_sep_m,
        "ue_los_links": [int((LOS[:num_bs, u] == 1).sum()) for u in idx],
        "ue_grid_index": [int(i) for i in idx],
        "ue_pos_m": [[float(x) for x in xy[i]] for i in idx],
        "snap_distance_m": snap_m,
        "snap_distance_max_m": float(np.nanmax(snap_m)),
        "n_sampled_ue": int(sum(1 for x in snap_m if x != x)),
        "links": len(counts),
        "links_with_zero_paths": int(sum(c == 0 for c in counts)),
        "links_below_L": int(sum(c < n_paths for c in counts)),
        "paths_per_link_min": int(min(counts)),
        "paths_available_median": float(np.median(
            [int(NPth[b, u]) for u in idx for b in range(num_bs)])),
        "kept_power_fraction_median": float(np.median(kept_frac)),
        "los_links": int(sum(int(LOS[b, u]) == 1
                             for u in idx for b in range(num_bs))),
    }
    _DM_CACHE[key] = (geo, meta)
    return geo, meta
