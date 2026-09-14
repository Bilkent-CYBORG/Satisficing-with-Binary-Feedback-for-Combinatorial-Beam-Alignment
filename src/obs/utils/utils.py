"""Utility functions for OBS library.

This module provides core utility functions used across the library including:
- Action encoding/decoding
- RSS computation
- Uncertainty ball operations
- Robust satisficing action selection
- Hungarian algorithm for optimal matching
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances


def encode_action(pos, bs_idx, cb_entry_complex, bs_size, rx_entry=None):
    """Encode position, BS index, TX beam, and optional RX beam into action vector.

    Parameters
    ----------
    pos : np.ndarray
        Position array (x, y, z).
    bs_idx : int
        Base station index.
    cb_entry_complex : np.ndarray
        Complex TX codebook entry of shape (N,).
    bs_size : int
        Number of base stations.
    rx_entry : np.ndarray, optional
        Complex RX codebook entry of shape (M,). None for MISO.

    Returns
    -------
    np.ndarray
        Encoded action vector. Shape (3 + bs_size + 2N) for MISO,
        or (3 + bs_size + 2N + 2M) for MIMO.
    """
    bs_one_hot = np.zeros(bs_size)
    bs_one_hot[bs_idx] = 1.0
    parts = [pos, bs_one_hot, np.real(cb_entry_complex), np.imag(cb_entry_complex)]
    if rx_entry is not None:
        parts.extend([np.real(rx_entry), np.imag(rx_entry)])
    return np.concatenate(parts)


def encode_action_compact(pos, bs_pos, beam_angle, rx_beam_angle=None):
    """Encode position, BS position, and beam angle into compact action vector.

    Parameters
    ----------
    pos : np.ndarray
        User position array (x, y, z).
    bs_pos : np.ndarray
        Base station position array (x, y, z).
    beam_angle : float
        TX beam angle in degrees (from arcsin formula).
    rx_beam_angle : float, optional
        RX beam angle in degrees. None for MISO.

    Returns
    -------
    np.ndarray
        Compact action vector. Shape (7,) for MISO, (8,) for MIMO.
    """
    parts = [np.asarray(pos), np.asarray(bs_pos), np.array([beam_angle])]
    if rx_beam_angle is not None:
        parts.append(np.array([rx_beam_angle]))
    return np.concatenate(parts)


def uniform_ball_over_grid(x_hat_t, pos_subset, radius):
    """Compute uniform ball over grid.

    Parameters
    ----------
    x_hat_t : np.ndarray
        Target position.
    pos_subset : np.ndarray
        Subset of positions.
    radius : float
        Radius of the ball.

    Returns
    -------
    np.ndarray
        Probability distribution over the subset.
    """
    dists = np.linalg.norm(pos_subset - x_hat_t, axis=1)
    mask = dists <= radius
    p = np.zeros(len(pos_subset))
    k = mask.sum()
    if k == 0:
        p[:] = 1.0 / len(pos_subset)
    else:
        p[mask] = 1.0 / k
    return p


def find_x_rs_action_RS1_diabetes_contextual_all_kappas(
    C: np.ndarray,
    Y: np.ndarray,
    ind_ref_cho: int,
    tau: float,
    dist_metric=euclidean_distances,
    pow=1,
):
    """Find the action with the highest RS value.

    Parameters
    ----------
    C : np.ndarray
        Context array with shape (#contexts, context_dim).
    Y : np.ndarray
        Value array with shape (#actions, #contexts).
    ind_ref_cho : int
        Index of reference context.
    tau : float
        Threshold value.
    dist_metric : callable, optional
        Distance metric function. Default is euclidean_distances.
    pow : int, optional
        Power value. Default is 1.

    Returns
    -------
    np.ndarray
        kappa values for all actions.
    """
    X_less_than_tau_index = np.where(Y[:, ind_ref_cho] < tau)[0]
    X_more_than_tau_index = np.where(Y[:, ind_ref_cho] >= tau)[0]
    N_x = Y.shape[0]
    kappa_values = np.inf * np.ones(N_x)

    if len(X_less_than_tau_index) == N_x:
        return kappa_values

    kappa_values[X_more_than_tau_index] = -np.inf
    kappa_values[X_less_than_tau_index] = np.inf

    distances = dist_metric(C[ind_ref_cho].reshape(1, -1), C) + 1e-6
    distances = np.power(distances, pow)

    for i, index_more in enumerate(X_more_than_tau_index):
        kappa_x_temp = (tau - Y[index_more, :]) / distances.flatten()
        kappa_values[index_more] = np.max(kappa_x_temp)

    kappa_values = np.power(kappa_values, 1 / pow)
    return kappa_values


def find_x_rs_action_RS1_diabetes_contextual(
    C: np.ndarray,
    Y: np.ndarray,
    ind_ref_cho: int,
    tau: float,
    dist_metric=euclidean_distances,
    pow=1,
):
    """Find the action with the highest RS value.

    Parameters
    ----------
    C : np.ndarray
        Context array with shape (#contexts, context_dim).
    Y : np.ndarray
        Value array with shape (#actions, #contexts).
    ind_ref_cho : int
        Index of reference context.
    tau : float
        Threshold value.
    dist_metric : callable, optional
        Distance metric function. Default is euclidean_distances.
    pow : int, optional
        Power value. Default is 1.

    Returns
    -------
    tuple
        Tuple of (index_x_rs, kappa_x_rs) where index_x_rs is the action index
        and kappa_x_rs is the action value.
    """
    X_less_than_tau_index = np.where(Y[:, ind_ref_cho] < tau)[0]
    X_more_than_tau_index = np.where(Y[:, ind_ref_cho] >= tau)[0]
    N_x = Y.shape[0]
    kappa_values = np.inf * np.ones(N_x)

    if len(X_less_than_tau_index) == N_x:
        return np.random.choice(N_x), np.inf

    kappa_values[X_more_than_tau_index] = -np.inf
    kappa_values[X_less_than_tau_index] = np.inf

    distances = dist_metric(C[ind_ref_cho].reshape(1, -1), C) + 1e-6
    distances = np.power(distances, pow)

    for i, index_more in enumerate(X_more_than_tau_index):
        kappa_x_temp = (tau - Y[index_more, :]) / distances.flatten()
        kappa_values[index_more] = np.max(kappa_x_temp)

    index_x_rs = np.argmin(kappa_values)
    kappa_x_rs = np.power(kappa_values[index_x_rs], 1 / pow)
    return index_x_rs, kappa_x_rs


def get_closest_grid_index(position: np.ndarray, grid_positions: np.ndarray) -> int:
    """Get the index of the closest grid position to the given position.

    Parameters
    ----------
    position : np.ndarray
        Target position.
    grid_positions : np.ndarray
        Array of grid positions.

    Returns
    -------
    int
        Index of the closest grid position.
    """
    distances = np.linalg.norm(grid_positions - position, axis=1)
    return np.argmin(distances)


def get_uncertainty_ball_indices(
    center_pos: np.ndarray, pos_subset: np.ndarray, eps: float, max_positions: int = 20
) -> np.ndarray:
    """Return indices of positions from pos_subset within radius eps of center_pos.

    Parameters
    ----------
    center_pos : np.ndarray
        Center position of the ball, shape (pos_dim,).
    pos_subset : np.ndarray
        Array of positions to search, shape (num_pos, pos_dim).
    eps : float
        Radius of the uncertainty ball.
    max_positions : int, optional
        Maximum number of indices to return. Default is 20.

    Returns
    -------
    np.ndarray
        Array of indices, shape (n,) where n <= max_positions.
    """
    distances = np.linalg.norm(pos_subset - center_pos, axis=1)
    mask = distances <= eps
    indices = np.where(mask)[0]
    if len(indices) == 0:
        # Return index of closest position
        return np.array([np.argmin(distances)])
    if len(indices) > max_positions:
        indices = np.random.choice(indices, size=max_positions, replace=False)
    return indices


def get_uncertainty_ball_positions(
    center_pos: np.ndarray, pos_subset: np.ndarray, eps: float, max_positions: int = 20
) -> np.ndarray:
    """Return positions from pos_subset within radius eps of center_pos.

    If there are more than max_positions, sample a subset.
    If none, return [center_pos] itself.

    Parameters
    ----------
    center_pos : np.ndarray
        Center position of the ball, shape (pos_dim,).
    pos_subset : np.ndarray
        Array of positions to search, shape (num_pos, pos_dim).
    eps : float
        Radius of the uncertainty ball.
    max_positions : int, optional
        Maximum number of positions to return. Default is 20.

    Returns
    -------
    np.ndarray
        Array of positions within the ball, shape (n, pos_dim) where n <= max_positions.
    """
    indices = get_uncertainty_ball_indices(center_pos, pos_subset, eps, max_positions)
    return pos_subset[indices]


def hungarian_best(values: np.ndarray) -> Tuple[List[int], List[int]]:
    """Find optimal user-beam-rate matching using Hungarian algorithm.

    Given a 3D array of values (e.g., expected throughput), find the optimal
    assignment of beams to users such that each user gets a unique beam and
    the sum of values is maximized.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (num_users, total_beams, num_rates) containing values.

    Returns
    -------
    tuple
        Tuple of (chosen_beams, chosen_rates) where:
        - chosen_beams: List of beam indices, one per user
        - chosen_rates: List of rate indices, one per user
    """
    U, Btot, R = values.shape
    # For each user-beam pair, find best rate
    best_r = values.argmax(axis=2)  # (U, Btot)
    scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(
        -1
    )  # (U, Btot)
    # Hungarian algorithm minimizes cost, so negate scores
    row_ind, col_ind = linear_sum_assignment(-scores)
    # Build result lists
    chosen_beams = [int(col_ind[u]) for u in range(U)]
    chosen_rates = [int(best_r[u, col_ind[u]]) for u in range(U)]
    return chosen_beams, chosen_rates


def capacitated_best(
    values: np.ndarray, beam_to_bs: np.ndarray, cap: np.ndarray
) -> Tuple[List[int], List[int]]:
    """Optimal matching subject to a per-BS RF-chain cap.

    Same as :func:`hungarian_best`, but additionally enforces the constraint
    that appears in the paper's super-arm set ``S``::

        |{m : b_m = b}| <= N_RF,b   for every BS b

    Solved exactly as a transportation LP.  The constraint structure is a flow
    network (UE -> beam -> BS -> sink), so the constraint matrix is totally
    unimodular and the LP optimum is attained at an integral vertex; no
    rounding is involved.  A Lagrangian penalty on the BS load was tried first
    and does NOT work -- a uniform per-BS penalty never breaks ties between
    users, so they herd onto the same BS and the multiplier diverges.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (num_users, total_beams, num_rates).
    beam_to_bs : np.ndarray
        Length ``total_beams``; ``beam_to_bs[b]`` is the BS owning beam ``b``.
    cap : np.ndarray
        Length ``num_bs``; per-BS RF-chain capacity.

    Returns
    -------
    tuple
        ``(chosen_beams, chosen_rates)``, one entry per user.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, vstack

    U, Btot, _ = values.shape
    nbs = len(cap)
    if int(cap.sum()) < U:
        raise ValueError(
            f"infeasible: total RF chains {int(cap.sum())} < users {U}"
        )

    best_r = values.argmax(axis=2)
    scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)

    nvar = U * Btot
    # one beam per user (equality)
    rows = np.repeat(np.arange(U), Btot)
    cols = np.arange(nvar)
    A_eq = csr_matrix((np.ones(nvar), (rows, cols)), shape=(U, nvar))

    # each beam used at most once
    beam_rows = np.tile(np.arange(Btot), U)
    A_beam = csr_matrix((np.ones(nvar), (beam_rows, cols)), shape=(Btot, nvar))
    # per-BS RF-chain cap
    bs_rows = np.tile(beam_to_bs, U)
    A_bs = csr_matrix((np.ones(nvar), (bs_rows, cols)), shape=(nbs, nvar))

    res = linprog(
        c=-scores.ravel(),
        A_ub=vstack([A_beam, A_bs]).tocsr(),
        b_ub=np.concatenate([np.ones(Btot), np.asarray(cap, dtype=float)]),
        A_eq=A_eq,
        b_eq=np.ones(U),
        bounds=(0, 1),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"capacitated assignment LP failed: {res.message}")

    x = res.x.reshape(U, Btot)
    chosen_beams = [int(np.argmax(x[u])) for u in range(U)]
    # Totally unimodular => the vertex is integral. Verify rather than assume.
    if not np.allclose([x[u, chosen_beams[u]] for u in range(U)], 1.0, atol=1e-6):
        raise RuntimeError("capacitated assignment LP returned a fractional vertex")
    chosen_rates = [int(best_r[u, chosen_beams[u]]) for u in range(U)]
    return chosen_beams, chosen_rates


def shared_best(values: np.ndarray) -> Tuple[List[int], List[int]]:
    """Optimal assignment when beams are SHARABLE and BS load is unconstrained.

    Dropping the one-beam-per-UE constraint removes the only coupling between
    UEs, so the problem separates: each UE independently takes the
    (beam, rate) pair maximizing its own expected throughput. No matching is
    needed, and this is exact rather than a relaxation.

    A beam is a spatial filter, not a resource. Under the default model each BS
    divides its carrier into q_b orthogonal sub-channels, so two UEs served on
    the same beam occupy different time-frequency resources and do not
    interfere; beam exclusivity is a modelling choice, not a physical
    requirement.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (num_users, total_beams, num_rates).

    Returns
    -------
    tuple
        ``(chosen_beams, chosen_rates)``, one entry per user. Beam indices may
        repeat.
    """
    U, Btot, R = values.shape
    flat = values.reshape(U, Btot * R).argmax(axis=1)
    chosen_beams = (flat // R).astype(int).tolist()
    chosen_rates = (flat % R).astype(int).tolist()
    return chosen_beams, chosen_rates


def shared_cap_best(
    values: np.ndarray, beam_to_bs: np.ndarray, cap: np.ndarray
) -> Tuple[List[int], List[int]]:
    """Beams sharable, but each BS serves at most ``cap[b]`` UEs per slot.

    With beams free the only contested resource is the RF chain, so this is a
    transportation problem on UE -> BS alone. Replicating BS ``b`` into
    ``cap[b]`` interchangeable slots turns it into a square assignment problem
    over ``U x sum(cap)``, which the Hungarian algorithm solves exactly; no LP
    and no rounding are involved.

    Within its assigned BS a UE simply takes its best (beam, rate) pair, since
    beams carry no capacity of their own.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (num_users, total_beams, num_rates).
    beam_to_bs : np.ndarray
        Length ``total_beams``; ``beam_to_bs[k]`` is the BS owning beam ``k``.
    cap : np.ndarray
        Length ``num_bs``; per-BS RF-chain capacity.

    Returns
    -------
    tuple
        ``(chosen_beams, chosen_rates)``, one entry per user.
    """
    U, Btot, R = values.shape
    beam_to_bs = np.asarray(beam_to_bs)
    cap = np.asarray(cap, dtype=int)
    nbs = len(cap)
    if int(cap.sum()) < U:
        raise ValueError(
            f"infeasible: total RF chains {int(cap.sum())} < users {U}"
        )

    best_r = values.argmax(axis=2)
    scores = np.take_along_axis(values, best_r[..., None], axis=2).squeeze(-1)

    # Best beam per (UE, BS), and its score.
    masks = [beam_to_bs == b for b in range(nbs)]
    beam_of_bs = np.empty((U, nbs), dtype=int)
    score_of_bs = np.empty((U, nbs))
    for b, m in enumerate(masks):
        idx = np.flatnonzero(m)
        local = scores[:, idx].argmax(axis=1)
        beam_of_bs[:, b] = idx[local]
        score_of_bs[:, b] = scores[:, idx][np.arange(U), local]

    slots = np.repeat(np.arange(nbs), cap)
    row_ind, col_ind = linear_sum_assignment(-score_of_bs[:, slots])
    bs_of_ue = np.empty(U, dtype=int)
    bs_of_ue[row_ind] = slots[col_ind]

    chosen_beams = [int(beam_of_bs[u, bs_of_ue[u]]) for u in range(U)]
    chosen_rates = [int(best_r[u, chosen_beams[u]]) for u in range(U)]
    return chosen_beams, chosen_rates
