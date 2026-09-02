"""One static UE on ray-traced geometry, with fading redrawn each round.

The angles and average path powers come from the ray tracer and never change.
Only the complex path gains beta_l(t) are redrawn, which is El Ayach's sparse
geometric channel with zero-mean complex Gaussian gains:

    h_{m,b}(t) = sqrt(N) sum_l sigma_l eps_l(t) a(cos theta_l),  eps ~ CN(0,1)

Because eps is zero-mean, h ~ CN(0, R) with rank(R) <= L out of N, so
|h^H f|^2 is exponential on every beam and psi has a closed form.
"""
import numpy as np

from obs.config import (
    DM_BLOCK_DB,
    DM_NOISE_DBM,
    DM_P01,
    DM_P10,
    DM_PTX_DBM,
    DM_RHO,
)
from obs.environment.deepmimo_geometry import dft_codebook, load_deepmimo_geometry


class BeamUser:
    """One static UE. Ray-traced geometry fixed; only the path gains fade.

    ``paths[b] = (cos_theta (L,), sigma2 (L,) LINEAR)`` straight from DeepMIMO.
    There is no per-BS reference SNR to normalise against: the ray tracer gives
    an absolute channel gain, so
        SNR_dB = P_tx[dBm] + 10log10(|h^H f|^2) - noise[dBm].
    """

    def __init__(self, user_id, N, K, num_bs, paths, rng=None,
                 rho=0.0, p01=0.0, p10=0.0, block_db=None):
        self.uid = user_id
        self.N, self.K, self.num_bs = N, K, num_bs
        self.rng = rng or np.random.default_rng()
        self.F = dft_codebook(N, K)
        self.paths = paths
        self._scale = 10 ** ((DM_PTX_DBM - DM_NOISE_DBM) / 10.0)
        self.rho = float(rho)
        self.p01 = float(p01)
        self.p10 = float(p10)
        self._block_gain = 10 ** (-(DM_BLOCK_DB if block_db is None
                                    else block_db) / 10.0)
        self._eps_state = [None] * num_bs
        self._blocked = [False] * num_bs

        self._A, self._s2 = [], []
        self.mean_snr = np.zeros(num_bs * K)
        for b in range(num_bs):
            cos_th, s2 = paths[b]
            self._s2.append(s2)
            if len(cos_th) == 0:
                self._A.append(np.zeros((N, 0), complex))
                continue
            A = np.exp(1j * np.pi * np.arange(N)[:, None] * cos_th[None, :])
            A /= np.sqrt(N)
            self._A.append(A)
            sf = N * ((s2[None, :] * np.abs(self.F.conj().T @ A) ** 2).sum(1))
            self.mean_snr[b * K:(b + 1) * K] = self._scale * sf

    def _draw_eps(self, bs, Lb):
        """eps_l(t) for one BS, either i.i.d. or AR(1) with coefficient rho.

        AR(1):  eps(t) = rho*eps(t-1) + sqrt(1-rho^2) w(t),  w ~ CN(0,1).

        The sqrt(1-rho^2) is what makes this useful: the MARGINAL law of eps is
        CN(0,1) for every rho, so gamma stays exponential with the same mean,
        psi and g* are unchanged, and any difference in regret is attributable
        to temporal correlation alone rather than to an easier or harder
        channel. rho=0 recovers the i.i.d. model; rho->1 approaches frozen.
        """
        w = (self.rng.normal(size=Lb) + 1j * self.rng.normal(size=Lb)) / np.sqrt(2.0)
        if self.rho <= 0.0:
            return w
        prev = self._eps_state[bs]
        if prev is None or len(prev) != Lb:
            self._eps_state[bs] = w
            return w
        cur = self.rho * prev + np.sqrt(1.0 - self.rho ** 2) * w
        self._eps_state[bs] = cur
        return cur

    def _step_blockage(self, bs):
        """Two-state Markov blockage on link (this UE, bs). Returns linear gain.

        mmWave links are intermittently blocked by bodies and vehicles rather
        than fading gracefully, so this is modelled as a discrete on/off process
        per link, not as extra fading: p01 enters blockage, p10 leaves it, and
        while blocked the whole BS contribution is attenuated by DM_BLOCK_DB.
        Steady-state blocked probability is p01/(p01+p10); mean blocked run is
        1/p10 slots.
        """
        if self.p01 <= 0.0:
            return 1.0
        if self._blocked[bs]:
            if self.rng.random() < self.p10:
                self._blocked[bs] = False
        elif self.rng.random() < self.p01:
            self._blocked[bs] = True
        return self._block_gain if self._blocked[bs] else 1.0

    def block_snr(self):
        """Per-beam beamformed SNR (linear) for THIS slot.

        Returns (num_bs*K,) aligned with global beam index bs*K + k. Only
        beta_l(t) is redrawn; theta_l and sigma_l^2 are fixed for the horizon.
        """
        out = np.zeros(self.num_bs * self.K)
        for bs in range(self.num_bs):
            s2 = self._s2[bs]
            Lb = len(s2)
            if Lb == 0:
                continue
            blk = self._step_blockage(bs)
            beta_t = np.sqrt(s2) * self._draw_eps(bs, Lb)
            h = np.sqrt(self.N) * (self._A[bs] @ beta_t)
            gains = np.abs(self.F.conj().T @ h) ** 2
            out[bs * self.K:(bs + 1) * self.K] = self._scale * blk * gains
        return out


def build_users(num_users, num_bs, N, K, rng, n_paths=3,
                rho=None, p01=None, p10=None):
    """Place the UEs on live grid points and build their arm sets.

    Each user gets its OWN generator, spawned from `rng`, so the per-slot
    channel stream is independent of how many decoding coins the methods
    consume. That is what keeps the channel identical when the method set
    changes. Returns ``(users, meta)``.
    """
    geo, meta = load_deepmimo_geometry(num_users, num_bs, n_paths)
    seeds = rng.spawn(num_users) if hasattr(rng, "spawn") else [None] * num_users
    users = [BeamUser(u, N, K, num_bs,
                      paths=[geo[(u, b)] for b in range(num_bs)],
                      rng=seeds[u] or np.random.default_rng(),
                      rho=DM_RHO if rho is None else rho,
                      p01=DM_P01 if p01 is None else p01,
                      p10=DM_P10 if p10 is None else p10)
             for u in range(num_users)]
    return users, meta
