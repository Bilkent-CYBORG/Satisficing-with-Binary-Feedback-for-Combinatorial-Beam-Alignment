# SAT-CTS: Satisficing Combinatorial Thompson Sampling

Reference implementation for the paper
**"Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits"**.

A base station picks a *beam* and a *transmission rate* per user and observes
only one **ACK/NACK bit** per user per slot. The goal is not to maximise
throughput but to *satisfice*: reach a target rate `τ_r` and stop exploring.
Regret is measured against that target rather than against the optimum.

## The model

Large-scale geometry comes from ray tracing, small-scale fading is the only
random object, and the ACK statistics come from a measured 5G-NR LDPC decoder.
The DeepMIMO `city_3_houston_28` scene supplies the `L = 3` strongest paths'
angles and powers per (UE, BS) link, fixed for the horizon; on top of them
`h(t) = √N Σ_ℓ σ_ℓ ε_ℓ(t) a(cos θ_ℓ)` is redrawn each round with
`ε ~ CN(0,1)`, so `γ = (P_tx/σ²_noise)|hᴴf|²` is exponential on every beam and
`ψ = P(ACK)` — read off a measured Sionna NR-LDPC BLER table at `n_cu = 624` —
has a closed form, which is what lets `g*` be computed without Monte-Carlo
noise. The setup is `M = 15` UEs, `B = 3` BSs, `N = 64`-element ULAs and
`K = 120` DFT beams over `R = 4` rates from TS 38.214 MCS Table 2, i.e. 21,600
base arms; assignment is Hungarian on the rate-collapsed `M × BK` matrix, or a
capacitated transportation LP when the per-BS RF-chain cap binds. `P_tx = 30`
dBm and `NF = 7` dB are the only assumed numbers and enter as a single additive
constant on every arm.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The `deepmimo` package downloads `city_3_houston_28` on first use (~200 MB).

## Running

```bash
# One threshold, all methods. Writes a gzipped JSON with a full config block.
SIM_T=10000 SIM_ITERS=15 SIM_TARGET=4.88 python dm_sim.py

# The τ ladder as independent processes — the only parallel split that
# preserves common random numbers, since all methods must share one channel
# stream per slot.
for tau in 2.79 3.90 4.74 5.29 5.85; do
  TAU=$tau N_EXP=15 OUT_DIR=results/dm_ladder python run_experiment_dm.py &
done; wait
```

Results land in `results/`. Every run records its own reproduction contract —
link budget, UE placement report, BLER table identity, git commit and dirty
flag — so a run is reproducible from its own output.

### Configuration

Everything is set by environment variable and read once, at import, before any
random number is drawn.

| variable | default | meaning |
|---|---|---|
| `SIM_T`, `SIM_ITERS`, `SIM_TARGET` | 10000, 5, 4.55 | horizon, independent runs, target `τ_r` |
| `SIM_USERS`, `SIM_BS`, `SIM_K`, `SIM_N_ANT` | 15, 3, 120, 64 | `M`, `B`, beams per BS, array size |
| `SIM_METHODS`, `SIM_SEED`, `OUT_DIR` | all, 0, `results` | method list, base seed, output directory |
| `PTX_DBM`, `NF_DB` | 30, 7 | transmit power [dBm] and receiver noise figure [dB] |
| `BLER_TABLE` | `nr_bler_table_v2.json` | which measured decoder table to use |
| `N_RF` | 8 | per-BS RF-chain cap; `0` disables it |
| `FEEDBACK_P` | 1.0 | ACK/NACK feedback reliability |
| `DM_MIN_PATHS` | 1 | minimum ray-traced paths per link at a UE position |
| `DM_REQUIRE_LOS` | 0 | minimum LoS links a UE position must have |
| `DM_MIN_UE_SEP_M` | 0 | minimum spacing between placed UEs [m] |
| `RHO` | 0 | AR(1) coefficient on `ε` (0 = i.i.d. block fading) |
| `BLOCK_P01`, `BLOCK_P10`, `BLOCK_DB` | 0, 0.005, 20 | two-state Markov blockage per link |
| `INTERFERENCE` | 0 | drop the `q_b` orthogonal split for spatial reuse |
| `UNCAPPED_METHODS` | *(empty)* | methods exempt from the RF-chain cap; empty means every method is capped |

## Layout

| path | role |
|---|---|
| `dm_sim.py` | command line entry point |
| `run_experiment_dm.py` | one-τ-per-process driver; writes `metrics_tau*.json` |
| `obs/config.py` | scenario, link budget, every environment knob |
| `obs/environment/deepmimo_geometry.py` | ray-traced angles and powers, DFT codebook |
| `obs/user/beam_user.py` | the per-round fading channel |
| `obs/simulation/bler.py` | measured NR-LDPC BLER → `P(ACK)` |
| `obs/simulation/ground_truth.py` | closed-form `ψ` and the oracle `g*` |
| `obs/simulation/methods.py` | which agents exist and how they are built |
| `obs/simulation/dm_simulation.py` | the round loop: CRN, regret, QoS metrics |
| `obs/simulation/provenance.py` | the reproduction contract |
| `obs/algorithms/combinatorial/` | the agents and the assignment oracle |
| `plot_results.py` | generic plotter for the gzipped JSON |

`obs/environment/{environment,codebook,channel_provider*}.py`,
`obs/user/user.py` and `obs/simulation/{combinatorial_simulation,regret,
reward,simulation}.py` are the older full-MIMO `DeepMIMOProvider` path. They are
kept as a library but are **not** the pipeline the paper uses.

## Algorithms

| Name | Description |
|---|---|
| `SAT-CTS` | Proposed. LCB → MEAN gate, committed CTS rounds on a `2^i` doubling schedule, fresh `Beta(1,1)` priors per round. |
| `SAT-CTS-Retain` | Same, but the global posterior is retained across rounds. |
| `CTS` | Combinatorial Thompson Sampling (Wang & Chen, 2018). |
| `CUCB` | Combinatorial UCB (Chen et al., 2013), `√(3 log t / 2n)`, unplayed arms first. |

## License

MIT — see `LICENSE`.
