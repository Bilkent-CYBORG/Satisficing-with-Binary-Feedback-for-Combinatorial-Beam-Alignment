# SAT-CTS: Satisficing Combinatorial Thompson Sampling

Reference implementation for the paper
**"Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits"**.

This repository contains the simulation code and plotting scripts used to produce the figures and tables in the paper.



## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install this package and its dependencies
pip install -e .
```

The `deepmimo` package automatically downloads the `city_3_houston_28` scenario
the first time the simulation runs.



## Reproducing the results

```bash
# Run the full 15-user experiment (slow — several hours on a laptop)
python run_combinatorial_simulation.py

# Generate IEEE-style figures from the output JSONs
python plot_results.py
```

Results land in `results/` with a timestamp.

**Note:** `plot_results.py` uses LaTeX for text rendering. On macOS: `brew install --cask basictex`. If LaTeX is unavailable, set `"text.usetex": False` in `plot_results.py`.

## Algorithms

| Name          | Description                                                               |
|---------------|---------------------------------------------------------------------------|
| `SAT-CTS`     | Proposed method with LCB → MEAN gate and committed CTS rounds              |
| `SAT-CTS-W`   | Workshop version (LCB → MEAN → UCB → TS gate, no doubling)                 |
| `CTS`         | Combinatorial Thompson Sampling (Wang & Chen, 2018)                        |
| `CUCB`        | Combinatorial UCB (Chen et al., 2013)                                      |



## License

MIT — see `LICENSE`.
