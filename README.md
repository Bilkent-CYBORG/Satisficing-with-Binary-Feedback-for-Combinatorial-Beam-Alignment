# SAT-CTS: Satisficing Combinatorial Thompson Sampling

Reference implementation for the paper
**"Satisficing with Binary Feedback for Combinatorial Beam Alignment"**.

This repository contains the simulation code, plotting scripts, and numerical
results used to produce the figures and tables in the paper.

## Repository layout

```
sat-cts-paper-code/
├── run_combinatorial_simulation.py   # entry point: runs SAT-CTS / SAT-CTS-W / CTS / CUCB
├── plot_results.py                   # regenerates the four figures from JSONs
├── results/
│   ├── real.json                     # realizable regime (τ_r = 8)
│   ├── non_real.json                 # non-realizable regime (τ_r = 25)
│   ├── combinatorial_15users_*.json  # 15-user run used for fairness figures
│   └── *.pdf                         # IEEE-style figures
└── src/obs/                          # supporting library (copied subset)
```

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

### Optional: LaTeX for publication-quality figures

`plot_results.py` renders text with LaTeX to match IEEE style. On macOS:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
```

If LaTeX is not available, set `"text.usetex": False` in
`plot_results.py` — figures will still render, just with Matplotlib's
built-in math text.

## Reproducing the results

```bash
# Regenerate figures from the included JSONs (fast)
python plot_results.py

# Re-run the full 15-user experiment (slow — several hours on a laptop)
python run_combinatorial_simulation.py
```

Results land in `results/` with a timestamp.

## Algorithms

| Name          | Description                                                               |
|---------------|---------------------------------------------------------------------------|
| `SAT-CTS`     | Proposed method with LCB → MEAN gate and committed CTS rounds              |
| `SAT-CTS-W`   | Workshop version (LCB → MEAN → UCB → TS gate, no doubling)                 |
| `CTS`         | Combinatorial Thompson Sampling (Wang & Chen, 2018)                        |
| `CUCB`        | Combinatorial UCB (Chen et al., 2013)                                      |

## Citation

If you use this code, please cite the paper:

```bibtex
@article{satcts2025,
  title   = {Satisficing with Binary Feedback for Combinatorial Beam Alignment},
  author  = {...},
  year    = {2025},
  journal = {...}
}
```

## License

MIT — see `LICENSE`.
