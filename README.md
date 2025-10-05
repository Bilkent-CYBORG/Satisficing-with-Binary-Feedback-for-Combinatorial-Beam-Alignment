# Combinatorial Beam Alignment Simulation

This project implements and compares bandit algorithms for combinatorial beam alignment in wireless communication systems.

## Setup

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

2. Install dependencies:
```bash
pip install numpy<2  # NumPy 1.x required for PyTorch compatibility
pip install torch scipy matplotlib
```

## Running the Simulation

```bash
python3 main_for_sim.py
```

### Configuration

Edit `main_for_sim.py` to modify simulation parameters:

```python
T = 10000                    # Time horizon (number of rounds)
num_iterations = 100          # Number of experimental iterations
target_throughput = 10      # Target throughput (bits/symbol per user)
```

### Channel Files

Place channel data files in the `channel_data/` directory with naming format:
- `h_U{user_id}_B{bs_id}.mat` (e.g., `h_U1_B1.mat`, `h_U2_B1.mat`, etc.)

The system will auto-detect the number of users based on available channel files.

**Note**: Channel files are already organized in the `channel_data/` directory.

## Output

The simulation produces:
1. **Console output**: Progress tracking and final statistics
2. **Plot**: Comparison of CTS vs SAT-CTS showing satisficing & standard regret

## Algorithms Compared

- **CTS**: Combinatorial Thompson Sampling
- **SAT-CTS**: Satisficing Combinatorial Thompson Sampling with hierarchical decision cascade
- **CUCB**: Combinatorial Upper Confidence Bound
