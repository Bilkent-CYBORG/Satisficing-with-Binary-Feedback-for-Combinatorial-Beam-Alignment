"""Generate IEEE-style plots from JSON results (SciencePlots-inspired)."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Style: science + ieee from SciencePlots (without requiring LaTeX)
# ------------------------------------------------------------------
matplotlib.rcParams.update({
    # Color cycle: SciencePlots default
    "axes.prop_cycle": plt.cycler("color",
        ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]),
    # Figure size: IEEE single column
    "figure.figsize": (3.3, 2.5),
    "figure.dpi": 600,
    # Font: serif, size 8 (IEEE)
    "font.size": 8,
    "axes.labelsize": 10,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    # LaTeX rendering
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    # Ticks: inward, both sides, minor visible
    "xtick.direction": "in",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    # Lines and axes
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    # Legend
    "legend.frameon": False,
    "legend.fontsize": 7,
    # Save
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi": 600,
})

PASTEL = True  # Set True for pastel, False for SciencePlots default

if PASTEL:
    COLORS = {
        "SAT-CTS":   "#6A9FD6",  # soft blue
        "SAT-CTS-W": "#E07B7B",  # soft red
        "CTS":       "#7BC88F",  # soft green
        "CUCB":      "#F0B860",  # soft orange
    }
else:
    COLORS = {
        "SAT-CTS":   "#0C5DA5",  # blue
        "SAT-CTS-W": "#FF2C00",  # red
        "CTS":       "#00B945",  # green
        "CUCB":      "#FF9500",  # orange
    }
MARKERS = {"SAT-CTS": "s", "SAT-CTS-W": "o", "CTS": "^", "CUCB": "D"}
LINESTYLES = {"SAT-CTS": "-", "SAT-CTS-W": "--", "CTS": ":", "CUCB": "-."}
ALGO_ORDER = ["SAT-CTS", "SAT-CTS-W", "CTS", "CUCB"]


def _plot_curve(ax, x, data, key_mean, key_std, ylabel, marker_every):
    for name in ALGO_ORDER:
        if name not in data["results"]:
            continue
        mean = np.array(data["results"][name][key_mean])
        std = np.array(data["results"][name][key_std])
        ax.plot(
            x, mean,
            color=COLORS[name], label=name, linewidth=1.0,
            linestyle=LINESTYLES[name], marker=MARKERS[name],
            markevery=marker_every, markersize=4,
            markerfacecolor="none", markeredgewidth=0.5,
            markeredgecolor=COLORS[name],
        )
        ax.fill_between(x, mean - std, mean + std, color=COLORS[name], alpha=0.15)
    ax.set_xlabel("Time Slot")
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_regret(json_path, save_path):
    with open(json_path) as f:
        data = json.load(f)
    T = data["config"]["T"]
    x = np.arange(0, T + 1)
    marker_every = max(1, T // 10)

    fig, ax = plt.subplots()
    _plot_curve(ax, x, data, "mean_cum_regret", "std_cum_regret",
                "Cumulative Satisficing Regret", marker_every)
    ax.legend(loc="upper left")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {save_path} + .pdf")


def plot_fairness(json_path, save_prefix):
    with open(json_path) as f:
        data = json.load(f)
    T = data["config"]["T"]
    x = np.arange(1, T + 1)
    marker_every = max(1, T // 10)

    # Jain's Fairness Index
    fig, ax = plt.subplots()
    _plot_curve(ax, x, data, "jain_mean", "jain_std",
                "Jain's Fairness Index", marker_every)
    ax.set_ylim(0, 1.05)
    plt.savefig(f"{save_prefix}_1.png")
    plt.savefig(f"{save_prefix}_1.pdf")
    plt.close()
    print(f"Saved: {save_prefix}_1.png + .pdf")

    # Log-Utility
    fig, ax = plt.subplots()
    _plot_curve(ax, x, data, "log_utility_mean", "log_utility_std",
                "Sum of Log Utilities", marker_every)
    plt.savefig(f"{save_prefix}_2.png")
    plt.savefig(f"{save_prefix}_2.pdf")
    plt.close()
    print(f"Saved: {save_prefix}_2.png + .pdf")


if __name__ == "__main__":
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    suffix = "_pastel" if PASTEL else ""
    plot_regret(f"{base}/real.json", f"{base}/regret{suffix}.png")
    plot_regret(f"{base}/non_real.json", f"{base}/non_realazible{suffix}.png")
    plot_fairness(f"{base}/combinatorial_15users_20260405_211244.json",
                  f"{base}/fairness{suffix}")
    print("Done!")
