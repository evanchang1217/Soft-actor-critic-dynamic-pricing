"""Rebuild follow-up comparison figures from CSVs, with common scenario-level SD."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .protocol import run_main
from .analyze_reward import open_step_log


COLORS = {"fixed": "#64748b", "random": "#c98326", "adaptive": "#2478b5",
          "tuned_adaptive": "#8163a6", "target_price": "#b34b51",
          "sac_fixed_fee": "#43a6a0", "sac": "#1c654f"}
LABELS = {"fixed": "Fixed", "random": "Random", "adaptive": "Adaptive (gain 0.5)",
          "tuned_adaptive": "Adaptive (validation-tuned)", "target_price": "Target-price rule",
          "sac_fixed_fee": "SAC (fixed fee)", "sac": "SAC (price + fee)"}


def save_figure(fig, path):
    for suffix in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=160)
    plt.close(fig)


def plot_review(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    labels = {**LABELS, "adaptive": f"Adaptive (gain {protocol['adaptive_gain']:g})"}
    trajectories = {}
    with open_step_log(directory / "steps.csv") as stream:
        for row in csv.DictReader(stream):
            if row["scenario"] != "nominal":
                continue
            markets = trajectories.setdefault(row["method"], {})
            markets.setdefault(row["evaluation_seed"], {}).setdefault(int(row["step"]), []).append(row)
    if not trajectories:
        raise ValueError("No nominal evaluation trajectories")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (("cpi", "Shop price index"), ("reward", "Reward per market step"),
              ("money", "Total player gold"), ("trade_volume", "Units traded per step"))
    for ax, (metric, title) in zip(axes.flat, panels):
        for method, markets in trajectories.items():
            times = sorted(next(iter(markets.values())))
            # Average model replicas within a market first. Every method's band
            # then uses the same held-out market-seed unit.
            values = np.array([[np.mean([float(r[metric]) for r in trajectory[t]]) for t in times]
                               for trajectory in markets.values()])
            mean = values.mean(axis=0)
            deviation = values.std(axis=0, ddof=1) if len(values) > 1 else None
            style = "--" if method == "sac_fixed_fee" else "-"
            ax.plot(times, mean, label=labels[method], color=COLORS[method], linestyle=style, linewidth=1.6)
            if deviation is not None:
                ax.fill_between(times, mean-deviation, mean+deviation, color=COLORS[method], alpha=0.07)
        ax.set(title=title, xlabel="Market step")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    title = "Fresh nominal test markets · mean ± 1 SD across markets (models averaged within market)"
    if any(len(markets) < 2 for markets in trajectories.values()):
        title += " · SD unavailable with one market"
    fig.suptitle(title, fontsize=11)
    save_figure(fig, directory / "comparison")

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    scenarios = list(dict.fromkeys(row["scenario"] for row in summary))
    methods = list(trajectories)
    matrix = np.array([[next(row["return_mean"] for row in summary
                             if row["method"] == method and row["scenario"] == scenario)
                        for scenario in scenarios] for method in methods])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    heatmap = ax.imshow(matrix, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(scenarios)), [s.replace("_", " ") for s in scenarios], fontsize=9)
    ax.set_yticks(range(len(methods)), [labels[m] for m in methods], fontsize=9)
    midpoint = (matrix.max() + matrix.min()) / 2
    for y in range(len(methods)):
        for x in range(len(scenarios)):
            ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center",
                    color="white" if matrix[y, x] > midpoint else "#20312c", fontsize=9)
    ax.set_title("Frozen strategies under explicit market shifts · mean episode return (higher is better)", fontsize=11, pad=15)
    fig.colorbar(heatmap, ax=ax, label="Mean episode return", shrink=0.8)
    save_figure(fig, directory / "robustness")
    return directory / "comparison.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(plot_review(parser.parse_args().directory))


if __name__ == "__main__":
    run_main(main)
