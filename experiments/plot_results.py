"""Regenerate figures entirely from saved CSV files (no retraining)."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .protocol import run_main


COLORS = {"fixed": "#64748b", "random": "#c98326", "adaptive": "#2478b5", "sac": "#237d65",
          "sac_fixed_fee": "#43a6a0"}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_results(directory):
    directory = Path(directory)
    rows = read_csv(directory / "steps.csv")
    if not rows:
        raise ValueError("No evaluation steps to plot")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    missing_sd = False
    panels = (("cpi", "Shop price index (relative to lower bounds)"),
              ("reward", "Reward per market step"),
              ("money", "Total player gold"),
              ("trade_volume", "Units traded per market step"))
    for ax, (metric, title) in zip(axes.flat, panels):
        for method in dict.fromkeys(row["method"] for row in rows):
            selected = [r for r in rows if r["method"] == method]
            x = sorted({int(r["step"]) for r in selected})
            means, deviations = [], []
            for step in x:
                by_unit = {}
                for row in selected:
                    if int(row["step"]) == step:
                        unit = row["training_seed"] if method in ("sac", "sac_fixed_fee") else row["evaluation_seed"]
                        by_unit.setdefault(unit, []).append(float(row[metric]))
                values = [np.mean(v) for v in by_unit.values()]
                means.append(np.mean(values))
                if len(values) > 1:
                    deviations.append(np.std(values, ddof=1))
                else:
                    deviations.append(np.nan)
                    missing_sd = True
            means, deviations = np.array(means), np.array(deviations)
            label = {"sac": "SAC", "sac_fixed_fee": "SAC (fixed fee)"}.get(method, method.title())
            ax.plot(x, means, label=label, color=COLORS[method])
            if np.isfinite(deviations).any():
                ax.fill_between(x, means-deviations, means+deviations, color=COLORS[method], alpha=0.12)
        ax.set(title=title, xlabel="Market step")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    suffix = " (where available)" if missing_sd else ""
    fig.suptitle("Held-out simulation evaluation · mean ± 1 standard deviation" + suffix, fontsize=13)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"comparison.{extension}", dpi=160)
    plt.close(fig)
    return directory / "comparison.png"


def plot_training(paths, output):
    histories = [read_csv(path) for path in paths]
    if not histories or any(not h for h in histories):
        return
    length = min(map(len, histories))
    values = np.array([[float(row["return"]) for row in h[:length]] for h in histories])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else None
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(1, length + 1)
    ax.plot(x, mean, color=COLORS["sac"])
    if std is not None:
        ax.fill_between(x, mean-std, mean+std, color=COLORS["sac"], alpha=0.15)
    ax.set(xlabel="Training episode", ylabel="Episode return",
           title=("SAC training · mean ± 1 SD across initialization seeds" if std is not None else
                  "SAC training · episode return (one initialization seed)"))
    ax.grid(alpha=0.2)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Evaluation directory containing steps.csv")
    args = parser.parse_args()
    print(plot_results(args.directory))


if __name__ == "__main__":
    run_main(main)
