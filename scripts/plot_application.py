"""Plot saved paired differences for the application overview; no evaluation.

Run from the repository root. Archived research code, models and records are
read-only. The fresh output contains one figure and its input/source hashes.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "nominal": "Nominal",
    "higher_response": "Higher response",
    "lower_response": "Lower response",
    "resource_scarcity": "Resource scarcity",
    "unseen_higher_response": "Outside-center higher",
    "unseen_lower_response": "Outside-center lower",
}


def plot(record, output):
    record, output = Path(record), Path(output)
    config = json.loads((record / "config.json").read_text())
    decision = json.loads((record / "frozen-selection.json").read_text())
    with (record / "differences.csv").open(newline="") as stream:
        differences = list(csv.DictReader(stream))
    with (record / "summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    lookup = {(r["scenario"], r["comparison"], r["metric"]): r for r in differences}
    if len(lookup) != len(differences):
        raise ValueError("Duplicate comparison identity")
    means = {(r["scenario"], r["method"]): r for r in summaries}
    scenarios = [s["name"] for s in config["test_scenarios"]]
    feedback = decision["feedback_method"]
    families = (("nominal_w0", "Nominal training", "#4669a3", -0.12),
                ("randomized_w0", "Randomized training", "#147c65", 0.12))
    panels = (("common_return", 1, "Common-score difference", "Higher favors SAC"),
              ("mean_gold_deviation", 100, "Gold-deviation difference (percentage points)",
               "Lower favors SAC"))
    plotted = []
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), sharey=True)
    for ax, (metric, factor, xlabel, direction) in zip(axes, panels):
        ax.axvline(0, color="#68717a", linewidth=1, linestyle="--")
        for method, label, color, offset in families:
            for index, scenario in enumerate(scenarios):
                row = lookup[(scenario, method + "_minus_" + feedback, metric)]
                point, low, high = (factor * float(row[k]) for k in
                                    ("difference", "exploratory_95_low", "exploratory_95_high"))
                expected = factor * (float(means[(scenario, method)][metric + "_mean"])
                                     - float(means[(scenario, feedback)][metric + "_mean"]))
                if not all(map(math.isfinite, (point, low, high))) or low > high:
                    raise ValueError("Invalid saved interval")
                if abs(point - expected) > 1e-9:
                    raise ValueError("Saved difference disagrees with method summaries")
                y = index + offset
                ax.plot([low, high], [y, y], color=color, linewidth=2)
                ax.plot(point, y, "o", color=color, markersize=5,
                        label=label if index == 0 else None)
                plotted.append({"scenario": scenario, "method": method, "metric": metric,
                                "display_multiplier": factor, "difference": point,
                                "exploratory_95_low": low, "exploratory_95_high": high})
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(direction, fontsize=11)
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(scenarios)), [LABELS.get(s, s) for s in scenarios], fontsize=10)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.57, 0.93),
               ncol=2, fontsize=9, frameon=False)
    fig.suptitle("SAC versus validation-selected currency feedback", fontsize=14, y=0.97)
    fig.text(0.5, 0.025,
             f"Five models per SAC condition; 30k steps; {len(config['evaluation_seeds'])} paired initial markets per scenario.\n"
             f"Saved exploratory 95% crossed-bootstrap intervals ({config['bootstrap_resamples']:,} resamples); "
             "no multiplicity correction.\n"
             "One controller selected on the validation mixture is used in every test scenario; "
             "higher score does not imply lower currency error.",
             ha="center", va="bottom", fontsize=8.2, color="#424a52")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.80, bottom=0.23, wspace=0.18)
    output.mkdir(parents=True, exist_ok=False)
    figure = output / "control-comparison.png"
    fig.savefig(figure, dpi=180, facecolor="white")
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {"record": record.as_posix(), "operation": "plot saved paired intervals; no new evaluation",
                "reference_method": feedback,
                "input_sha256": {name: sha(record / name) for name in
                                 ("config.json", "frozen-selection.json", "differences.csv", "summary.csv")},
                "script": "scripts/plot_application.py", "script_sha256": sha(Path(__file__)),
                "figure_sha256": sha(figure), "plotted_values": plotted}
    (output / "control-comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="results/confirmation/2026-11-05-robustness")
    parser.add_argument("--out", required=True, help="Fresh output directory")
    args = parser.parse_args()
    plot(args.record, args.out)
