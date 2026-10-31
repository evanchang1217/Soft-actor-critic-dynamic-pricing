"""Audit weighted reward components from recorded steps without retraining."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from rl.artifacts import new_run_directory, write_json
from .evaluate import write_csv
from .protocol import run_main


COMPONENTS = ("count", "volume", "cpi", "money")


def open_step_log(path):
    """Published logs may be gzip-compressed; local run CSVs stay easy to inspect."""
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".gz").is_file():
        path = path.with_suffix(path.suffix + ".gz")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def reward_analysis(steps_path, market):
    weights = {name: market[f"reward_{name}"] for name in COMPONENTS}
    groups = {}
    with open_step_log(steps_path) as stream:
        for row in csv.DictReader(stream):
            key = (row.get("scenario", "nominal"), row["method"])
            group = groups.setdefault(key, {"n_steps": 0, **{name: 0.0 for name in COMPONENTS}})
            penalties = {name: weights[name] * float(row[f"loss_{name}"]) for name in COMPONENTS}
            if not math.isclose(-float(row["reward"]), sum(penalties.values()), rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("Saved reward does not match recorded component losses and configured weights")
            if not all(math.isfinite(value) and value >= 0 for value in penalties.values()):
                raise ValueError("Reward penalties must be finite and nonnegative")
            group["n_steps"] += 1
            for name, value in penalties.items():
                group[name] += value
    if not groups:
        raise ValueError("No evaluation steps to analyze")
    rows = []
    for (scenario, method), group in groups.items():
        total = sum(group[name] for name in COMPONENTS)
        row = {"scenario": scenario, "method": method, "n_steps": group["n_steps"],
               "mean_total_penalty": total / group["n_steps"]}
        for name in COMPONENTS:
            row[f"{name}_mean_penalty"] = group[name] / group["n_steps"]
            row[f"{name}_share_percent"] = 100 * group[name] / total if total else 0.0
        rows.append(row)
    return rows


def write_reward_analysis(steps_path, market, output):
    rows = reward_analysis(steps_path, market)
    write_csv(Path(output) / "reward_components.csv", rows)
    write_json(Path(output) / "reward_components.json", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Evaluation directory with steps.csv and config.json")
    parser.add_argument("--out", required=True, help="New analysis directory; input artifacts remain unchanged")
    args = parser.parse_args()
    directory = Path(args.directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    output = new_run_directory(args.out)
    rows = write_reward_analysis(directory / "steps.csv", config["market"], output)
    for row in rows:
        trade = row["count_share_percent"] + row["volume_share_percent"]
        print(f"{row['scenario']:16s} {row['method']:16s} | trade-target penalty share {trade:.4f}%")


if __name__ == "__main__":
    run_main(main)
