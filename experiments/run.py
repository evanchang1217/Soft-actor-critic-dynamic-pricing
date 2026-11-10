"""Train all configured seeds, evaluate four policies, and generate result figures."""

import argparse

from rl.artifacts import manifest, new_run_directory, write_json
from rl.training import train
from .evaluate import evaluate, format_mean_std
from .plot_results import plot_results, plot_training
from .protocol import load_protocol, run_main


def run(config_path, output):
    market, sac, protocol = load_protocol(config_path)
    output = new_run_directory(output)
    write_json(output / "config.json", protocol)
    write_json(output / "manifest.json", manifest("comparison-suite"))
    checkpoints, histories = [], []
    for seed in protocol["training_seeds"]:
        directory = output / f"seed-{seed}"
        print(f"Training SAC seed {seed} ({sac.episodes} episodes)", flush=True)
        train(market, sac, seed, directory,
              callback=lambda row, info: print(f"  Episode {row['episode']:3d} | return {row['return']:.2f}", flush=True))
        checkpoints.append(directory / "policy.pt")
        histories.append(directory / "training.csv")
    print("Evaluating frozen policies on held-out market seeds", flush=True)
    summary = evaluate(market, protocol, checkpoints, output / "evaluation")
    plot_results(output / "evaluation")
    plot_training(histories, output / "training.png")
    for row in summary:
        print(f"{row['method']:8s} | return {format_mean_std(row['return_mean'], row['return_std'])} | CPI {row['mean_cpi_mean']:.3f}")
    print(f"Artifacts: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/quickstart.json")
    parser.add_argument("--out", required=True, help="New run directory")
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    run_main(main)
