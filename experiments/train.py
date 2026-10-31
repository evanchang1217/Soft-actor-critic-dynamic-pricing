"""Train one model: python -m experiments.train --help."""

import argparse

from rl.training import train
from .protocol import load_protocol, run_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/default.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    market, sac, _ = load_protocol(args.config)
    result = train(market, sac, args.seed, args.out,
                   callback=lambda row, info: print(f"Episode {row['episode']:3d} | return {row['return']:.2f} | CPI {row['mean_cpi']:.3f}"))
    print(f"Saved {result['steps']} steps / {result['updates']} gradient updates to {args.out}")
    if result["updates"] == 0:
        print("No parameter updates occurred; increase the training budget before evaluating this model.")


if __name__ == "__main__":
    run_main(main)
