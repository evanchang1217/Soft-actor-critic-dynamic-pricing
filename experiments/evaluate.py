"""Evaluate frozen policies on held-out market seeds; no learning during evaluation."""

import argparse
import csv
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from rl.environment import MultiStoreEnv
from rl.sac import SACAgent, verify_checkpoint_updates
from rl.training import episode_summary
from .baselines import Baseline, METHODS
from .protocol import load_protocol, run_main


SUMMARY_METRICS = ("return", "mean_cpi", "cpi_std", "final_money", "total_sink", "total_volume", "mean_participation")


def sample_std(values):
    """Sample SD is undefined with fewer than two independent units."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def format_mean_std(mean, deviation):
    return f"{mean:.2f} ± {deviation:.2f}" if deviation is not None else f"{mean:.2f} (SD unavailable)"


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(episodes):
    """SAC: average scenarios per training seed first; baselines: one unit per scenario."""
    summaries = []
    for method in dict.fromkeys(row["method"] for row in episodes):
        rows = [row for row in episodes if row["method"] == method]
        units = {}
        for row in rows:
            key = row["training_seed"] if method in ("sac", "sac_fixed_fee") else row["evaluation_seed"]
            units.setdefault(key, []).append(row)
        summary = {"method": method, "aggregation_unit": "training_seed" if method in ("sac", "sac_fixed_fee") else "evaluation_seed",
                   "n_units": len(units), "n_episodes": len(rows)}
        for metric in SUMMARY_METRICS:
            values = [np.mean([row[metric] for row in unit]) for unit in units.values()]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = sample_std(values)
        summaries.append(summary)
    return summaries


def evaluate(market, protocol, checkpoints, output, methods=METHODS, *, allow_unverified=False):
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError(f"Choose distinct methods from {METHODS}")
    if "sac" in methods and not checkpoints:
        raise ValueError("SAC evaluation requires at least one trained checkpoint")
    torch.set_num_threads(1)
    env = MultiStoreEnv(market)
    models, model_records = [], []
    expected_config = None
    expected_steps = None
    known_updates = None
    for path in checkpoints if "sac" in methods else []:
        agent, checkpoint = SACAgent.load(path, env)
        verification = verify_checkpoint_updates(checkpoint, path, allow_unverified=allow_unverified)
        if checkpoint["seed"] in [record["training_seed"] for record in model_records]:
            raise ValueError("Checkpoints must have distinct training seeds")
        if expected_config is not None and agent.config.control_fee != expected_config.control_fee:
            raise ValueError("Cannot pool fixed-fee and fee-control SAC checkpoints; use experiments.review to compare them")
        if expected_config is not None and agent.config != expected_config:
            raise ValueError("Cannot pool SAC checkpoints with different training settings")
        if expected_steps is not None and checkpoint["trained_steps"] != expected_steps:
            raise ValueError("Cannot pool SAC checkpoints with different training budgets")
        if (verification["gradient_updates"] is not None and known_updates is not None
                and verification["gradient_updates"] != known_updates):
            raise ValueError("Cannot pool SAC checkpoints with different gradient-update counts")
        expected_config, expected_steps = agent.config, checkpoint["trained_steps"]
        if verification["gradient_updates"] is not None:
            known_updates = verification["gradient_updates"]
        # Catch leakage even when an independently trained checkpoint is supplied.
        used_seeds = {checkpoint["seed"] + ep * 10_000 for ep in range(1, checkpoint["sac"]["episodes"] + 1)}
        if set(protocol["evaluation_seeds"]) & (used_seeds | {checkpoint["seed"]}):
            raise ValueError("Evaluation seeds overlap this checkpoint's training seeds")
        models.append((agent, checkpoint["seed"]))
        model_records.append({"path": str(path), "training_seed": checkpoint["seed"],
                              "trained_steps": checkpoint["trained_steps"],
                              **verification,
                              "control_fee": agent.config.control_fee,
                              "sac_settings": asdict(agent.config),
                              "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    output = new_run_directory(output)
    write_json(output / "config.json", {**protocol, "methods": list(methods), "checkpoints": model_records})
    write_json(output / "manifest.json", manifest("held-out-evaluation"))
    episodes, steps = [], []
    for method in methods:
        policies = models if method == "sac" else [(None, "")]
        for agent, training_seed in policies:
            report_method = "sac_fixed_fee" if agent is not None and not agent.config.control_fee else method
            for seed in protocol["evaluation_seeds"]:
                observation, initial = env.reset(seed=seed)
                baseline = Baseline(method, env, seed=seed + 1_000_000, gain=protocol["adaptive_gain"]) if agent is None else None
                infos, rewards = [], []
                for step in range(1, market.max_steps + 1):
                    action = agent.act(observation, deterministic=True) if agent else baseline.act(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    rewards.append(reward)
                    infos.append(info)
                    steps.append({"method": report_method, "training_seed": training_seed, "evaluation_seed": seed,
                                  "step": step, "reward": reward,
                                  **{key: info[key] for key in ("cpi", "money", "sink", "trade_volume", "participation", "fee_rate",
                                                               "loss_count", "loss_volume", "loss_cpi", "loss_money")}})
                    if terminated or truncated:
                        break
                episodes.append({"method": report_method, "training_seed": training_seed, "evaluation_seed": seed,
                                 "initial_cpi": initial["cpi"], **episode_summary(infos, rewards)})
    summary = summarize(episodes)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "steps.csv", steps)
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", summary)
    env.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/default.json")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--allow-unverified-checkpoint", action="store_true",
                        help="Explicitly allow a checkpoint without an update record (results are marked unverified)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    market, _, protocol = load_protocol(args.config)
    summary = evaluate(market, protocol, args.checkpoints, args.out, args.methods,
                       allow_unverified=args.allow_unverified_checkpoint)
    for row in summary:
        print(f"{row['method']:8s} | return {format_mean_std(row['return_mean'], row['return_std'])} | CPI {row['mean_cpi_mean']:.3f}")


if __name__ == "__main__":
    run_main(main)
