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
from rl.rewards import reward_components
from rl.sac import SACAgent, verify_checkpoint_updates
from rl.training import episode_summary
from .baselines import Baseline, METHODS
from .protocol import load_protocol, run_main


SUMMARY_METRICS = ("return", "mean_cpi", "cpi_std", "final_money", "total_sink", "total_volume", "mean_participation")


def sample_std(values):
    """Sample SD is undefined with fewer than two independent units."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def market_summary(rows):
    """Average model draws within each market before summarizing markets."""
    groups = {}
    for row in rows:
        groups.setdefault(row["evaluation_seed"], []).append(row)
    if not groups:
        raise ValueError("At least one evaluation market is required")
    summary = {}
    for metric in ("return", "mean_cpi", "final_cpi", "total_volume", "final_money"):
        values = [float(np.mean([float(row[metric]) for row in unit])) for unit in groups.values()]
        summary[metric] = float(np.mean(values))
        summary[metric + "_market_sd"] = sample_std(values)
    return summary


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
            raise ValueError("Cannot pool fixed-fee and fee-control SAC checkpoints; evaluate each fee-control setting as a separate condition")
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



METRICS = ("common_return", "mean_gold_deviation", "max_gold_deviation", "total_volume",
           "mean_participation", "mean_cpi", "mean_price_change", "mean_item_price_change",
           "final_gold_ratio", "penalty_count", "penalty_volume", "penalty_cpi", "penalty_money")

class BlendPolicy:
    def __init__(self, agent, feedback, weight):
        if type(weight) not in (int, float) or not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Feedback weight must be between zero and one")
        self.agent, self.feedback, self.weight = agent, feedback, weight

    def act(self, observation):
        learned = self.agent.act(observation, deterministic=True)
        controlled = self.feedback.act(observation)
        action = (1-self.weight)*learned + self.weight*controlled
        action[-1] = 0
        env = self.feedback.env
        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def control_rollout(env, policy, seed):
    observation, _ = env.reset(seed=seed)
    target = env.config.num_players * env.config.initial_gold
    parts = {k: [] for k in ("count", "volume", "cpi", "money")}
    gold, volumes, participation, prices, changes, item_changes = [], [], [], [], [], []
    for _ in range(env.config.max_steps):
        previous = env.store_sell_prices.copy()
        action = policy.act(observation)
        if not env.action_space.contains(action) or action[-1] != 0:
            raise ValueError("Policy broke the common fixed-fee action constraints")
        observation, reward, terminated, truncated, info = env.step(action)
        penalties = reward_components(env.config, info, info["money"])
        if not np.isclose(reward, -sum(penalties.values()), rtol=0, atol=1e-12):
            raise ValueError("Reward reconstruction failed")
        for key, value in penalties.items():
            parts[key].append(value)
        gold.append(info["money"]/target)
        volumes.append(info["trade_volume"])
        participation.append(info["participation"])
        prices.append(info["cpi"])
        changes.append(info["loss_cpi"])
        item_changes.append(float(np.mean(np.abs(env.store_sell_prices/previous-1))))
        if terminated or truncated:
            break
    return {"common_return": -sum(sum(v) for v in parts.values()),
            "mean_gold_deviation": float(np.mean(np.abs(np.array(gold)-1))),
            "max_gold_deviation": float(np.max(np.abs(np.array(gold)-1))),
            "total_volume": float(sum(volumes)), "mean_participation": float(np.mean(participation)),
            "mean_cpi": float(np.mean(prices)), "mean_price_change": float(np.mean(changes)),
            "mean_item_price_change": float(np.mean(item_changes)), "final_gold_ratio": gold[-1],
            **{f"penalty_{k}": float(sum(v)) for k, v in parts.items()}}


def control_summary(rows):
    summaries = []
    for scenario, method in dict.fromkeys((r["scenario"], r["method"]) for r in rows):
        subset = [r for r in rows if (r["scenario"], r["method"]) == (scenario, method)]
        markets = sorted({r["evaluation_seed"] for r in subset})
        models = sorted({r["training_seed"] for r in subset if r["training_seed"] != ""})
        result = {"scenario": scenario, "method": method, "n_markets": len(markets), "n_models": len(models)}
        for metric in METRICS:
            values = [np.mean([r[metric] for r in subset if r["evaluation_seed"] == m]) for m in markets]
            result[metric+"_mean"] = float(np.mean(values))
            result[metric+"_market_sd"] = sample_std(values)
            model_values = [np.mean([r[metric] for r in subset if r["training_seed"] == s]) for s in models]
            result[metric+"_model_sd"] = sample_std(model_values)
        summaries.append(result)
    return summaries


def paired_interval(proposed, reference, metric, rng, resamples=10000):
    markets = sorted({r["evaluation_seed"] for r in proposed})
    models = sorted({r["training_seed"] for r in proposed})
    def matrix(rows):
        lookup = {(r["training_seed"], r["evaluation_seed"]): r[metric] for r in rows}
        if len(lookup) != len(rows):
            raise ValueError("Duplicate episode identity")
        row_models = sorted({r["training_seed"] for r in rows})
        if set(markets) != {r["evaluation_seed"] for r in rows}:
            raise ValueError("Paired methods need identical markets")
        if row_models != models and row_models != [""]:
            raise ValueError("Paired learned policies need matched training seeds")
        if len(lookup) != len(row_models)*len(markets):
            raise ValueError("Incomplete model by market grid")
        return np.array([[lookup[(s,m)] for m in markets] for s in row_models])
    difference = matrix(proposed)-matrix(reference)
    draws = []
    for start in range(0, resamples, 1000):
        count = min(1000, resamples-start)
        si = rng.integers(len(models), size=(count,len(models)))
        mi = rng.integers(len(markets), size=(count,len(markets)))
        draws.extend(difference[si[:,:,None],mi[:,None,:]].mean(axis=(1,2)))
    lo, hi = np.percentile(draws,[2.5,97.5])
    return {"metric": metric, "difference": float(difference.mean()),
            "exploratory_95_low": float(lo), "exploratory_95_high": float(hi)}


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
