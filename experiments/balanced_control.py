"""Equal-budget reward ablation and validation-tuned controls under a common score."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import gzip
import hashlib
import io
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from rl.config import MarketConfig, integer
from rl.environment import MultiStoreEnv
from rl.rewards import reward_components
from rl.sac import SACAgent, verify_checkpoint_updates
from rl.training import train
from .analyze_reward import write_reward_analysis, open_step_log
from .baselines import Baseline, CurrencyFeedbackPolicy, FixedStepAdaptive, TargetPricePolicy
from .evaluate import sample_std, write_csv
from .protocol import resolve_protocol, run_main
from .common import snapshot_research


LEARNED = ("legacy_sac", "trade_only_sac", "balanced_sac")
BASELINES = ("fixed", "random_fixed_fee", "adaptive", "tuned_adaptive", "fixed_step_adaptive",
             "target_price", "currency_feedback")
METHODS = BASELINES + LEARNED
METRICS = ("common_return", "mean_gold_deviation", "max_gold_deviation", "final_gold_ratio",
           "total_volume", "mean_participation", "mean_cpi", "mean_price_change", "legacy_return")
REWARD_FIELDS = {"reward_mode", "reward_count", "reward_volume", "reward_cpi", "reward_money"}
SHIFT_FIELDS = {"shock_step", "shock_multiplier", "price_sensitivity_low", "price_sensitivity_high"}


def load_config(path):
    raw = json.loads(Path(path).read_text())
    fields = {"description", "market", "sac", "training_seeds", "evaluation_seeds", "validation_seeds",
              "adaptive_gains", "adaptive_magnitudes", "target_buy_fractions", "target_sell_fractions",
              "currency_gains", "scenarios", "bootstrap_seed", "bootstrap_resamples"}
    # Resolved reports include the shared protocol's compatibility field.
    # Accept them as inputs as well as the compact, authored configurations.
    if fields - set(raw) or set(raw) - fields - {"adaptive_gain"}:
        raise ValueError("Balanced-control configuration has missing or unknown fields")
    market, sac, protocol = resolve_protocol({k: raw[k] for k in
        ("description", "market", "sac", "training_seeds", "evaluation_seeds", "adaptive_gain") if k in raw})
    if market.reward_mode != "balanced_v1" or sac.control_fee:
        raise ValueError("Use balanced_v1 scoring and fixed-fee SAC for this comparison")
    if min(len(protocol["training_seeds"]), len(protocol["evaluation_seeds"])) < 2:
        raise ValueError("At least two training seeds and test markets are required")
    used = {s + ep * 10000 for s in protocol["training_seeds"] for ep in range(sac.episodes + 1)}
    validation = raw["validation_seeds"]
    if not isinstance(validation, list) or len(validation) < 2:
        raise ValueError("At least two validation markets are required")
    for seed in validation:
        integer(seed, "validation seed", 0)
    if len(set(validation)) != len(validation) or set(validation) & (used | set(protocol["evaluation_seeds"])):
        raise ValueError("Training, validation and test markets must be distinct")
    for field in ("adaptive_gains", "adaptive_magnitudes", "target_buy_fractions", "target_sell_fractions", "currency_gains"):
        values = raw[field]
        if (not isinstance(values, list) or not values
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Invalid candidate grid: {field}")
        for value in values:
            if "fractions" in field:
                valid = 0 <= value <= 1
            elif field == "adaptive_magnitudes":
                valid = 0 < value <= market.price_offset
            else:
                valid = value > 0
            if not valid:
                raise ValueError(f"Out-of-range candidate: {field}")
    if not raw["scenarios"] or raw["scenarios"][0] != {"name": "nominal", "changes": {}}:
        raise ValueError("First scenario must be nominal")
    names = set()
    for scenario in raw["scenarios"]:
        if not isinstance(scenario, dict) or set(scenario) != {"name", "changes"}:
            raise ValueError("Scenarios require name and changes")
        name, changes = scenario["name"], scenario["changes"]
        if not isinstance(name, str) or not name.isidentifier() or name in names or not isinstance(changes, dict):
            raise ValueError("Invalid or duplicate scenario")
        if set(changes) - SHIFT_FIELDS:
            raise ValueError("Scenario changes must preserve state/action scales and the scoring rule")
        replace(market, **changes)
        names.add(name)
    integer(raw["bootstrap_seed"], "bootstrap seed", 0)
    integer(raw["bootstrap_resamples"], "bootstrap resamples", 100)
    return market, sac, {**raw, **protocol}


def training_market(market, method):
    if method == "legacy_sac":
        return replace(market, reward_mode="legacy", reward_count=1, reward_volume=1,
                       reward_cpi=0.5, reward_money=0.5)
    if method == "trade_only_sac":
        return replace(market, reward_money=0)
    if method == "balanced_sac":
        return market
    raise ValueError("Unknown learned method")


def baseline(method, env, seed, parameters):
    if method in ("fixed", "random_fixed_fee", "adaptive"):
        return Baseline("random" if method == "random_fixed_fee" else method, env, seed=seed + 1000000)
    if method == "tuned_adaptive":
        return Baseline("adaptive_proportional", env, **parameters)
    if method == "fixed_step_adaptive":
        return FixedStepAdaptive(env, **parameters)
    if method == "target_price":
        return TargetPricePolicy(env, **parameters)
    if method == "currency_feedback":
        return CurrencyFeedbackPolicy(env, **parameters)
    raise ValueError("Unknown baseline")


def rollout(env, policy, seed):
    observation, initial = env.reset(seed=seed)
    target = env.config.num_players * env.config.initial_gold
    old = replace(env.config, reward_mode="legacy", reward_count=1, reward_volume=1, reward_cpi=0.5, reward_money=0.5)
    steps = []
    for step in range(1, env.config.max_steps + 1):
        action = policy.act(observation, deterministic=True) if isinstance(policy, SACAgent) else policy.act(observation)
        action = action.copy()
        action[-1] = 0  # equal fixed-fee action space, including Random
        observation, reward, terminated, truncated, info = env.step(action)
        components = reward_components(env.config, info, info["money"])
        if not np.isclose(reward, -sum(components.values()), rtol=0, atol=1e-12):
            raise ValueError("Reward reconstruction failed")
        gold_ratio = info["money"] / target
        steps.append({"step": step, "reward": reward, "gold_ratio": gold_ratio,
                      "gold_deviation": abs(gold_ratio - 1),
                      "legacy_reward": -sum(reward_components(old, info).values()),
                      **{k: info[k] for k in ("cpi", "money", "sink", "trade_volume", "participation", "fee_rate",
                                             "loss_count", "loss_volume", "loss_cpi", "loss_money")},
                      **{f"penalty_{k}": v for k, v in components.items()}})
        if not np.isclose(info["fee_rate"], env.config.initial_fee, rtol=0, atol=1e-8):
            raise ValueError("Fee changed during a fixed-fee comparison")
        if terminated or truncated:
            break
    summary = {"initial_cpi": initial["cpi"], "common_return": sum(r["reward"] for r in steps),
               "mean_gold_deviation": float(np.mean([r["gold_deviation"] for r in steps])),
               "max_gold_deviation": max(r["gold_deviation"] for r in steps),
               "final_gold_ratio": steps[-1]["gold_ratio"],
               "total_volume": sum(r["trade_volume"] for r in steps),
               "mean_participation": float(np.mean([r["participation"] for r in steps])),
               "mean_cpi": float(np.mean([r["cpi"] for r in steps])),
               "mean_price_change": float(np.mean([r["loss_cpi"] for r in steps])),
               "legacy_return": sum(r["legacy_reward"] for r in steps)}
    return summary, steps


def select_baselines(market, protocol):
    env = MultiStoreEnv(market)
    specifications = [("tuned_adaptive", {"gain": v}) for v in protocol["adaptive_gains"]]
    specifications += [("fixed_step_adaptive", {"magnitude": v}) for v in protocol["adaptive_magnitudes"]]
    pairs = list(itertools.product(protocol["target_buy_fractions"], protocol["target_sell_fractions"]))
    specifications += [("target_price", {"buy_fraction": b, "sell_fraction": s}) for b, s in pairs]
    specifications += [("currency_feedback", {"buy_fraction": b, "sell_fraction": s, "gain": g})
                       for (b, s), g in itertools.product(pairs, protocol["currency_gains"])]
    candidates, trials = [], []
    for index, (method, parameters) in enumerate(specifications):
        values = []
        for seed in protocol["validation_seeds"]:
            summary, _ = rollout(env, baseline(method, env, seed, parameters), seed)
            values.append(summary["common_return"])
            trials.append({"candidate": index, "method": method, "evaluation_seed": seed, **summary})
        candidates.append({"candidate": index, "method": method, "parameters": parameters,
                           "validation_common_return": float(np.mean(values))})
    selected = {method: max((r for r in candidates if r["method"] == method),
                           key=lambda r: r["validation_common_return"])
                for method in ("tuned_adaptive", "fixed_step_adaptive", "target_price", "currency_feedback")}
    env.close()
    return {"metric": "mean nominal validation common_return", "candidates": candidates, "selected": selected}, trials


def _train_job(market, sac, seed, output):
    from rl.config import SACConfig
    result = train(MarketConfig.from_dict(market), SACConfig(**sac), seed, output)
    if result["cancelled"] or result["updates"] <= 0:
        raise ValueError("Model did not complete training")
    return result["steps"], result["updates"]


def prepare_models(market, sac, protocol, output, workers, reuse_root):
    jobs = []
    if reuse_root is None:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            for method in LEARNED:
                for seed in protocol["training_seeds"]:
                    folder = output / "models" / method / f"seed-{seed}"
                    future = pool.submit(_train_job, training_market(market, method).to_dict(), asdict(sac), seed, str(folder))
                    jobs.append((future, method, seed))
            identities = {f: (m, s) for f, m, s in jobs}
            for future in as_completed(identities):
                steps, updates = future.result()
                method, seed = identities[future]
                print(f"Trained {method} seed {seed}: {steps} steps, {updates} updates", flush=True)
    models, records = {}, []
    for method in LEARNED:
        models[method] = []
        for seed in protocol["training_seeds"]:
            folder = output / "models" / method / f"seed-{seed}"
            if reuse_root is not None:
                source = Path(reuse_root) / "models" / method / f"seed-{seed}"
                shutil.copytree(source, folder)
            path = folder / "policy.pt"
            agent, meta = SACAgent.load(path, MultiStoreEnv(training_market(market, method)))
            verified = verify_checkpoint_updates(meta, path)
            if (meta["seed"] != seed or meta["trained_steps"] != sac.episodes * market.max_steps
                    or asdict(agent.config) != asdict(sac)):
                raise ValueError("Checkpoint seed, training budget or settings differ from protocol")
            models[method].append((seed, agent))
            records.append({"method": method, "training_seed": seed, "trained_steps": meta["trained_steps"],
                            **verified, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "path": path.relative_to(output).as_posix(), "training_market": meta["market"]})
    return models, records


def evaluate_markets(market, protocol, selection, models, seeds, scenarios, output, prefix):
    episodes = []
    with (output / f"{prefix}-steps.csv.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = None
                for scenario in scenarios:
                    env = MultiStoreEnv(replace(market, **scenario["changes"]))
                    for method in METHODS:
                        policies = models[method] if method in LEARNED else [("", None)]
                        for training_seed, agent in policies:
                            if agent is not None:
                                # Only reward and named market shifts differ. Encoding,
                                # products, player count and all action bounds stay identical.
                                before, after = agent.env.config.to_dict(), env.config.to_dict()
                                if any(before[k] != after[k] for k in before.keys() - REWARD_FIELDS - SHIFT_FIELDS):
                                    raise ValueError("Evaluation changes observation/action semantics")
                                agent.env = env
                            for seed in seeds:
                                policy = agent if agent is not None else baseline(method, env, seed,
                                    selection["selected"].get(method, {}).get("parameters", {}))
                                summary, steps = rollout(env, policy, seed)
                                identity = {"scenario": scenario["name"], "method": method,
                                            "training_seed": training_seed, "evaluation_seed": seed}
                                episodes.append({**identity, **summary})
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=[*identity, *steps[0]])
                                    writer.writeheader()
                                writer.writerows({**identity, **r} for r in steps)
                    env.close()
                    print(f"Evaluated {prefix}: {scenario['name']}", flush=True)
    write_csv(output / f"{prefix}-episodes.csv", episodes)
    return episodes


def summarize(episodes):
    result = []
    for scenario, method in dict.fromkeys((r["scenario"], r["method"]) for r in episodes):
        rows = [r for r in episodes if (r["scenario"], r["method"]) == (scenario, method)]
        markets, models = {}, {}
        for row in rows:
            markets.setdefault(row["evaluation_seed"], []).append(row)
            if method in LEARNED:
                models.setdefault(row["training_seed"], []).append(row)
        entry = {"scenario": scenario, "method": method, "n_markets": len(markets), "n_models": len(models)}
        for metric in METRICS:
            values = [np.mean([r[metric] for r in group]) for group in markets.values()]
            model_values = [np.mean([r[metric] for r in group]) for group in models.values()]
            entry[metric + "_mean"] = float(np.mean(values))
            entry[metric + "_market_sd"] = sample_std(values)
            entry[metric + "_model_sd"] = sample_std(model_values)
        result.append(entry)
    return result


def uncertainty(episodes, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    seeds, markets = sorted(protocol["training_seeds"]), sorted(protocol["evaluation_seeds"])
    records = {(r["scenario"], r["method"], r["training_seed"], r["evaluation_seed"]): r for r in episodes}
    if len(records) != len(episodes):
        raise ValueError("Duplicate episode identity")
    result = []
    for scenario in protocol["scenarios"]:
        sc = scenario["name"]
        for method in METHODS[:-1]:
            for metric in ("common_return", "mean_gold_deviation", "total_volume"):
                proposed = np.array([[records[(sc, "balanced_sac", s, m)][metric] for m in markets] for s in seeds])
                reference = (np.array([[records[(sc, method, s, m)][metric] for m in markets] for s in seeds])
                             if method in LEARNED else np.array([records[(sc, method, "", m)][metric] for m in markets]))
                delta = proposed - reference
                samples = np.empty(protocol["bootstrap_resamples"])
                for start in range(0, len(samples), 1000):
                    end = min(start + 1000, len(samples))
                    si = rng.integers(len(seeds), size=(end-start, len(seeds)))
                    mi = rng.integers(len(markets), size=(end-start, len(markets)))
                    samples[start:end] = delta[si[:, :, None], mi[:, None, :]].mean(axis=(1, 2))
                lo, hi = np.percentile(samples, [2.5, 97.5])
                result.append({"scenario": sc, "comparison": "balanced_sac_minus_" + method,
                               "metric": metric, "difference": float(delta.mean()),
                               "exploratory_95_low": float(lo), "exploratory_95_high": float(hi)})
    return result


def plot(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nominal = [r for r in summary if r["scenario"] == "nominal"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("common_return", "mean_gold_deviation", "total_volume"),
                                ("Common control score (higher)", "Mean gold deviation (fraction; lower)", "Units traded (higher)")):
        ax.bar(range(len(nominal)), [r[metric + "_mean"] for r in nominal],
               yerr=[r[metric + "_market_sd"] or 0 for r in nominal],
               color=["#237d65" if r["method"] == "balanced_sac" else "#8a9daa" for r in nominal], capsize=2)
        ax.set(title=title, xticks=range(len(nominal)), xticklabels=[r["method"] for r in nominal])
        ax.tick_params(axis="x", rotation=75)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Nominal simulated markets; bars show mean ± market SD after averaging models")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    groups = {}
    with open_step_log(output / "test-steps.csv.gz") as stream:
        for row in csv.DictReader(stream):
            if row["scenario"] == "nominal" and row["method"] in ("balanced_sac", "legacy_sac", "target_price", "currency_feedback"):
                group = groups.setdefault((row["method"], int(row["step"])), [0., 0])
                group[0] += float(row["gold_ratio"])
                group[1] += 1
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for method in ("balanced_sac", "legacy_sac", "target_price", "currency_feedback"):
        x = sorted(step for name, step in groups if name == method)
        ax.plot(x, [groups[(method, step)][0] / groups[(method, step)][1] for step in x], label=method)
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="chosen currency target")
    ax.set(xlabel="Market step within episode", ylabel="Total player gold / initial total gold",
           title="Mean currency trajectories in nominal simulated markets")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "currency-trajectories.png", dpi=160)
    plt.close(fig)


def run(config_path, output, *, workers=1, reuse_root=None, validation_only=False):
    torch.set_num_threads(1)
    integer(workers, "workers")
    if workers > 4:
        raise ValueError("Use at most four independent CPU workers")
    market, sac, protocol = load_config(config_path)
    output = new_run_directory(output)
    write_json(output / "config.json", protocol)
    write_json(output / "manifest.json", manifest("balanced-control-comparison"))
    root, hashes = snapshot_research(output)
    models, records = prepare_models(market, sac, protocol, output, workers, reuse_root)
    write_json(output / "models.json", records)
    selection, trials = select_baselines(market, protocol)
    write_json(output / "baseline-selection.json", selection)
    write_csv(output / "validation-candidates.csv", trials)
    validation = evaluate_markets(market, protocol, selection, models, protocol["validation_seeds"],
                                  [{"name": "nominal", "changes": {}}], output, "validation")
    write_csv(output / "validation-summary.csv", summarize(validation))
    if not validation_only:
        test = evaluate_markets(market, protocol, selection, models, protocol["evaluation_seeds"],
                                protocol["scenarios"], output, "test")
        summary = summarize(test)
        write_csv(output / "summary.csv", summary)
        write_json(output / "summary.json", summary)
        write_csv(output / "differences.csv", uncertainty(test, protocol))
        write_reward_analysis(output / "test-steps.csv.gz", market.to_dict(), output)
        plot(output, summary)
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError("Research source changed during this run; retain the snapshot and rerun evaluation")
    write_json(output / "completion.json", {"validation_only": validation_only, "models": len(records),
        "validation_episodes": len(validation), "test_episodes": 0 if validation_only else len(test),
        "source_unchanged": True, "uncertainty": "Exploratory paired seed/market percentile bootstrap; no multiplicity correction"})
    print(f"Artifacts: {output.resolve()}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/balanced-control.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-root")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.config, args.out, workers=args.workers, reuse_root=args.reuse_root, validation_only=args.validation_only)


if __name__ == "__main__":
    run_main(main)
