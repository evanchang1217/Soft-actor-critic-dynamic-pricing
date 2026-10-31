"""Stronger baselines, a true fixed-fee SAC ablation, and explicit market shifts.

Run python -m experiments.review --help. The original four-method suite remains
available through experiments.run; this protocol adds a separate evidence set.
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from rl.config import SACConfig, integer
from rl.environment import MultiStoreEnv
from rl.sac import SACAgent, verify_checkpoint_updates
from rl.training import episode_summary, train
from .analyze_reward import write_reward_analysis
from .baselines import Baseline, TargetPricePolicy
from .evaluate import format_mean_std, sample_std, write_csv
from .protocol import resolve_protocol, run_main


REVIEW_METHODS = ("fixed", "random", "adaptive", "tuned_adaptive", "target_price", "sac_fixed_fee", "sac")
LEARNED_METHODS = ("sac_fixed_fee", "sac")
BASE_FIELDS = {"description", "market", "sac", "training_seeds", "evaluation_seeds", "adaptive_gain"}
REVIEW_FIELDS = {"validation_seeds", "adaptive_gains", "target_buy_fractions", "target_sell_fractions", "scenarios"}
SCENARIO_FIELDS = {"shock_step", "price_sensitivity_low", "price_sensitivity_high"}
STEP_METRICS = ("cpi", "money", "sink", "trade_volume", "participation", "fee_rate",
                "loss_count", "loss_volume", "loss_cpi", "loss_money")
REPORT_METRICS = ("return", "mean_cpi", "cpi_std", "final_money", "total_sink", "total_volume", "mean_participation", "mean_fee")


def load_review_protocol(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) - BASE_FIELDS - REVIEW_FIELDS:
        raise ValueError("Unknown review protocol fields")
    market, sac, base = resolve_protocol({key: value for key, value in raw.items() if key in BASE_FIELDS})
    if not sac.control_fee:
        raise ValueError("The base SAC configuration must control fees; the ablation is derived separately")
    validation = raw.get("validation_seeds", [])
    if not isinstance(validation, list) or not validation:
        raise ValueError("validation_seeds must be a nonempty list")
    for seed in validation:
        integer(seed, "validation seed", 0)
    training_seeds = {seed + episode * 10_000 for seed in base["training_seeds"]
                      for episode in range(sac.episodes + 1)}
    if (len(set(validation)) != len(validation) or set(validation) & set(base["evaluation_seeds"])
            or set(validation) & training_seeds):
        raise ValueError("Validation, training, and test seeds must be distinct")
    for key in ("adaptive_gains", "target_buy_fractions", "target_sell_fractions"):
        values = raw.get(key, [])
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        minimum = 0 if key.startswith("target") else np.nextafter(0.0, 1.0)
        if any(type(v) not in (int, float) or not np.isfinite(v) or not minimum <= v <= 1 for v in values):
            raise ValueError(f"Invalid {key}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {key}")
    scenarios = raw.get("scenarios", [])
    if not scenarios or scenarios[0] != {"name": "nominal", "changes": {}}:
        raise ValueError("The first scenario must be nominal with no changes")
    names = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {"name", "changes"}:
            raise ValueError("Each scenario needs name and changes")
        name, changes = scenario["name"], scenario["changes"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError("Scenario names must be unique lowercase identifiers")
        if not isinstance(changes, dict) or set(changes) - SCENARIO_FIELDS:
            raise ValueError("Only shock timing and player-response bounds may shift in this protocol")
        replace(market, **changes)  # run normal MarketConfig validation
        names.add(name)
    return market, sac, {**base, **{key: raw[key] for key in REVIEW_FIELDS}}


def rollout(env, policy, seed):
    observation, initial = env.reset(seed=seed)
    rewards, infos, steps = [], [], []
    for step in range(1, env.config.max_steps + 1):
        action = policy.act(observation, deterministic=True) if isinstance(policy, SACAgent) else policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        infos.append(info)
        steps.append({"step": step, "reward": reward, **{key: info[key] for key in STEP_METRICS}})
        if terminated or truncated:
            break
    summary = {"initial_cpi": initial["cpi"], **episode_summary(infos, rewards),
               "mean_fee": float(np.mean([info["fee_rate"] for info in infos]))}
    return summary, steps


def select_baselines(market, validation_seeds, adaptive_gains, buy_fractions, sell_fractions):
    """This function receives no test seeds, so selection cannot inspect test scores."""
    env = MultiStoreEnv(market)
    candidates, raw = [], []
    specifications = [("tuned_adaptive", {"gain": gain}) for gain in adaptive_gains]
    specifications += [("target_price", {"buy_fraction": buy, "sell_fraction": sell})
                       for buy in buy_fractions for sell in sell_fractions]
    for index, (method, parameters) in enumerate(specifications):
        returns = []
        for seed in validation_seeds:
            policy = (Baseline("adaptive", env, gain=parameters["gain"]) if method == "tuned_adaptive"
                      else TargetPricePolicy(env, **parameters))
            row, _ = rollout(env, policy, seed)
            returns.append(row["return"])
            raw.append({"candidate": index, "method": method, "validation_seed": seed,
                        "gain": parameters.get("gain", ""), "buy_fraction": parameters.get("buy_fraction", ""),
                        "sell_fraction": parameters.get("sell_fraction", ""), "return": row["return"]})
        candidates.append({"candidate": index, "method": method, "parameters": parameters,
                           "mean_validation_return": float(np.mean(returns))})
    selected = {}
    for method in ("tuned_adaptive", "target_price"):
        # Stable first-in-grid tie breaking, with no test-set decisions.
        selected[method] = max((row for row in candidates if row["method"] == method),
                               key=lambda row: row["mean_validation_return"])
    env.close()
    return {"selection_metric": "mean nominal validation episode return",
            "validation_seeds": list(validation_seeds), "selected": selected,
            "candidates": candidates}, raw


def summarize_review(episodes):
    """Use the same test-market variability unit for all methods; report model SD separately."""
    summary = []
    for scenario, method in dict.fromkeys((r["scenario"], r["method"]) for r in episodes):
        rows = [r for r in episodes if r["scenario"] == scenario and r["method"] == method]
        by_market, by_model = {}, {}
        for row in rows:
            by_market.setdefault(row["evaluation_seed"], []).append(row)
            if method in LEARNED_METHODS:
                by_model.setdefault(row["training_seed"], []).append(row)
        record = {"scenario": scenario, "method": method, "n_test_markets": len(by_market),
                  "n_training_seeds": len(by_model), "n_episodes": len(rows)}
        for metric in REPORT_METRICS:
            markets = [np.mean([r[metric] for r in group]) for group in by_market.values()]
            models = [np.mean([r[metric] for r in group]) for group in by_model.values()]
            record[f"{metric}_mean"] = float(np.mean(markets))
            record[f"{metric}_scenario_std"] = sample_std(markets)
            record[f"{metric}_model_std"] = sample_std(models)
        summary.append(record)
    return summary


def snapshot_research(output):
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("rl/*.py"), *root.glob("experiments/*.py"), *root.glob("experiments/configs/*.json")])
    paths.append(root / "requirements-lock.txt")
    digests = {}
    with zipfile.ZipFile(output / "research-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative, content = path.relative_to(root).as_posix(), path.read_bytes()
            archive.writestr(relative, content)
            digests[relative] = hashlib.sha256(content).hexdigest()
    write_json(output / "research-source-hashes.json", digests)
    return root, digests


def prepare_models(market, sac, seeds, supplied, method, output, forbidden_seeds, *, allow_unverified=False):
    config = replace(sac, control_fee=method == "sac")
    models, records = [], []
    paths = [Path(path) for path in supplied]
    if paths and len(paths) != len(seeds):
        raise ValueError(f"{method} needs exactly one checkpoint for each configured training seed")
    if not paths:
        for seed in seeds:
            folder = output / "models" / method / f"seed-{seed}"
            print(f"Training {method} seed {seed}", flush=True)
            train(market, config, seed, folder,
                  callback=lambda row, info: print(f"  Episode {row['episode']:3d} | return {row['return']:.2f}", flush=True))
            paths.append(folder / "policy.pt")
    seen = set()
    for path in paths:
        agent, metadata = SACAgent.load(path, MultiStoreEnv(market))
        if asdict(SACConfig(**metadata["sac"])) != asdict(config):
            raise ValueError(f"{method} checkpoint hyperparameters or fee-control setting differ from the protocol")
        seed = metadata["seed"]
        if seed not in seeds or seed in seen or metadata["trained_steps"] != config.episodes * market.max_steps:
            raise ValueError(f"{method} checkpoint seed or completed training budget differs from the protocol")
        verification = verify_checkpoint_updates(metadata, path, allow_unverified=allow_unverified)
        if set(forbidden_seeds) & {seed + ep * 10_000 for ep in range(config.episodes + 1)}:
            raise ValueError("Checkpoint training overlaps validation/test market seeds")
        seen.add(seed)
        destination = output / "models" / method / f"seed-{seed}" / "policy.pt"
        if path.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(path, destination)
            for name in ("config.json", "manifest.json", "training.csv", "outcome.json"):
                if (path.parent / name).is_file():
                    shutil.copy2(path.parent / name, destination.parent / name)
        records.append({"method": method, "training_seed": seed, "input_path": str(path),
                        "stored_path": destination.relative_to(output).as_posix(),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "trained_steps": metadata["trained_steps"], "control_fee": config.control_fee,
                        **verification})
        models.append((agent, seed))
    return models, records


def run_review(config_path, output, full_checkpoints=(), fixed_fee_checkpoints=(), *, allow_unverified=False):
    market, sac, protocol = load_review_protocol(config_path)
    output = new_run_directory(output)
    torch.set_num_threads(1)
    write_json(output / "config.json", protocol)
    write_json(output / "manifest.json", manifest("review-comparison-suite"))
    source_root, digests = snapshot_research(output)
    print("Selecting simple baselines using validation markets only", flush=True)
    selection, calibration = select_baselines(market, protocol["validation_seeds"], protocol["adaptive_gains"],
                                              protocol["target_buy_fractions"], protocol["target_sell_fractions"])
    write_json(output / "baseline_selection.json", selection)
    write_csv(output / "validation_rollouts.csv", calibration)
    models, model_records = {}, []
    for method, supplied in (("sac", full_checkpoints), ("sac_fixed_fee", fixed_fee_checkpoints)):
        models[method], records = prepare_models(market, sac, protocol["training_seeds"], supplied, method, output,
                                                 protocol["validation_seeds"] + protocol["evaluation_seeds"],
                                                 allow_unverified=allow_unverified)
        model_records.extend(records)
    write_json(output / "models.json", model_records)
    episodes = []
    selected = selection["selected"]
    identifiers = ("scenario", "method", "training_seed", "evaluation_seed")
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*identifiers, "step", "reward", *STEP_METRICS])
        writer.writeheader()
        for scenario in protocol["scenarios"]:
            print(f"Evaluating scenario: {scenario['name']}", flush=True)
            env = MultiStoreEnv(replace(market, **scenario["changes"]))
            for method in REVIEW_METHODS:
                policies = models[method] if method in LEARNED_METHODS else [(None, "")]
                for agent, training_seed in policies:
                    for seed in protocol["evaluation_seeds"]:
                        policy = agent
                        if method == "target_price":
                            policy = TargetPricePolicy(env, **selected[method]["parameters"])
                        elif method == "tuned_adaptive":
                            policy = Baseline("adaptive", env, **selected[method]["parameters"])
                        elif agent is None:
                            policy = Baseline(method, env, seed=seed + 1_000_000, gain=protocol["adaptive_gain"])
                        # Agent retains its training encoder. Allowed shifts do not
                        # change observation bounds/scales, products, or reward.
                        row, steps = rollout(env, policy, seed)
                        identity = {"scenario": scenario["name"], "method": method,
                                    "training_seed": training_seed, "evaluation_seed": seed}
                        episodes.append({**identity, **row})
                        writer.writerows({**identity, **step} for step in steps)
                stream.flush()
            env.close()
    write_csv(output / "episodes.csv", episodes)
    summary = summarize_review(episodes)
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", summary)
    write_reward_analysis(output / "steps.csv", market.to_dict(), output)
    from .plot_review import plot_review
    plot_review(output)
    if any(hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest for name, digest in digests.items()):
        raise RuntimeError("Research source changed during the run; inspect the saved source archive before publishing")
    for row in summary:
        print(f"{row['scenario']:16s} {row['method']:16s} | return {format_mean_std(row['return_mean'], row['return_scenario_std'])}")
    print(f"Artifacts: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/review.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--full-checkpoints", nargs="*", default=[], help="Reuse compatible full-control SAC models")
    parser.add_argument("--fixed-fee-checkpoints", nargs="*", default=[], help="Reuse compatible fixed-fee SAC models")
    parser.add_argument("--allow-unverified-checkpoint", action="store_true",
                        help="Explicitly allow a checkpoint without an update record (results are marked unverified)")
    args = parser.parse_args()
    run_review(args.config, args.out, args.full_checkpoints, args.fixed_fee_checkpoints,
               allow_unverified=args.allow_unverified_checkpoint)


if __name__ == "__main__":
    run_main(main)
