"""Response-randomized SAC training with a fixed budget and observable-state parity."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import time

from rl.artifacts import manifest, new_run_directory, write_json
from rl.config import MarketConfig, SACConfig
from rl.environment import MultiStoreEnv
from rl.sac import ReplayBuffer, SACAgent
from rl.training import episode_summary
from .balanced_control import load_config


def sample_market(base, distribution, rng):
    lo, hi, width = (distribution[k] for k in ("center_low", "center_high", "half_width"))
    if not all(np.isfinite(v) for v in (lo, hi, width)) or not 0 < width < lo <= hi:
        raise ValueError("Response centers/width must be finite and positive")
    center = float(rng.uniform(lo, hi))
    return replace(base, price_sensitivity_low=center-width, price_sensitivity_high=center+width), center


def train_randomized(market_values, sac_values, seed, distribution, output):
    base, config = MarketConfig.from_dict(market_values), SACConfig(**sac_values)
    if config.control_fee or base.reward_mode != "balanced_v1":
        raise ValueError("The robustness comparison uses balanced reward and fixed fees")
    output = new_run_directory(output)
    write_json(output/"config.json", {"market":base.to_dict(),"sac":asdict(config),"seed":seed,
                                     "training_distribution":distribution})
    write_json(output/"manifest.json",manifest("response-randomized-sac"))
    sources = ("rl/environment.py","rl/sac.py","rl/config.py","rl/rewards.py",
               "experiments/robust_training.py","experiments/configs/robustness.json")
    source_hashes = {n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources}
    write_json(output/"training-source-hashes.json",source_hashes)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = MultiStoreEnv(base)
    agent = SACAgent(env,config)
    replay = ReplayBuffer(config.replay_size,env.observation_space.shape[0],agent.action_dim,seed+1)
    exploration = np.random.default_rng(seed+2)
    domains = np.random.default_rng(seed+distribution["rng_offset"])
    steps, updates, last = 0, 0, {}
    with (output/"training.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = None
        for episode in range(1,config.episodes+1):
            episode_market, center = sample_market(base,distribution,domains)
            env.close()
            env = MultiStoreEnv(episode_market)
            agent.env = env
            observation,_ = env.reset(seed=seed+episode*10000)
            rewards,infos = [],[]
            for _ in range(base.max_steps):
                if steps < config.warmup_steps:
                    action = exploration.uniform(env.action_space.low,env.action_space.high).astype(np.float32)
                    action[-1] = 0
                else:
                    action = agent.act(observation)
                nxt,reward,terminated,truncated,info = env.step(action)
                replay.push(env.encode(observation),agent.normalize_action(action),reward,env.encode(nxt),terminated)
                observation = nxt
                steps += 1
                if steps >= config.warmup_steps and len(replay) >= config.batch_size:
                    last = agent.update(replay)
                    updates += 1
                rewards.append(reward)
                infos.append(info)
                if terminated or truncated:
                    break
            row = {"episode":episode,"steps":steps,**episode_summary(infos,rewards),"updates":updates,
                   "critic_loss":last.get("critic_loss",0.),"actor_loss":last.get("actor_loss",0.),
                   "alpha":last.get("alpha",config.initial_alpha),"response_center":center,
                   "response_low":episode_market.price_sensitivity_low,
                   "response_high":episode_market.price_sensitivity_high}
            if writer is None:
                writer = csv.DictWriter(stream,fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if steps % 10000 == 0:
                print(f"Randomized SAC seed {seed}: {steps} steps, {updates} updates",flush=True)
    # The saved market describes inference/encoding semantics. The additional
    # checkpoint field and per-episode log record the actual training domains.
    agent.env = MultiStoreEnv(base)
    agent.save(output/"policy.pt",seed=seed,trained_steps=steps,gradient_updates=updates)
    checkpoint = torch.load(output/"policy.pt",weights_only=True)
    checkpoint["training_distribution"] = dict(distribution)
    torch.save(checkpoint,output/"policy.tmp")
    (output/"policy.tmp").replace(output/"policy.pt")
    if any(hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h for n,h in source_hashes.items()):
        raise RuntimeError("Training source changed; retain this run and create a new experiment")
    write_json(output/"outcome.json", {"steps":steps,"updates":updates,"completed_episodes":config.episodes,
        "training_distribution":distribution,"training_source_unchanged":True,"cancelled":False})
    env.close()
    return {"seed":seed,"steps":steps,"updates":updates}



def train_curve(config_path, seed, budgets, output, reference_root):
    market, config, _ = load_config(config_path)
    if (not budgets or sorted(set(budgets)) != list(budgets)
            or any(type(b) is not int or b <= 0 or b % market.max_steps for b in budgets)):
        raise ValueError("Budgets must increase and end on full episodes")
    config = replace(config, episodes=budgets[-1] // market.max_steps)
    output = new_run_directory(output)
    write_json(output / "config.json", {"market": market.to_dict(), "sac": asdict(config),
        "seed": seed, "budgets": list(budgets), "reference_root": str(reference_root)})
    write_json(output / "manifest.json", manifest("sac-budget-diagnostic"))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = MultiStoreEnv(market)
    agent = SACAgent(env, config)
    replay = ReplayBuffer(config.replay_size, env.observation_space.shape[0], agent.action_dim, seed + 1)
    exploration_rng = np.random.default_rng(seed + 2)
    steps, updates, records = 0, 0, []
    started = time.monotonic()
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for episode in range(1, config.episodes + 1):
            observation, _ = env.reset(seed=seed + episode * 10000)
            rewards, infos, last = [], [], {}
            for _ in range(market.max_steps):
                if steps < config.warmup_steps:
                    action = exploration_rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    if not config.control_fee:
                        action[-1] = 0
                else:
                    action = agent.act(observation)
                nxt, reward, terminated, truncated, info = env.step(action)
                replay.push(env.encode(observation), agent.normalize_action(action), reward,
                            env.encode(nxt), terminated)
                observation = nxt
                steps += 1
                if steps >= config.warmup_steps and len(replay) >= config.batch_size:
                    last = agent.update(replay)
                    updates += 1
                rewards.append(reward)
                infos.append(info)
                if terminated or truncated:
                    break
            row = {"episode": episode, "steps": steps, **episode_summary(infos, rewards),
                   "updates": updates, "critic_loss": last.get("critic_loss", 0.),
                   "actor_loss": last.get("actor_loss", 0.), "alpha": last.get("alpha", config.initial_alpha)}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if steps in budgets:
                folder = new_run_directory(output / f"steps-{steps}")
                # Record the completed budget, so leakage checks remain correct.
                original = agent.config
                agent.config = replace(config, episodes=episode)
                agent.save(folder / "policy.pt", seed=seed, trained_steps=steps, gradient_updates=updates)
                agent.config = original
                record = {"steps": steps, "updates": updates, "seconds": time.monotonic()-started}
                if steps == 10000:
                    old = Path(reference_root) / "models/balanced_sac" / f"seed-{seed}"
                    previous = torch.load(old / "policy.pt", weights_only=True)["policy"]
                    record["matches_published_policy"] = all(torch.equal(v, previous[k])
                                                             for k, v in agent.policy.state_dict().items())
                    record["matches_published_history"] = (output / "training.csv").read_bytes() == (old / "training.csv").read_bytes()
                    if not record["matches_published_policy"] or not record["matches_published_history"]:
                        raise ValueError("The 10k diagnostic does not reproduce published training")
                records.append(record)
                write_json(output / "checkpoints.json", records)
                print(f"Budget diagnostic seed {seed}: {steps} steps, {updates} updates", flush=True)
    write_json(output / "completion.json", {"seed": seed, "steps": steps, "updates": updates,
                                            "checkpoints": records})
    env.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="experiments/configs/robustness.json")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--out")
    parser.add_argument("--include-nominal-extension",action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("Use 1–3 workers")
    raw = json.loads(Path(args.config).read_text())
    market,sac,_ = load_config(raw["base_config"])
    sac = replace(sac,episodes=raw["training_steps"]//market.max_steps)
    output = new_run_directory(args.out or raw["randomized_training_root"])
    write_json(output/"protocol.json",raw)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
        futures = []
        if args.include_nominal_extension:
            nominal = new_run_directory(raw["nominal_extension_root"])
            write_json(nominal/"protocol.json",raw)
            futures.extend(pool.submit(train_curve,raw["base_config"],seed,[10000,20000,30000],
                                       str(nominal/f"seed-{seed}"),raw["reference_root"])
                           for seed in raw["new_nominal_seeds"])
        futures.extend(pool.submit(train_randomized,market.to_dict(),asdict(sac),seed,raw["response_distribution"],
                                   str(output/f"seed-{seed}")) for seed in raw["training_seeds"])
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
