"""Headless training shared by experiments and the web demo."""

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .artifacts import manifest, new_run_directory, write_json
from .config import MarketConfig, SACConfig, integer
from .environment import MultiStoreEnv
from .sac import ReplayBuffer, SACAgent


def episode_summary(infos, rewards):
    row = {"return": float(np.sum(rewards)), "mean_reward": float(np.mean(rewards)),
           "mean_cpi": float(np.mean([i["cpi"] for i in infos])),
           "cpi_std": float(np.std([i["cpi"] for i in infos])),
           "final_money": infos[-1]["money"],
           "player_gold_mean": infos[-1]["player_gold_mean"],
           "total_sink": float(np.sum([i["sink"] for i in infos])),
           "total_volume": float(np.sum([i["trade_volume"] for i in infos])),
           "mean_participation": float(np.mean([i["participation"] for i in infos])),
           "final_cpi": infos[-1]["cpi"]}
    return row


def train(market: MarketConfig, config: SACConfig, seed: int, output,
          callback: Callable | None = None, should_stop: Callable | None = None):
    integer(seed, "seed", 0)
    output = new_run_directory(output)
    write_json(output / "config.json", {"market": market.to_dict(), "sac": asdict(config), "seed": seed})
    write_json(output / "manifest.json", manifest("sac-training"))
    # Local runs use one CPU thread to avoid tiny-network thread-pool overhead.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = MultiStoreEnv(market)
    agent = SACAgent(env, config)
    replay = ReplayBuffer(config.replay_size, env.observation_space.shape[0], agent.action_dim, seed + 1)
    exploration_rng = np.random.default_rng(seed + 2)
    steps, updates = 0, 0
    history, last_losses = [], {}
    cancelled = False
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for episode in range(1, config.episodes + 1):
            observation, _ = env.reset(seed=seed + episode * 10_000)
            rewards, infos = [], []
            for _ in range(market.max_steps):
                if should_stop and should_stop():
                    cancelled = True
                    break
                if steps < config.warmup_steps:
                    action = exploration_rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    if not config.control_fee:
                        action[-1] = 0
                else:
                    action = agent.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                # Time-limit truncations bootstrap; only true terminations suppress V(s').
                replay.push(env.encode(observation), agent.normalize_action(action), reward,
                            env.encode(next_observation), terminated)
                observation = next_observation
                steps += 1
                if steps >= config.warmup_steps and len(replay) >= config.batch_size:
                    last_losses = agent.update(replay)
                    updates += 1
                rewards.append(reward)
                infos.append(info)
                if terminated or truncated:
                    break
            if cancelled:
                break
            row = {"episode": episode, "steps": steps, **episode_summary(infos, rewards),
                   "updates": updates, "critic_loss": last_losses.get("critic_loss", 0.0),
                   "actor_loss": last_losses.get("actor_loss", 0.0),
                   "alpha": last_losses.get("alpha", config.initial_alpha)}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            history.append(row)
            if callback:
                callback(row, infos[-1])
    agent.save(output / "policy.pt", seed=seed, trained_steps=steps, gradient_updates=updates)
    write_json(output / "outcome.json", {"cancelled": cancelled, "steps": steps, "updates": updates,
                                         "completed_episodes": len(history)})
    env.close()
    return {"history": history, "output": str(Path(output)), "steps": steps,
            "updates": updates, "cancelled": cancelled}
