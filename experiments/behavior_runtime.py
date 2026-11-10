"""Saved-policy inference and metrics shared with the original behavioral study."""
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from rl.sac import SACAgent
from .baselines import CurrencyFeedbackPolicy
from .behavior_market import BEHAVIORS,IntegralFeedback

METRICS = ("score_per_step", "mean_gold_error", "max_gold_error", "final_gold_ratio",
           "volume_per_step", "mean_participation", "mean_low_gold_fraction", "final_low_gold_fraction",
           "mean_gini", "final_gini", "final_gold_p10", "final_gold_p50", "final_gold_p90",
           "buy_price_change", "sell_price_change", "max_price_change", "price_bound_fraction",
           "inventory_per_player", "consumed_per_step", "penalty_count", "penalty_volume",
           "penalty_price", "penalty_money")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def behavior_for(config, name):
    b = BEHAVIORS[name]
    return replace(b, consumption_rate=config["consumption_rate"],
                   saturation_quantity=config["saturation_quantity"])


def gini(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    total = ordered.sum()
    if total == 0:
        return 0.
    n = len(ordered)
    return float(2*np.dot(np.arange(1, n+1), ordered)/(n*total)-(n+1)/n)


def episode(env, policy, seed, threshold, trace=False):
    observation, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    c = env.config
    target = c.num_players*c.initial_gold
    sums = {k:0. for k in METRICS}
    worst_error, worst_price, trace_rows = 0., 0., []
    from rl.rewards import reward_components
    for step in range(1, c.max_steps+1):
        old_buy, old_sell = env.store_buy_prices.copy(), env.store_sell_prices.copy()
        action = policy.act(observation, deterministic=True) if isinstance(policy, SACAgent) else policy.act(observation)
        if action[-1] != 0:
            raise ValueError("A policy changed the fixed fee")
        previous_gold = float(env.player_gold.sum())
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.isclose(info["money"]-previous_gold, -info["sink"], rtol=0, atol=1e-6):
            raise ValueError("Currency accounting failure")
        if np.any(env.player_gold < 0) or np.any(env.player_inv < 0):
            raise ValueError("Negative player holdings")
        err = abs(info["money"]/target-1)
        low = float(np.mean(env.player_gold < threshold*c.initial_gold))
        inequality = gini(env.player_gold)
        buy = np.abs(env.store_buy_prices/old_buy-1)
        sell = np.abs(env.store_sell_prices/old_sell-1)
        bound = np.r_[np.isclose(env.store_buy_prices, env.buy_low, rtol=0, atol=1e-6)
                      | np.isclose(env.store_buy_prices, env.buy_high, rtol=0, atol=1e-6),
                      np.isclose(env.store_sell_prices, env.sell_low, rtol=0, atol=1e-6)
                      | np.isclose(env.store_sell_prices, env.sell_high, rtol=0, atol=1e-6)]
        penalties = reward_components(c, info, info["money"])
        if not np.isclose(reward, -sum(penalties.values()), atol=1e-12, rtol=0):
            raise ValueError("Reward reconstruction failed")
        values = dict(score_per_step=reward, mean_gold_error=err, volume_per_step=info["trade_volume"],
                      mean_participation=info["participation"], mean_low_gold_fraction=low, mean_gini=inequality,
                      buy_price_change=float(buy.mean()), sell_price_change=float(sell.mean()),
                      price_bound_fraction=float(bound.mean()), consumed_per_step=info["consumed_units"],
                      penalty_count=penalties["count"], penalty_volume=penalties["volume"],
                      penalty_price=penalties["cpi"], penalty_money=penalties["money"])
        for name, value in values.items():
            sums[name] += value
        worst_error = max(worst_error, err)
        worst_price = max(worst_price, float(max(buy.max(), sell.max())))
        if trace:
            trace_rows.append(dict(step=step, gold_ratio=info["money"]/target, low_gold_fraction=low,
                                   gini=inequality, inventory_per_player=float(env.player_inv.sum()/c.num_players),
                                   **values, **{f"buy_{i}":float(p) for i,p in enumerate(env.store_buy_prices)},
                                   **{f"sell_{i}":float(p) for i,p in enumerate(env.store_sell_prices)}))
        if terminated or truncated:
            break
    row = {k:v/step for k,v in sums.items()}
    q10, q50, q90 = np.quantile(env.player_gold, [.1,.5,.9])
    row.update(max_gold_error=worst_error, max_price_change=worst_price,
               final_gold_ratio=float(env.player_gold.sum()/target), final_low_gold_fraction=low,
               final_gini=inequality, final_gold_p10=float(q10), final_gold_p50=float(q50), final_gold_p90=float(q90),
               inventory_per_player=float(env.player_inv.sum()/c.num_players), steps=step)
    if not all(np.isfinite(v) for v in row.values()):
        raise ValueError("Non-finite episode metric")
    return row, trace_rows, env.player_gold.tolist()


def make_policy(env, spec):
    if spec["kind"] == "sac":
        agent, _ = SACAgent.load(spec["path"])
        agent.env = env
        return agent
    cls = CurrencyFeedbackPolicy if spec["kind"] == "feedback" else IntegralFeedback
    return cls(env, **spec["parameters"])
