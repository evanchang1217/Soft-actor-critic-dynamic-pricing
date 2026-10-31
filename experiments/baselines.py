"""Transparent new baselines; these do not reconstruct the poster's baselines."""

import numpy as np


METHODS = ("fixed", "random", "adaptive", "sac")


class Baseline:
    def __init__(self, name, env, seed=0, gain=0.5):
        if name not in METHODS[:-1]:
            raise ValueError(f"Unknown baseline: {name}")
        self.name, self.env, self.gain = name, env, gain
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        env, n = self.env, self.env.n
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        if self.name == "random":
            return self.rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        if self.name == "adaptive":
            # Use only the previous observation, never future decisions or hidden inventory.
            # Relative to the same N/2 participation target as the reward:
            # excess sellers -> reduce shop buy price; excess buyers -> increase sell price.
            if env.elapsed_steps:
                target = env.config.num_players / 2
                action[:-1:2] = -self.gain * env.config.price_offset * (observation[2*n:3*n] - target) / target
                action[1:-1:2] = self.gain * env.config.price_offset * (observation[3*n:4*n] - target) / target
        # Fixed and adaptive keep the initial fee; random and SAC can change it.
        return np.clip(action, env.action_space.low, env.action_space.high)


class TargetPricePolicy:
    """Approach constant price targets with bounded actions; keep the initial fee.

    Fractions locate targets within each product's price interval. They are
    selected on validation markets, never on the reported evaluation markets.
    """

    def __init__(self, env, buy_fraction, sell_fraction):
        for value in (buy_fraction, sell_fraction):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Target fractions must be finite numbers between 0 and 1")
        self.env = env
        self.buy_target = env.buy_low + buy_fraction * (env.buy_high - env.buy_low)
        self.sell_target = env.sell_low + sell_fraction * (env.sell_high - env.sell_low)

    def act(self, observation):
        env, n = self.env, self.env.n
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[:-1:2] = self.buy_target / observation[:n] - 1
        action[1:-1:2] = self.sell_target / observation[n:2*n] - 1
        return np.clip(action, env.action_space.low, env.action_space.high)
