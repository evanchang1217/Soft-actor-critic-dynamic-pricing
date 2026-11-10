"""Transparent new baselines; these do not reconstruct the poster's baselines."""

import numpy as np


METHODS = ("fixed", "random", "adaptive", "sac")


class Baseline:
    def __init__(self, name, env, seed=0, gain=0.5):
        if name not in (*METHODS[:-1], "adaptive_proportional"):
            raise ValueError(f"Unknown baseline: {name}")
        self.name, self.env, self.gain = name, env, gain
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        env, n = self.env, self.env.n
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        if self.name == "random":
            return self.rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        if self.name in ("adaptive", "adaptive_proportional"):
            # Use only the previous observation, never future decisions or hidden inventory.
            # Relative to the same N/2 participation target as the reward:
            # excess sellers -> reduce shop buy price; excess buyers -> increase sell price.
            if env.elapsed_steps:
                target = env.config.num_players / 2
                sellers = observation[2*n:3*n] - target
                buyers = observation[3*n:4*n] - target
                if self.name == "adaptive":
                    action[:-1:2] = -np.sign(sellers) * self.rng.uniform(0, env.config.price_offset, n)
                    action[1:-1:2] = np.sign(buyers) * self.rng.uniform(0, env.config.price_offset, n)
                else:
                    action[:-1:2] = -self.gain * env.config.price_offset * sellers / target
                    action[1:-1:2] = self.gain * env.config.price_offset * buyers / target
        # Fixed and adaptive keep the initial fee; random and SAC can change it.
        return np.clip(action, env.action_space.low, env.action_space.high)


class FixedStepAdaptive:
    """Participation feedback with a fixed proportional price step; fee is held."""

    def __init__(self, env, magnitude):
        if not isinstance(magnitude, (int, float)) or not np.isfinite(magnitude) or not 0 < magnitude <= env.config.price_offset:
            raise ValueError("Fixed magnitude must be positive and within the price action bound")
        self.env, self.magnitude = env, magnitude

    def act(self, observation):
        env, n = self.env, self.env.n
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        if env.elapsed_steps:
            target = env.config.num_players / 2
            action[:-1:2] = -np.sign(observation[2*n:3*n] - target) * self.magnitude
            action[1:-1:2] = np.sign(observation[3*n:4*n] - target) * self.magnitude
        return action


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


class CurrencyFeedbackPolicy:
    """A strong, observable-state baseline with proportional currency feedback.

    Excess player gold lowers both price targets to reduce payouts and encourage
    purchases; insufficient gold raises them. All parameters are selected on
    validation markets. The baseline sees the same aggregate gold as SAC.
    """

    def __init__(self, env, buy_fraction, sell_fraction, gain):
        TargetPricePolicy(env, buy_fraction, sell_fraction)  # shared validation
        if type(gain) not in (int, float) or not np.isfinite(gain) or gain <= 0:
            raise ValueError("Currency feedback gain must be positive and finite")
        self.env = env
        self.buy_fraction, self.sell_fraction, self.gain = buy_fraction, sell_fraction, gain

    def act(self, observation):
        target = self.env.config.num_players * self.env.config.initial_gold
        error = float(observation[-2]) / target - 1
        buy = float(np.clip(self.buy_fraction - self.gain * error, 0, 1))
        sell = float(np.clip(self.sell_fraction - self.gain * error, 0, 1))
        return TargetPricePolicy(self.env, buy, sell).act(observation)
