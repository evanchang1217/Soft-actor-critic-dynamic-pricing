"""Vectorized market derived from the capstone demo; see docs/methodology.md."""

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from .config import MarketConfig


class MultiStoreEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: MarketConfig | None = None):
        super().__init__()
        self.config = config or MarketConfig()
        c = self.config
        self.n = len(c.products)
        self.buy_low = np.array([p.buy_low for p in c.products])
        self.buy_high = np.array([p.buy_high for p in c.products])
        self.sell_low = np.array([p.sell_low for p in c.products])
        self.sell_high = np.array([p.sell_high for p in c.products])
        self.cpi_high = float(np.mean(self.sell_high / self.sell_low))
        low = np.r_[self.buy_low, self.sell_low, np.zeros(4 * self.n), -np.inf, 1, 0, c.fee_min]
        high = np.r_[self.buy_high, self.sell_high, np.full(2 * self.n, c.num_players),
                     np.full(2 * self.n, np.inf), np.inf, self.cpi_high, np.inf, c.fee_max]
        self.observation_space = Box(low.astype(np.float32), high.astype(np.float32))
        offsets = np.r_[np.full(2 * self.n, c.price_offset), c.fee_offset].astype(np.float32)
        # Actions are interleaved: buy_0, sell_0, buy_1, sell_1, ..., fee.
        self.action_space = Box(-offsets, offsets)
        self.elapsed_steps = 0
        self._ready = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        # A separate resource stream keeps exogenous drops independent of policy RNG use.
        self.resource_rng = np.random.default_rng(self.np_random.integers(0, 2**63))
        self.player_gold = np.full(c.num_players, c.initial_gold, dtype=np.float64)
        self.player_inv = np.zeros((c.num_players, self.n), dtype=np.int64)
        self.price_sensitivity = self.np_random.uniform(c.price_sensitivity_low, c.price_sensitivity_high, c.num_players)
        self.fee_sensitivity = self.np_random.uniform(0.5, 1.5, c.num_players)
        self.store_buy_prices = self.np_random.uniform(self.buy_low, self.buy_high)
        self.store_sell_prices = self.np_random.uniform(self.sell_low, self.sell_high)
        self.fee_rate = c.initial_fee
        self.drop_rate = np.full(self.n, c.avg_qty, dtype=np.float64)
        self.elapsed_steps = 0
        self.pre_cpi = self.price_index
        self.pre_money = float(self.player_gold.sum())
        self._ready = True
        zeros = np.zeros(self.n)
        self.state = self._observation(zeros, zeros, zeros, zeros, 0)
        return self.state.copy(), self._metrics(zeros, zeros, zeros, zeros, 0)

    @property
    def price_index(self):
        return float(np.mean(self.store_sell_prices / self.sell_low))

    def _observation(self, buy_count, sell_count, buy_volume, sell_volume, sink):
        return np.r_[self.store_buy_prices, self.store_sell_prices,
                     buy_count, sell_count, buy_volume, sell_volume,
                     sink, self.price_index, self.player_gold.sum(), self.fee_rate].astype(np.float32)

    def encode(self, observation):
        """Fixed semantic normalization; unbounded features use saturating transforms."""
        c, n = self.config, self.n
        x = np.asarray(observation, dtype=np.float32)
        out = np.empty_like(x)
        out[..., :n] = 2 * (x[..., :n] - self.buy_low) / (self.buy_high - self.buy_low) - 1
        out[..., n:2*n] = 2 * (x[..., n:2*n] - self.sell_low) / (self.sell_high - self.sell_low) - 1
        out[..., 2*n:4*n] = 2 * x[..., 2*n:4*n] / c.num_players - 1
        out[..., 4*n:6*n] = 2 * np.tanh(x[..., 4*n:6*n] / (c.num_players * c.avg_qty)) - 1
        gold_scale = c.num_players * c.initial_gold
        out[..., -4] = np.tanh(x[..., -4] / gold_scale)
        out[..., -3] = 2 * (x[..., -3] - 1) / (self.cpi_high - 1) - 1
        out[..., -2] = 2 * x[..., -2] / (x[..., -2] + gold_scale) - 1
        out[..., -1] = 2 * (x[..., -1] - c.fee_min) / (c.fee_max - c.fee_min) - 1
        return np.clip(out, -1, 1)

    def _sell(self):
        c = self.config
        net_price = self.store_buy_prices * (1 - self.fee_rate)
        position = np.clip((net_price - self.buy_low) / (self.buy_high - self.buy_low), 0, 1)
        probability = position[None, :] * self.price_sensitivity[:, None]
        probability /= 1 + self.fee_sensitivity[:, None] * self.fee_rate
        probability = np.clip(probability, 0, 1)
        inventory_ratio = self.player_inv / (self.player_inv + 2 * c.avg_qty)
        mask = self.np_random.random(self.player_inv.shape) < probability
        quantities = self.np_random.poisson(c.avg_qty * probability * inventory_ratio) * mask
        return np.minimum(quantities, self.player_inv)

    def _buy(self):
        c = self.config
        unit_cost = self.store_sell_prices * (1 + self.fee_rate)
        position = np.clip(1 - (unit_cost - self.sell_low) / (self.sell_high - self.sell_low), 0, 1)
        probability = position[None, :] * self.price_sensitivity[:, None]
        probability /= 1 + self.fee_sensitivity[:, None] * self.fee_rate
        probability = np.clip(probability, 0, 1)
        for rule in c.association_rules:
            antecedent = rule["antecedent"]
            selected = (self.np_random.random((c.num_players, len(antecedent))) < probability[:, antecedent]).all(axis=1)
            target = rule["consequent"]
            if rule["type"] == "complementary":
                probability[selected, target] = np.maximum(probability[selected, target], rule["confidence"])
            else:
                probability[selected, target] *= 1 - rule["probability"]
        budget_ratio = self.player_gold[:, None] / (self.player_gold[:, None] + c.avg_qty * unit_cost)
        mask = self.np_random.random(self.player_inv.shape) < probability * budget_ratio
        quantities = self.np_random.poisson(c.avg_qty * probability * budget_ratio) * mask
        total_cost = (quantities * unit_cost).sum(axis=1)
        scale = np.minimum(1, self.player_gold / np.maximum(total_cost, 1e-12))
        return np.floor(quantities * scale[:, None]).astype(np.int64)

    def _metrics(self, buy_count, sell_count, buy_volume, sell_volume, sink):
        c = self.config
        return {
            "product_names": [p.name for p in c.products],
            "cpi": self.price_index,
            "money": float(self.player_gold.sum()),
            "player_gold_mean": float(self.player_gold.mean()),
            "sink": float(sink),
            "trade_volume": float(np.sum(buy_volume) + np.sum(sell_volume)),
            "participation": float((np.sum(buy_count) + np.sum(sell_count)) / (2 * self.n * c.num_players)),
            "fee_rate": float(self.fee_rate),
            "buy_prices": self.store_buy_prices.tolist(),
            "sell_prices": self.store_sell_prices.tolist(),
            "buy_count": buy_count.tolist(), "sell_count": sell_count.tolist(),
            "buy_volume": buy_volume.tolist(), "sell_volume": sell_volume.tolist(),
            "drop_rate": self.drop_rate.tolist(),
        }

    def step(self, action):
        if not self._ready or self.elapsed_steps >= self.config.max_steps:
            raise RuntimeError("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must be finite and inside the declared per-dimension bounds")
        c = self.config
        self.elapsed_steps += 1
        self.fee_rate = float(np.clip(self.fee_rate + action[-1], c.fee_min, c.fee_max))
        self.store_buy_prices = np.clip(self.store_buy_prices * (1 + action[:-1:2]), self.buy_low, self.buy_high)
        self.store_sell_prices = np.clip(self.store_sell_prices * (1 + action[1:-1:2]), self.sell_low, self.sell_high)
        if c.shock_step == self.elapsed_steps:
            self.drop_rate *= c.shock_multiplier
        noise = self.resource_rng.normal(1, c.resource_noise, self.n)
        # Keep a floating resource process; rounding only the actual inventory input
        # avoids the systematic downward bias of repeatedly truncating drop_rate.
        self.drop_rate = np.clip(self.drop_rate * noise, 1, c.avg_qty * 5)
        self.player_inv += np.rint(self.drop_rate).astype(np.int64)[None, :]
        sold = self._sell()
        self.player_inv -= sold
        payouts = (sold * self.store_buy_prices * (1 - self.fee_rate)).sum(axis=1)
        self.player_gold += payouts
        bought = self._buy()
        costs = (bought * self.store_sell_prices * (1 + self.fee_rate)).sum(axis=1)
        self.player_gold -= costs
        self.player_gold = np.maximum(self.player_gold, 0)  # floating-point roundoff only
        self.player_inv += bought
        buy_count, sell_count = (sold > 0).sum(axis=0), (bought > 0).sum(axis=0)
        buy_volume, sell_volume = sold.sum(axis=0), bought.sum(axis=0)
        sink = float(costs.sum() - payouts.sum())
        money = float(self.player_gold.sum())
        target_count = c.num_players / 2
        target_volume = target_count * c.avg_qty
        losses = {
            "loss_count": float((np.abs(buy_count-target_count) + np.abs(sell_count-target_count)).sum() / target_count),
            "loss_volume": float((np.abs(buy_volume-target_volume) + np.abs(sell_volume-target_volume)).sum() / target_volume),
            "loss_cpi": abs(self.price_index - self.pre_cpi) / self.pre_cpi,
            # Preserve the demo's additional num_players divisor; document its effect.
            "loss_money": abs(money-self.pre_money) / max(c.num_players * self.pre_money, 1e-12),
        }
        reward = -(c.reward_count * losses["loss_count"] + c.reward_volume * losses["loss_volume"]
                   + c.reward_cpi * losses["loss_cpi"] + c.reward_money * losses["loss_money"])
        self.pre_cpi, self.pre_money = self.price_index, money
        self.state = self._observation(buy_count, sell_count, buy_volume, sell_volume, sink)
        info = self._metrics(buy_count, sell_count, buy_volume, sell_volume, sink)
        info.update(losses)
        return self.state.copy(), float(reward), False, self.elapsed_steps >= c.max_steps, info
