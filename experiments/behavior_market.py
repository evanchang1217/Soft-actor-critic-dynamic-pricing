"""Explicit alternative market assumptions; the published core is unchanged.

These are synthetic sensitivity models, not fitted or validated player models.
The inventory factor is a bundled consumption/satiation assumption, not two
separately identified causal effects. Observation and reward semantics are fixed.
"""

from dataclasses import asdict, dataclass
import numpy as np

from rl.environment import MultiStoreEnv
from .baselines import TargetPricePolicy


@dataclass(frozen=True)
class Behavior:
    separate_quantity: bool = False
    inventory_demand: bool = False
    consumption_rate: float = 0.05
    saturation_quantity: float = 20.0

    def __post_init__(self):
        if type(self.separate_quantity) is not bool or type(self.inventory_demand) is not bool:
            raise ValueError("Behavior switches must be booleans")
        if not np.isfinite(self.consumption_rate) or not 0 <= self.consumption_rate <= 1:
            raise ValueError("Consumption rate must be between zero and one")
        if not np.isfinite(self.saturation_quantity) or self.saturation_quantity <= 0:
            raise ValueError("Saturation quantity must be positive")

    def to_dict(self):
        return asdict(self)


BEHAVIORS = {
    "original": Behavior(),
    "separated": Behavior(separate_quantity=True),
    "inventory": Behavior(inventory_demand=True),
    "combined": Behavior(separate_quantity=True, inventory_demand=True),
}


class BehaviorMarket(MultiStoreEnv):
    def __init__(self, config, behavior=Behavior()):
        super().__init__(config)
        self.behavior = behavior
        if config.association_rules and behavior != Behavior():
            raise ValueError("The assumption audit does not vary association rules")

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        # Do not draw from either legacy RNG. Policy changes cannot change the
        # exogenous resource stream. Consumption counts can depend on inventory.
        self.consumption_rng = np.random.default_rng(
            None if seed is None else np.random.SeedSequence([int(seed), 718391]))
        self.consumed_units = 0
        return result

    def _sell(self):
        if not self.behavior.separate_quantity:
            return super()._sell()
        c = self.config
        net = self.store_buy_prices * (1-self.fee_rate)
        position = np.clip((net-self.buy_low)/(self.buy_high-self.buy_low), 0, 1)
        p = position[None, :] * self.price_sensitivity[:, None]
        p = np.clip(p/(1+self.fee_sensitivity[:, None]*self.fee_rate), 0, 1)
        inventory_ratio = self.player_inv/(self.player_inv+2*c.avg_qty)
        mask = self.np_random.random(self.player_inv.shape) < p
        quantity = self.np_random.poisson(c.avg_qty*inventory_ratio)*mask
        return np.minimum(quantity, self.player_inv)

    def _buy(self):
        b, c = self.behavior, self.config
        if not b.separate_quantity and not b.inventory_demand:
            return super()._buy()
        cost = self.store_sell_prices*(1+self.fee_rate)
        position = np.clip(1-(cost-self.sell_low)/(self.sell_high-self.sell_low), 0, 1)
        p = position[None, :]*self.price_sensitivity[:, None]
        p = np.clip(p/(1+self.fee_sensitivity[:, None]*self.fee_rate), 0, 1)
        budget = self.player_gold[:, None]/(self.player_gold[:, None]+c.avg_qty*cost)
        satiation = b.saturation_quantity/(b.saturation_quantity+self.player_inv) if b.inventory_demand else 1.
        # Satiation acts ONCE, on participation, in both quantity variants.
        mask = self.np_random.random(self.player_inv.shape) < p*budget*satiation
        rate = c.avg_qty*budget*(1. if b.separate_quantity else p)
        quantity = self.np_random.poisson(rate)*mask
        total = (quantity*cost).sum(axis=1)
        scale = np.minimum(1, self.player_gold/np.maximum(total, 1e-12))
        return np.floor(quantity*scale[:, None]).astype(np.int64)

    def step(self, action):
        # Reject invalid steps before consuming inventory or changing RNG state.
        if not self._ready or self.elapsed_steps >= self.config.max_steps:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError("Action must be inside the declared bounds")
        consumed = 0
        if self.behavior.inventory_demand:
            quantities = self.consumption_rng.binomial(self.player_inv, self.behavior.consumption_rate)
            self.player_inv -= quantities
            consumed = int(quantities.sum())
        self.consumed_units += consumed
        obs, reward, terminated, truncated, info = super().step(action)
        info["consumed_units"] = consumed
        return obs, reward, terminated, truncated, info


class IntegralFeedback:
    """Observable currency PI control with a bounded integral state.

The integral is in target-fraction units and clamped at +/-0.5 (anti-windup).
The price actions retain the same per-step limits as every other method.
"""
    def __init__(self, env, buy_fraction, sell_fraction, gain, integral_gain):
        TargetPricePolicy(env, buy_fraction, sell_fraction)
        if not all(np.isfinite(x) and x > 0 for x in (gain, integral_gain)):
            raise ValueError("PI gains must be positive")
        self.env, self.buy, self.sell = env, buy_fraction, sell_fraction
        self.gain, self.integral_gain = gain, integral_gain
        self.reset()

    def reset(self):
        self.integral = 0.

    def act(self, observation):
        target = self.env.config.num_players*self.env.config.initial_gold
        error = float(observation[-2])/target-1
        self.integral = float(np.clip(self.integral+self.integral_gain*error, -.5, .5))
        correction = self.gain*error+self.integral
        buy = float(np.clip(self.buy-correction, 0, 1))
        sell = float(np.clip(self.sell-correction, 0, 1))
        return TargetPricePolicy(self.env, buy, sell).act(observation)
