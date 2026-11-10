"""Independent consumption/satiation switches with the original quantity rule."""
from dataclasses import asdict, dataclass
import numpy as np
from .behavior_market import Behavior, BehaviorMarket


@dataclass(frozen=True)
class Mechanism:
    consumption: bool = False
    satiation: bool = False
    consumption_rate: float = .05
    saturation_quantity: float = 20.

    def __post_init__(self):
        if type(self.consumption) is not bool or type(self.satiation) is not bool:
            raise ValueError("Mechanism switches must be booleans")
        Behavior(consumption_rate=self.consumption_rate, saturation_quantity=self.saturation_quantity)

    def to_dict(self):
        return asdict(self)


MECHANISMS = {
    "neither": Mechanism(),
    "consumption": Mechanism(consumption=True),
    "satiation": Mechanism(satiation=True),
    "both": Mechanism(consumption=True, satiation=True),
}


class MechanismMarket(BehaviorMarket):
    def __init__(self, config, mechanism=Mechanism()):
        if config.association_rules:
            raise ValueError("Mechanism study holds association rules absent")
        self.mechanism = mechanism
        super().__init__(config, Behavior(inventory_demand=mechanism.satiation,
            consumption_rate=mechanism.consumption_rate if mechanism.consumption else 0.,
            saturation_quantity=mechanism.saturation_quantity))

    def step(self, action):
        # The bundled implementation already handles neither, satiation and both.
        if not self.mechanism.consumption or self.mechanism.satiation:
            return super().step(action)
        if not self._ready or self.elapsed_steps >= self.config.max_steps:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError("Action must be inside the declared bounds")
        quantities = self.consumption_rng.binomial(self.player_inv, self.mechanism.consumption_rate)
        self.player_inv -= quantities
        consumed = int(quantities.sum())
        self.consumed_units += consumed
        result = super().step(action)
        result[-1]["consumed_units"] = consumed
        return result
