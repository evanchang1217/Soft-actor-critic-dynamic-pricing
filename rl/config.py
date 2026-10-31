"""Validated, JSON-serializable experiment settings. No Django dependencies."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any


def positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ProductConfig:
    name: str = "Iron"
    buy_low: float = 10.0
    buy_high: float = 20.0
    sell_low: float = 25.0
    sell_high: float = 50.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Each product needs a name")
        for key in ("buy_low", "buy_high", "sell_low", "sell_high"):
            positive(getattr(self, key), key)
        if self.buy_low >= self.buy_high or self.sell_low >= self.sell_high:
            raise ValueError("Price lower bounds must be strictly below upper bounds")
        if self.buy_high > self.sell_low:
            raise ValueError("Shop buy_high must not exceed sell_low (avoid risk-free resale)")


def default_products():
    return (
        ProductConfig(),
        ProductConfig("Gold", 20, 40, 50, 100),
        ProductConfig("Diamond", 40, 80, 100, 200),
    )


@dataclass(frozen=True)
class MarketConfig:
    products: tuple[ProductConfig, ...] = field(default_factory=default_products)
    num_players: int = 200
    avg_qty: float = 5.0
    initial_gold: float = 1000.0
    max_steps: int = 100
    price_offset: float = 0.05
    fee_offset: float = 0.01
    fee_min: float = 0.01
    fee_max: float = 0.20
    initial_fee: float = 0.05
    price_sensitivity_low: float = 0.7
    price_sensitivity_high: float = 1.3
    resource_noise: float = 0.20
    shock_step: int | None = 50
    shock_multiplier: float = 2.0
    association_rules: tuple[dict, ...] = ()
    reward_count: float = 1.0
    reward_volume: float = 1.0
    reward_cpi: float = 0.5
    reward_money: float = 0.5

    def __post_init__(self):
        if not self.products or any(not isinstance(p, ProductConfig) for p in self.products):
            raise ValueError("products must contain ProductConfig values")
        if len({p.name for p in self.products}) != len(self.products):
            raise ValueError("Product names must be unique")
        integer(self.num_players, "num_players")
        integer(self.max_steps, "max_steps")
        for key in ("avg_qty", "initial_gold", "price_offset", "fee_offset", "shock_multiplier"):
            positive(getattr(self, key), key)
        if not (0 < self.fee_min < self.fee_max < 1 and self.fee_min <= self.initial_fee <= self.fee_max):
            raise ValueError("Fee bounds must satisfy 0 < fee_min <= initial_fee <= fee_max < 1")
        if self.price_offset >= 1 or self.fee_offset >= 1:
            raise ValueError("Action offsets must be below 1")
        positive(self.price_sensitivity_low, "price_sensitivity_low")
        positive(self.price_sensitivity_high, "price_sensitivity_high")
        if self.price_sensitivity_low >= self.price_sensitivity_high:
            raise ValueError("price_sensitivity_low must be below price_sensitivity_high")
        if not math.isfinite(self.resource_noise) or not 0 <= self.resource_noise <= 1:
            raise ValueError("resource_noise must be between 0 and 1")
        if self.shock_step is not None:
            integer(self.shock_step, "shock_step")
        for key in ("reward_count", "reward_volume", "reward_cpi", "reward_money"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for rule in self.association_rules:
            if not isinstance(rule, dict) or rule.get("type") not in ("complementary", "substitute"):
                raise ValueError("Rule type must be complementary or substitute")
            antecedent = rule.get("antecedent", [])
            indices = [*antecedent, rule.get("consequent")]
            if not antecedent or any(type(i) is not int or not 0 <= i < len(self.products) for i in indices):
                raise ValueError("Rule indices must refer to existing products (zero-based)")
            if len(set(antecedent)) != len(antecedent) or rule["consequent"] in antecedent:
                raise ValueError("Rule items must be distinct")
            key = "confidence" if rule["type"] == "complementary" else "probability"
            value = rule.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Rule {key} must be between 0 and 1")

    @classmethod
    def from_dict(cls, values: dict[str, Any]):
        values = dict(values)
        if "products" in values:
            values["products"] = tuple(ProductConfig(**p) for p in values["products"])
        if "association_rules" in values:
            values["association_rules"] = tuple(values["association_rules"])
        return cls(**values)

    def to_dict(self):
        # JSON round-tripping gives consistent lists for checkpoint/config comparisons.
        import json
        return json.loads(json.dumps(asdict(self), allow_nan=False))


@dataclass(frozen=True)
class SACConfig:
    episodes: int = 100
    hidden: int = 128
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 1000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 0.0003
    initial_alpha: float = 0.2
    control_fee: bool = True

    def __post_init__(self):
        if type(self.control_fee) is not bool:
            raise ValueError("control_fee must be a boolean")
        for key in ("episodes", "hidden", "batch_size", "replay_size"):
            integer(getattr(self, key), key)
        integer(self.warmup_steps, "warmup_steps", 0)
        if self.batch_size > self.replay_size:
            raise ValueError("batch_size cannot exceed replay_size")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        positive(self.learning_rate, "learning_rate")
        positive(self.initial_alpha, "initial_alpha")
