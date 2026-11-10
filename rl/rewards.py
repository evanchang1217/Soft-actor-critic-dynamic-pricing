"""Versioned control objectives; market dynamics do not depend on the objective."""


def reward_components(config, losses, money=None):
    """Return nonnegative weighted penalties in a fixed order.

    balanced_v1 averages trade losses over product/direction slots, scales
    price changes by the action limit, and regulates a fixed currency setpoint.
    Its currency target is a chosen control objective, not an estimate of welfare.
    Legacy values and arithmetic order are retained for archived checkpoints.
    """
    if config.reward_mode == "legacy":
        return {name: getattr(config, f"reward_{name}") * float(losses[f"loss_{name}"])
                for name in ("count", "volume", "cpi", "money")}
    if money is None:
        raise ValueError("balanced_v1 requires the current total player gold")
    slots = 2 * len(config.products)
    target = config.num_players * config.initial_gold
    return {
        "count": config.reward_count * float(losses["loss_count"]) / slots,
        "volume": config.reward_volume * float(losses["loss_volume"]) / slots,
        "cpi": config.reward_cpi * float(losses["loss_cpi"]) / config.price_offset,
        "money": config.reward_money * abs(float(money) / target - 1),
    }
