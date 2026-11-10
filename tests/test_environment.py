from dataclasses import replace

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from rl.config import MarketConfig, ProductConfig
from rl.environment import MultiStoreEnv


def test_gymnasium_contract_and_initial_observation():
    env = MultiStoreEnv(MarketConfig(num_players=12, max_steps=5))
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(observation[:env.n], env.store_buy_prices)
    assert observation[-2] == 12 * env.config.initial_gold
    assert observation[-1] == np.float32(env.config.initial_fee)
    assert info["cpi"] == env.pre_cpi
    assert np.any(observation != 0)


def test_same_seed_repeats_market_and_resource_shock_survives_reset():
    config = MarketConfig(num_players=10, max_steps=4, resource_noise=0, shock_step=2)
    env = MultiStoreEnv(config)
    initial, _ = env.reset(seed=8)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    first, _, _, _, info1 = env.step(action)
    assert info1["drop_rate"] == [5] * env.n
    assert info1["loss_cpi"] == 0  # reset uses actual initial prices, not a fictitious CPI=1
    second, reward, _, _, info2 = env.step(action)
    assert info2["drop_rate"] == [10] * env.n
    reset, _ = env.reset(seed=8)
    np.testing.assert_array_equal(initial, reset)
    again, *_ = env.step(action)
    np.testing.assert_array_equal(first, again)
    again, repeated_reward, *_, repeated_info = env.step(action)
    np.testing.assert_array_equal(second, again)
    assert reward == repeated_reward
    assert info2 == repeated_info


def test_budget_inventory_conservation_bounds_and_time_limit():
    env = MultiStoreEnv(MarketConfig(num_players=20, max_steps=12))
    observation, _ = env.reset(seed=4)
    previous_money = env.player_gold.sum()
    for step in range(12):
        action = env.action_space.high if step % 2 else env.action_space.low
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert np.isfinite(env.encode(observation)).all()
        assert np.max(np.abs(env.encode(observation))) <= 1
        assert (env.player_inv >= 0).all()
        assert (env.player_gold >= 0).all()
        assert abs(env.player_gold.sum() - previous_money + info["sink"]) < 1e-7
        previous_money = env.player_gold.sum()
        assert not terminated
        assert truncated == (step == 11)
    with pytest.raises(RuntimeError):
        env.step(action)


def test_resource_stream_does_not_depend_on_strategy_actions():
    config = MarketConfig(num_players=20, max_steps=8, shock_step=4)
    a, b = MultiStoreEnv(config), MultiStoreEnv(config)
    a.reset(seed=3)
    b.reset(seed=3)
    for _ in range(8):
        *_, info_a = a.step(a.action_space.low)
        *_, info_b = b.step(b.action_space.high)
        assert info_a["drop_rate"] == info_b["drop_rate"]


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.05])
def test_invalid_fee_action_is_rejected(value):
    env = MultiStoreEnv()
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[-1] = value
    with pytest.raises(ValueError):
        env.step(action)


def test_invalid_config_and_association_rules():
    with pytest.raises(ValueError):
        ProductConfig(sell_low=0)
    with pytest.raises(ValueError):
        ProductConfig(buy_high=30, sell_low=25)
    with pytest.raises(ValueError):
        replace(MarketConfig(), association_rules=({"antecedent": [0], "consequent": 99,
                                                    "type": "complementary", "confidence": 0.7},))
    config = replace(MarketConfig(num_players=10, max_steps=2), association_rules=(
        {"antecedent": [0], "consequent": 1, "type": "complementary", "confidence": 0.7},
        {"antecedent": [1], "consequent": 2, "type": "substitute", "probability": 0.5}))
    assert MarketConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    env = MultiStoreEnv(config)
    env.reset(seed=0)
    *_, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.isfinite(info["trade_volume"])
