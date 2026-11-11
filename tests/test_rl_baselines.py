"""Optional baseline integration; install requirements-baselines.txt to run."""
from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip('stable_baselines3')

from experiments.rl_baselines import (PricingEnv, load_config, train_one, evaluate_one,
                                     check_historical_settings, METHODS)
from rl.config import MarketConfig
from rl.environment import MultiStoreEnv


def small_market():
    return MarketConfig(num_players=8, max_steps=4, shock_step=2,
                        reward_mode='balanced_v1', reward_count=.5, reward_volume=.5,
                        reward_cpi=.05, reward_money=1.)


def test_wrapper_preserves_transitions_and_removes_fee_control():
    env = PricingEnv(small_market())
    reference = MultiStoreEnv(small_market())
    encoded, _ = env.reset(seed=21)
    raw, _ = reference.reset(seed=21)
    np.testing.assert_array_equal(encoded, reference.encode(raw))
    move = np.array([-1., 0., 1., .25, -.5, .75], np.float32)
    actual = env.step(move)
    expected = reference.step(np.r_[move*.05, 0.].astype(np.float32))
    np.testing.assert_array_equal(actual[0], reference.encode(expected[0]))
    assert actual[1:] == expected[1:]
    assert env.action_space.shape == (6,) and env.physical_action(move)[-1] == 0
    assert env.observation_space.contains(actual[0])


def test_dqn_grid_has_all_joint_moves_and_rejects_large_or_invalid_actions():
    env = PricingEnv(small_market(), discrete=True)
    assert env.action_space.n == 729 and np.unique(env.grid, axis=0).shape == (729, 6)
    np.testing.assert_array_equal(env.physical_action(364), np.zeros(7))
    assert np.all(env.physical_action(0)[:-1] == np.float32(-.05))
    assert np.all(env.physical_action(728)[:-1] == np.float32(.05))
    for invalid in (-1, 729, .5):
        with pytest.raises(ValueError):
            env.physical_action(invalid)
    products = tuple(replace(small_market().products[0], name=f'Item {i}') for i in range(6))
    with pytest.raises(ValueError, match='too large'):
        PricingEnv(replace(small_market(), products=products), discrete=True)


def test_training_markets_match_original_sac_episode_seeds():
    env = PricingEnv(small_market(), training_seed=71)
    reference = MultiStoreEnv(small_market())
    for episode in (1, 2):
        observation, _ = env.reset(seed=999)
        raw, _ = reference.reset(seed=71+episode*10000)
        np.testing.assert_array_equal(observation, reference.encode(raw))


@pytest.mark.parametrize('patch', [dict(training_steps=6401), dict(ppo_rollout_steps=130),
                                 dict(evaluation_seeds=[10000, 10001])])
def test_comparison_rejects_budget_mismatch_and_seed_leakage(tmp_path, patch):
    _, p = load_config('experiments/configs/rl-baselines.json')
    p.update(patch)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_config(path)


def test_historical_settings_are_checked_before_training():
    market, p = load_config('experiments/configs/rl-baselines.json')
    metadata = check_historical_settings(market, p)
    assert metadata['test_markets_previously_used']
    for field, value in [('training_seeds', [71, 72, 73]), ('sac', dict(p['sac'], hidden=64))]:
        with pytest.raises(ValueError, match=f'historical {field}'):
            check_historical_settings(market, dict(p, **{field: value}))


@pytest.mark.parametrize('method', METHODS)
def test_each_method_learns_saves_and_evaluates_under_exact_budget(tmp_path, method):
    _, p = load_config('experiments/configs/rl-baselines.json')
    p.update(market=small_market().to_dict(), training_steps=64,
             sac=dict(p['sac'], episodes=16, hidden=12, batch_size=8, warmup_steps=8, replay_size=128),
             ppo_rollout_steps=16, ppo_batch_size=8,
             scenarios=[dict(name='nominal', changes={})], evaluation_seeds=[311000010, 311000011])
    metadata = train_one(tmp_path, p, method, 71)
    assert metadata['steps'] == 64 and metadata['update_counter'] > 0
    assert (tmp_path/metadata['path']).is_file()
    rows = evaluate_one(tmp_path, p, metadata)
    assert len(rows) == 2 and all(row['training_seed'] == 71 for row in rows)
    assert all(np.isfinite(row['common_return']) for row in rows)
    assert all(row['method'] == method for row in rows)
