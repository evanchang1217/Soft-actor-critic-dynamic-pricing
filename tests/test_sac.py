from dataclasses import replace

import numpy as np
import pytest
import torch

from rl.config import MarketConfig, SACConfig
from rl.environment import MultiStoreEnv
from rl.sac import GaussianPolicy, ReplayBuffer, SACAgent
from rl.training import train


def test_policy_respects_heterogeneous_action_bounds_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    env = MultiStoreEnv(MarketConfig(num_players=8, max_steps=4))
    observation, _ = env.reset(seed=2)
    agent = SACAgent(env, SACConfig(hidden=16))
    for _ in range(30):
        action = agent.act(observation)
        assert env.action_space.contains(action)
        assert abs(action[-1]) <= 0.01
    before = agent.act(observation, deterministic=True)
    path = tmp_path / "policy.pt"
    agent.save(path, seed=1, trained_steps=10)
    restored, metadata = SACAgent.load(path, env)
    np.testing.assert_array_equal(before, restored.act(observation, deterministic=True))
    assert metadata["trained_steps"] == 10
    with pytest.raises(ValueError, match="configuration differs"):
        SACAgent.load(path, MultiStoreEnv(replace(env.config, initial_fee=0.1)))


def test_saturated_policy_log_probability_and_gradients_are_finite():
    policy = GaussianPolicy(4, 3, 8)
    with torch.no_grad():
        policy.net[-1].bias[:3].fill_(30)
    action, log_probability = policy.sample(torch.zeros(5, 4))
    assert torch.isfinite(log_probability).all()
    assert (action.abs() <= 1).all()
    (-log_probability.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters())


def test_replay_buffer_wraps_and_does_not_mix_old_entries():
    buffer = ReplayBuffer(3, 2, 1, seed=4)
    for i in range(5):
        buffer.push([i, i], [0], i, [i+1, i+1], False)
    states, _, rewards, *_ = buffer.sample(3, "cpu")
    assert len(buffer) == 3
    assert set(rewards.flatten().tolist()) == {2, 3, 4}
    assert set(states[:, 0].tolist()) == {2, 3, 4}


def test_training_updates_and_is_reproducible_without_django(tmp_path):
    market = MarketConfig(num_players=8, max_steps=6, shock_step=3)
    config = SACConfig(episodes=2, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    first = train(market, config, 3, tmp_path / "first")
    second = train(market, config, 3, tmp_path / "second")
    assert first["updates"] > 0
    assert first["steps"] == 12
    assert first["history"] == second["history"]
    assert (tmp_path / "first" / "training.csv").is_file()
    assert (tmp_path / "first" / "policy.pt").is_file()
    with pytest.raises(FileExistsError):
        train(market, config, 3, tmp_path / "first")


def test_cancellation_produces_explicit_partial_outcome(tmp_path):
    result = train(MarketConfig(num_players=8, max_steps=4), SACConfig(episodes=2, hidden=16),
                   0, tmp_path / "cancelled", should_stop=lambda: True)
    assert result["cancelled"]
    assert result["steps"] == 0
    assert not result["history"]


def test_fixed_fee_sac_removes_the_dimension_and_trains_with_zero_fee_actions(tmp_path):
    market = MarketConfig(num_players=8, max_steps=6, shock_step=3)
    config = SACConfig(control_fee=False, episodes=2, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    fees = []
    result = train(market, config, 0, tmp_path / "model", callback=lambda row, info: fees.append(info["fee_rate"]))
    assert result["updates"] > 0
    assert fees == pytest.approx([market.initial_fee] * config.episodes, abs=1e-8)
    env = MultiStoreEnv(market)
    observation, _ = env.reset(seed=10)
    agent, metadata = SACAgent.load(tmp_path / "model/policy.pt", env)
    assert agent.action_dim == 2 * env.n
    assert agent.target_entropy == -2 * env.n
    assert metadata["sac"]["control_fee"] is False
    for _ in range(market.max_steps):
        action = agent.act(observation)
        assert action[-1] == 0
        observation, *_ = env.step(action)
        assert env.fee_rate == pytest.approx(market.initial_fee, abs=1e-8)
    for row in result["history"]:
        assert row["updates"] >= 0
