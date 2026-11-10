"""Checks grouped by maintained responsibility."""
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.balanced_control import METHODS, load_config, run
from experiments.baselines import CurrencyFeedbackPolicy
from rl.config import MarketConfig
from rl.environment import MultiStoreEnv
from rl.rewards import reward_components
from experiments.baselines import FixedStepAdaptive
import torch
from experiments.balanced_control import load_config, rollout as reference_rollout
from experiments.evaluate import BlendPolicy, paired_interval, control_rollout as rollout
from experiments.robust_training import train_curve
from rl.sac import SACAgent
from rl.training import train
from experiments.analyze_reward import reward_analysis
from experiments.baselines import TargetPricePolicy
from rl.config import MarketConfig, SACConfig


SMOKE = Path("experiments/configs/balanced-control-smoke.json")


def test_currency_setpoint_penalizes_persistent_depletion_and_excess_symmetrically():
    c = MarketConfig(reward_mode="balanced_v1", reward_count=0.5, reward_volume=0.5,
                     reward_cpi=0.05, reward_money=1)
    losses = dict(loss_count=6, loss_volume=3, loss_cpi=0.025, loss_money=0)
    low = reward_components(c, losses, 100000)
    high = reward_components(c, losses, 300000)
    assert low == pytest.approx(dict(count=0.5, volume=0.25, cpi=0.025, money=0.5))
    assert high == pytest.approx(low)
    assert reward_components(c, losses, 200000)["money"] == 0
    with pytest.raises(ValueError):
        reward_components(c, losses)


def test_reward_revision_does_not_change_any_market_transition():
    legacy = MultiStoreEnv(MarketConfig(num_players=8, max_steps=4, shock_step=2))
    revised = MultiStoreEnv(replace(legacy.config, reward_mode="balanced_v1"))
    np.testing.assert_array_equal(legacy.reset(seed=7)[0], revised.reset(seed=7)[0])
    for _ in range(4):
        action = legacy.action_space.low
        first = legacy.step(action)
        second = revised.step(action)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[2:] == second[2:]
        info = first[-1]
        original = -(info["loss_count"] + info["loss_volume"] + 0.5 * info["loss_cpi"] + 0.5 * info["loss_money"])
        assert first[1] == original


def test_currency_feedback_uses_observed_gold_and_respects_action_limits():
    env = MultiStoreEnv(MarketConfig(num_players=8))
    observation, _ = env.reset(seed=9)
    policy = CurrencyFeedbackPolicy(env, 0.75, 0.25, 1)
    rich, poor = observation.copy(), observation.copy()
    rich[-2], poor[-2] = 12000, 4000
    a, b = policy.act(rich), policy.act(poor)
    assert env.action_space.contains(a) and env.action_space.contains(b)
    assert a[-1] == b[-1] == 0
    assert (a[:-1] <= b[:-1]).all()


@pytest.mark.parametrize("patch", [
    {"validation_seeds": [10000, 10001]},
    {"validation_seeds": [73200, 73201]},
    {"adaptive_magnitudes": [0.1]},
    {"currency_gains": [0]},
    {"scenarios": [{"name": "nominal", "changes": {}}, {"name": "bad", "changes": {"initial_gold": 2000}}]},
])
def test_balanced_protocol_rejects_leakage_and_incompatible_comparisons(tmp_path, patch):
    raw = json.loads(SMOKE.read_text())
    raw.update(patch)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_balanced_pipeline_and_frozen_replay(tmp_path):
    trained = run(SMOKE, tmp_path / "trained", workers=2)
    # Follow the saved-config replay path used by the public documentation.
    replayed = run(trained / "config.json", tmp_path / "replayed", reuse_root=trained)
    for name in ("test-episodes.csv", "test-steps.csv.gz", "validation-candidates.csv", "summary.csv",
                 "differences.csv", "reward_components.csv"):
        assert (trained / name).read_bytes() == (replayed / name).read_bytes()
    rows = json.loads((trained / "summary.json").read_text())
    assert {r["method"] for r in rows} == set(METHODS)
    assert all(r["n_markets"] == 2 for r in rows)
    models = json.loads((trained / "models.json").read_text())
    assert len(models) == 6 and all(r["gradient_updates"] == 5 for r in models)
    with gzip.open(trained / "test-steps.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            assert float(row["fee_rate"]) == pytest.approx(0.05, abs=1e-8)
            assert -float(row["reward"]) == pytest.approx(sum(float(row["penalty_" + k]) for k in ("count", "volume", "cpi", "money")))
    assert json.loads((trained / "completion.json").read_text())["test_episodes"] == 52


def test_resolved_protocol_roundtrip_preserves_validated_compatibility_field(tmp_path):
    market, sac, resolved = load_config(SMOKE)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(resolved))
    assert load_config(path) == (market, sac, resolved)
    resolved["adaptive_gain"] = -1
    path.write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="adaptive_gain"):
        load_config(path)
    resolved["adaptive_gain"] = .5
    resolved["unexpected"] = True
    path.write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(path)


def test_fixed_step_feedback_and_action_bounds():
    env = MultiStoreEnv(MarketConfig(num_players=10, max_steps=3))
    observation, _ = env.reset(seed=7)
    policy = FixedStepAdaptive(env, 0.005)
    np.testing.assert_array_equal(policy.act(observation), np.zeros(7))
    observation, *_ = env.step(policy.act(observation))
    observation[6:9] = [0, 5, 10]
    observation[9:12] = [0, 5, 10]
    action = policy.act(observation)
    np.testing.assert_allclose(action, [0.005, -0.005, 0, 0, -0.005, 0.005, 0])
    assert env.action_space.contains(action)
    for invalid in (0, -0.01, 0.051, float('nan')):
        with pytest.raises(ValueError):
            FixedStepAdaptive(env, invalid)


class ConstantAgent:
    def __init__(self, action):
        self.action = action

    def act(self, observation, deterministic):
        assert deterministic
        return self.action.copy()


def test_blend_preserves_both_endpoints_and_physical_action_constraints():
    env = MultiStoreEnv(MarketConfig(num_players=8, max_steps=4))
    obs, _ = env.reset(seed=1)
    feedback = CurrencyFeedbackPolicy(env, .75, .25, 1.)
    action = env.action_space.high.copy()
    action[-1] = 0
    agent = ConstantAgent(action)
    np.testing.assert_array_equal(BlendPolicy(agent, feedback, 0.).act(obs), action)
    np.testing.assert_array_equal(BlendPolicy(agent, feedback, 1.).act(obs), feedback.act(obs))
    for ratio in (.1, .5, 1., 2., 10.):
        obs[-2] = ratio * 8000
        mixed = BlendPolicy(agent, feedback, .25).act(obs)
        assert env.action_space.contains(mixed) and mixed[-1] == 0
        np.testing.assert_allclose(mixed,.75*action+.25*feedback.act(obs),atol=1e-8)
    with pytest.raises(ValueError):
        BlendPolicy(agent, feedback, np.nan)


def test_diagnostic_reward_and_metrics_match_existing_rollout():
    env = MultiStoreEnv(MarketConfig(num_players=8, max_steps=4, reward_mode="balanced_v1"))
    feedback = CurrencyFeedbackPolicy(env,.75,.25,1.)
    expected, _ = reference_rollout(env,feedback,9)
    actual = rollout(env,feedback,9)
    for metric in ("common_return","mean_gold_deviation","max_gold_deviation","total_volume",
                   "mean_participation","mean_cpi","mean_price_change","final_gold_ratio"):
        assert actual[metric] == pytest.approx(expected[metric],abs=1e-12)
    assert actual["common_return"] == pytest.approx(-sum(actual["penalty_"+k] for k in ("count","volume","cpi","money")))
    assert 0 <= actual["mean_item_price_change"] <= env.config.price_offset + 1e-8


def test_crossed_interval_pairs_models_and_markets_and_rejects_incomplete_grid():
    baseline = [{"training_seed":"","evaluation_seed":m,"metric":float(m)} for m in (10,11,12)]
    learned = [{"training_seed":s,"evaluation_seed":m,"metric":float(m+5)}
               for s in (0,1) for m in (10,11,12)]
    result = paired_interval(learned,baseline,"metric",np.random.default_rng(1),1000)
    assert result["difference"] == pytest.approx(5)
    assert result["exploratory_95_low"] == pytest.approx(5)
    assert result["exploratory_95_high"] == pytest.approx(5)
    with pytest.raises(ValueError,match="Incomplete"):
        paired_interval(learned[:-1],baseline,"metric",np.random.default_rng(1),100)


def test_checkpointing_does_not_change_training_or_random_stream(tmp_path):
    source = Path("experiments/configs/balanced-control-smoke.json")
    market, config, _ = load_config(source)
    standard = train(market,replace(config,episodes=4),0,tmp_path/"standard")
    train_curve(source,0,[8,16],tmp_path/"curve",tmp_path/"unused-reference")
    assert (tmp_path/"standard/training.csv").read_bytes() == (tmp_path/"curve/training.csv").read_bytes()
    a = torch.load(tmp_path/"standard/policy.pt",weights_only=True)
    b = torch.load(tmp_path/"curve/steps-16/policy.pt",weights_only=True)
    assert all(torch.equal(a["policy"][k],b["policy"][k]) for k in a["policy"])
    assert a["gradient_updates"] == b["gradient_updates"] == standard["updates"]
    middle, metadata = SACAgent.load(tmp_path/"curve/steps-8/policy.pt")
    assert middle.config.episodes == 2 and metadata["trained_steps"] == 8


def test_target_rule_uses_bounded_observed_prices_and_holds_fee():
    env = MultiStoreEnv(MarketConfig(num_players=8, max_steps=3))
    observation, _ = env.reset(seed=1)
    policy = TargetPricePolicy(env, 0.75, 0.0)
    before = np.abs(observation[:env.n] - policy.buy_target)
    action = policy.act(observation)
    assert env.action_space.contains(action)
    assert action[-1] == 0
    after, *_ = env.step(action)
    assert np.all(np.abs(after[:env.n] - policy.buy_target) <= before + 1e-5)
    assert env.fee_rate == pytest.approx(env.config.initial_fee, abs=1e-8)
    with pytest.raises(ValueError):
        TargetPricePolicy(env, np.nan, 0)


def test_reward_audit_checks_reconstruction_and_component_shares(tmp_path):
    path = tmp_path / "steps.csv"
    path.write_text("method,reward,loss_count,loss_volume,loss_cpi,loss_money\nsac,-3.15,1,2,0.2,0.1\n")
    row = reward_analysis(path, MarketConfig().to_dict())[0]
    assert row["mean_total_penalty"] == pytest.approx(3.15)
    assert sum(row[f"{name}_share_percent"] for name in ("count", "volume", "cpi", "money")) == pytest.approx(100)
    with gzip.open(path.with_suffix(".csv.gz"), "wt") as stream:
        stream.write(path.read_text())
    path.unlink()
    assert reward_analysis(path, MarketConfig().to_dict())[0] == row
    path.write_text("method,reward,loss_count,loss_volume,loss_cpi,loss_money\nsac,-3.15,1,2,0.2,0.1\n")
    path.write_text(path.read_text().replace("-3.15", "-9"))
    with pytest.raises(ValueError, match="does not match"):
        reward_analysis(path, MarketConfig().to_dict())
