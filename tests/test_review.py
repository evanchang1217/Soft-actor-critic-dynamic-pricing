import json
import gzip
from pathlib import Path

import numpy as np
import pytest

from experiments.analyze_reward import reward_analysis
from experiments.baselines import TargetPricePolicy
from experiments.review import (REVIEW_METHODS, load_review_protocol, prepare_models,
                                run_review, select_baselines, summarize_review)
from rl.config import MarketConfig, SACConfig
from rl.environment import MultiStoreEnv
from rl.sac import SACAgent
from rl.training import train


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


def test_review_rejects_full_budget_checkpoint_with_no_updates(tmp_path):
    market = MarketConfig(num_players=8, max_steps=3, shock_step=None)
    config = SACConfig(episodes=1, hidden=16, batch_size=8, warmup_steps=10, replay_size=30)
    result = train(market, config, 0, tmp_path / "model")
    assert result["steps"] == 3 and result["updates"] == 0
    with pytest.raises(ValueError, match="no gradient updates"):
        prepare_models(market, config, [0], [tmp_path / "model/policy.pt"], "sac", tmp_path / "out", [])


def test_baseline_selection_is_reproducible_and_retains_validation_trials():
    market = MarketConfig(num_players=8, max_steps=4, shock_step=2)
    args = (market, [2000, 2001], [0.25, 0.5], [0.5, 0.75], [0.0])
    selection, trials = select_baselines(*args)
    assert (selection, trials) == select_baselines(*args)
    assert len(trials) == 4 * 2
    assert {row["validation_seed"] for row in trials} == {2000, 2001}
    for method in ("tuned_adaptive", "target_price"):
        best = selection["selected"][method]
        assert best["mean_validation_return"] == max(
            c["mean_validation_return"] for c in selection["candidates"] if c["method"] == method)


@pytest.mark.parametrize("patch", [
    {"validation_seeds": [3000]},
    {"validation_seeds": [10000]},
    {"scenarios": [{"name": "nominal", "changes": {}}, {"name": "bad", "changes": {"reward_cpi": 100}}]},
    {"target_buy_fractions": [1.1]},
])
def test_review_protocol_rejects_leakage_and_uncontrolled_changes(tmp_path, patch):
    raw = json.loads(Path("experiments/configs/review-smoke.json").read_text())
    raw.update(patch)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_review_protocol(path)


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


def test_review_summary_separates_market_variation_from_model_variation():
    from experiments.review import REPORT_METRICS
    rows = []
    for model, values in ((0, [0, 2]), (1, [10, 12])):
        for seed, value in enumerate(values):
            rows.append({"scenario": "nominal", "method": "sac_fixed_fee", "training_seed": model,
                         "evaluation_seed": seed, **{metric: value for metric in REPORT_METRICS}})
    row = summarize_review(rows)[0]
    assert row["n_training_seeds"] == 2
    assert row["return_mean"] == 6
    assert row["return_scenario_std"] == pytest.approx(np.std([5, 7], ddof=1))
    assert row["return_model_std"] == pytest.approx(np.std([1, 11], ddof=1))


def test_review_does_not_report_zero_variation_for_single_unit():
    from experiments.review import REPORT_METRICS
    rows = [{"scenario": "nominal", "method": method, "training_seed": seed,
             "evaluation_seed": 3000, **{metric: 1 for metric in REPORT_METRICS}}
            for method, seed in (("sac", 0), ("fixed", ""))]
    for row in summarize_review(rows):
        assert row["return_scenario_std"] is None
        assert row["return_model_std"] is None


def test_complete_review_pipeline_and_reused_checkpoints(tmp_path):
    source = Path("experiments/configs/review-smoke.json")
    output = run_review(source, tmp_path / "review")
    episodes = json.loads((output / "summary.json").read_text())
    assert {row["method"] for row in episodes} == set(REVIEW_METHODS)
    assert {row["scenario"] for row in episodes} == {"nominal", "no_shock", "higher_response"}
    assert (output / "comparison.png").is_file()
    assert (output / "robustness.svg").is_file()
    assert (output / "research-source.zip").is_file()
    components = json.loads((output / "reward_components.json").read_text())
    assert len(components) == 3 * 7
    repeated = run_review(source, tmp_path / "repeated",
                          sorted((output / "models/sac").glob("*/policy.pt")),
                          sorted((output / "models/sac_fixed_fee").glob("*/policy.pt")))
    for filename in ("summary.csv", "episodes.csv", "steps.csv", "reward_components.csv"):
        assert (output / filename).read_bytes() == (repeated / filename).read_bytes()
