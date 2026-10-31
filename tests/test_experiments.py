import csv
import json
import shutil

import numpy as np
import pytest
import torch

from experiments.baselines import Baseline
from experiments.evaluate import evaluate, format_mean_std, summarize
from experiments.plot_results import plot_results
from experiments.protocol import load_protocol
from rl.config import MarketConfig, SACConfig
from rl.environment import MultiStoreEnv
from rl.training import train


def test_fixed_random_and_adaptive_actions():
    env = MultiStoreEnv(MarketConfig(num_players=10, max_steps=3))
    observation, _ = env.reset(seed=7)
    initial_prices = observation[:2*env.n].copy()
    fixed = Baseline("fixed", env)
    observation, *_ = env.step(fixed.act(observation))
    np.testing.assert_array_equal(initial_prices, observation[:2*env.n])
    for method in ("random", "adaptive"):
        assert env.action_space.contains(Baseline(method, env).act(observation))
    # An excess of sellers should reduce the shop's purchase offer.
    observation[2*env.n:3*env.n] = env.config.num_players
    observation[3*env.n:4*env.n] = env.config.num_players
    action = Baseline("adaptive", env).act(observation)
    assert (action[:-1:2] < 0).all()
    assert (action[1:-1:2] > 0).all()
    assert action[-1] == 0


def test_end_to_end_held_out_evaluation_and_plot_regeneration(tmp_path):
    market = MarketConfig(num_players=8, max_steps=5, shock_step=3)
    sac = SACConfig(episodes=2, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    train(market, sac, 0, tmp_path / "train")
    protocol = {"market": market.to_dict(), "evaluation_seeds": [100, 101], "adaptive_gain": 0.5}
    output = tmp_path / "evaluation"
    summary = evaluate(market, protocol, [tmp_path / "train" / "policy.pt"], output)
    assert {row["method"] for row in summary} == {"fixed", "random", "adaptive", "sac"}
    with (output / "steps.csv").open() as stream:
        steps = list(csv.DictReader(stream))
    assert len(steps) == 4 * 2 * 5
    fixed = [row for row in steps if row["method"] == "fixed" and row["evaluation_seed"] == "100"]
    assert len({row["cpi"] for row in fixed}) == 1
    assert plot_results(output).is_file()
    assert (output / "comparison.svg").is_file()
    with pytest.raises(ValueError, match="requires"):
        evaluate(market, protocol, [], tmp_path / "bad")


def test_summary_averages_scenarios_within_training_seed_first():
    rows = []
    from experiments.evaluate import SUMMARY_METRICS
    for seed, values in ((0, [0, 2]), (1, [10, 12])):
        for eval_seed, value in enumerate(values):
            rows.append({"method": "sac", "training_seed": seed, "evaluation_seed": eval_seed,
                         **{metric: value for metric in SUMMARY_METRICS}})
    result = summarize(rows)[0]
    assert result["n_units"] == 2
    assert result["return_mean"] == 6
    assert result["return_std"] == pytest.approx(np.std([1, 11], ddof=1))


def test_single_training_seed_has_no_sample_standard_deviation():
    from experiments.evaluate import SUMMARY_METRICS
    rows = [{"method": "sac", "training_seed": 0, "evaluation_seed": seed,
             **{metric: value for metric in SUMMARY_METRICS}}
            for seed, value in enumerate([1, 3])]
    result = summarize(rows)[0]
    assert result["return_mean"] == 2
    assert result["return_std"] is None
    assert "SD unavailable" in format_mean_std(result["return_mean"], result["return_std"])


def test_protocol_rejects_held_out_seed_leakage(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"training_seeds": [0], "evaluation_seeds": [10_000]}))
    with pytest.raises(ValueError, match="held out"):
        load_protocol(path)


def test_evaluation_rejects_checkpoint_with_no_gradient_updates(tmp_path):
    market = MarketConfig(num_players=8, max_steps=3, shock_step=None)
    config = SACConfig(episodes=1, hidden=16, batch_size=8, warmup_steps=10, replay_size=30)
    result = train(market, config, 0, tmp_path / "model")
    assert result["steps"] == 3 and result["updates"] == 0
    checkpoint = torch.load(tmp_path / "model/policy.pt", weights_only=True)
    assert checkpoint["gradient_updates"] == 0
    protocol = {"market": market.to_dict(), "evaluation_seeds": [100], "adaptive_gain": 0.5}
    with pytest.raises(ValueError, match="no gradient updates"):
        evaluate(market, protocol, [tmp_path / "model/policy.pt"], tmp_path / "evaluation", methods=("sac",))
    with pytest.raises(ValueError, match="no gradient updates"):
        evaluate(market, protocol, [tmp_path / "model/policy.pt"], tmp_path / "override",
                 methods=("sac",), allow_unverified=True)


def test_legacy_checkpoint_requires_update_evidence_or_explicit_override(tmp_path):
    market = MarketConfig(num_players=8, max_steps=5, shock_step=None)
    config = SACConfig(episodes=1, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    result = train(market, config, 0, tmp_path / "legacy")
    path = tmp_path / "legacy/policy.pt"
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["gradient_updates"]
    torch.save(checkpoint, path)
    protocol = {"market": market.to_dict(), "evaluation_seeds": [100], "adaptive_gain": 0.5}
    evaluate(market, protocol, [path], tmp_path / "verified", methods=("sac",))
    verified = json.loads((tmp_path / "verified/config.json").read_text())["checkpoints"][0]
    assert verified["gradient_updates"] == result["updates"]
    assert verified["update_verification"] == "legacy_outcome"
    (path.parent / "outcome.json").write_text(json.dumps({"steps": result["steps"] + 1, "updates": result["updates"]}))
    with pytest.raises(ValueError, match="inconsistent"):
        evaluate(market, protocol, [path], tmp_path / "mismatched", methods=("sac",))
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    shutil.copy2(path, orphan / "policy.pt")
    with pytest.raises(ValueError, match="no gradient-update record"):
        evaluate(market, protocol, [orphan / "policy.pt"], tmp_path / "missing", methods=("sac",))
    with pytest.warns(RuntimeWarning, match="explicitly unverified"):
        evaluate(market, protocol, [orphan / "policy.pt"], tmp_path / "unverified", methods=("sac",),
                 allow_unverified=True)
    recorded = json.loads((tmp_path / "unverified/config.json").read_text())["checkpoints"][0]
    assert recorded["gradient_updates"] is None
    assert recorded["update_verification"] == "unverified"


def test_fixed_fee_evaluation_is_labeled_and_cannot_be_pooled_with_full_sac(tmp_path):
    market = MarketConfig(num_players=8, max_steps=5, shock_step=None)
    full = SACConfig(episodes=1, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    fixed = SACConfig(episodes=1, hidden=16, batch_size=4, warmup_steps=4, replay_size=30, control_fee=False)
    train(market, full, 0, tmp_path / "full")
    train(market, fixed, 1, tmp_path / "fixed")
    protocol = {"market": market.to_dict(), "evaluation_seeds": [100, 101], "adaptive_gain": 0.5}
    paths = [tmp_path / "full/policy.pt", tmp_path / "fixed/policy.pt"]
    with pytest.raises(ValueError, match="fixed-fee and fee-control"):
        evaluate(market, protocol, paths, tmp_path / "mixed", methods=("sac",))
    summary = evaluate(market, protocol, [paths[1]], tmp_path / "fixed-eval", methods=("sac",))
    assert summary[0]["method"] == "sac_fixed_fee"
    assert summary[0]["aggregation_unit"] == "training_seed"
    assert summary[0]["return_std"] is None
    model = json.loads((tmp_path / "fixed-eval/config.json").read_text())["checkpoints"][0]
    assert model["control_fee"] is False
    assert model["sac_settings"]["control_fee"] is False
    assert model["gradient_updates"] > 0
    assert plot_results(tmp_path / "fixed-eval").is_file()


def test_evaluation_does_not_pool_different_training_settings_or_budgets(tmp_path):
    market = MarketConfig(num_players=8, max_steps=5, shock_step=None)
    config = SACConfig(episodes=2, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    different = SACConfig(episodes=2, hidden=32, batch_size=4, warmup_steps=4, replay_size=30)
    train(market, config, 0, tmp_path / "full")
    train(market, different, 1, tmp_path / "different")
    steps = 0
    def stop_after_nine_steps():
        nonlocal steps
        steps += 1
        return steps > 9
    result = train(market, config, 2, tmp_path / "partial", should_stop=stop_after_nine_steps)
    assert result["cancelled"] and result["steps"] == 9 and result["updates"] > 0
    protocol = {"market": market.to_dict(), "evaluation_seeds": [100], "adaptive_gain": 0.5}
    with pytest.raises(ValueError, match="different training settings"):
        evaluate(market, protocol, [tmp_path / "full/policy.pt", tmp_path / "different/policy.pt"],
                 tmp_path / "mixed-config", methods=("sac",))
    with pytest.raises(ValueError, match="different training budgets"):
        evaluate(market, protocol, [tmp_path / "full/policy.pt", tmp_path / "partial/policy.pt"],
                 tmp_path / "mixed-budget", methods=("sac",))
