"""Checks grouped by maintained responsibility."""
import json
from pathlib import Path
import pytest
from experiments.evaluate import METRICS
from experiments.robustness import evaluate_grid, initialize, mixture_rows, pareto_front, select_operating_points, validate_protocol
from rl.config import MarketConfig,SACConfig
from rl.training import train
import csv
from dataclasses import asdict
import numpy as np
import torch
from experiments.robust_training import sample_market, train_randomized
from rl.config import MarketConfig, SACConfig
from rl.sac import SACAgent


def summary(method,error,volume,price):
    return {"method":method,"mean_gold_deviation_mean":error,"total_volume_mean":volume,
            "mean_item_price_change_mean":price}


def test_mixture_averages_conditions_within_model_and_market_and_rejects_missing():
    scenarios = [{"name":"a"},{"name":"b"}]
    rows = [{"scenario":sc,"method":"sac","training_seed":0,"evaluation_seed":10,
             **{m:v for m in METRICS}} for sc,v in (("a",2.),("b",4.))]
    combined = mixture_rows(rows,scenarios)
    assert len(combined)==1 and combined[0]["common_return"]==3.
    with pytest.raises(ValueError,match="missing/duplicate"):
        mixture_rows(rows[:-1],scenarios)


def test_frozen_operating_points_respect_caps_and_matching_tolerance():
    specs = [{"method":"f0","family":"feedback"},{"method":"f1","family":"feedback"},
             {"method":"n0","family":"nominal","weight":0.},
             {"method":"r0","family":"randomized","weight":0.}]
    stats = [summary("f0",.02,100,.02),summary("f1",.04,200,.03),
             summary("n0",.025,150,.01),summary("r0",.08,300,.02)]
    points,matches = select_operating_points(stats,specs,"f0",[.01,.03,.1],.006)
    assert len(points)==9 and len(matches)==6
    nominal = next(r for r in points if r["family"]=="nominal" and r["cap"]==.03)
    assert nominal["method"]=="n0" and nominal["validation_feasible"]
    tight = next(r for r in points if r["family"]=="randomized" and r["cap"]==.01)
    assert not tight["validation_feasible"] and tight["method"]=="f0"
    pair = next(r for r in matches if r["family"]=="nominal" and r["cap"]==.03)
    assert pair["feedback_method"]=="f0"
    unmatched = next(r for r in matches if r["family"]=="randomized" and r["cap"]==.1)
    assert unmatched["feedback_method"] is None


def test_pareto_front_uses_currency_volume_and_per_item_price_stability():
    a = summary("a",.02,100,.01)
    dominated = summary("bad",.03,90,.02)
    tradeoff = summary("tradeoff",.04,150,.01)
    price_tradeoff = summary("smooth",.03,90,.005)
    assert {r["method"] for r in pareto_front([a,dominated,tradeoff,price_tradeoff])}=={"a","tradeoff","smooth"}


def test_declared_protocol_rejects_leakage_and_semantic_changes():
    path = Path("experiments/configs/robustness.json")
    config = json.loads(path.read_text())
    validate_protocol(config)
    config["evaluation_seeds"] = config["validation_seeds"]
    with pytest.raises(ValueError,match="leakage"):
        validate_protocol(config)
    config = json.loads(path.read_text())
    config["test_scenarios"][0]["changes"] = {"num_players":100}
    with pytest.raises(ValueError,match="semantics"):
        validate_protocol(config)


def test_parallel_evaluation_is_frozen_and_reproducible(tmp_path):
    market = MarketConfig(num_players=8,max_steps=4,shock_step=2,reward_mode="balanced_v1")
    sac = SACConfig(episodes=2,hidden=16,batch_size=4,warmup_steps=4,replay_size=32,control_fee=False)
    folder = tmp_path/"model"
    train(market,sac,0,folder)
    params = dict(buy_fraction=.75,sell_fraction=.25,gain=1.)
    specs = [dict(method="feedback",family="feedback",parameters=params),
             dict(method="sac",family="nominal",parameters=params,weight=.25)]
    models = [dict(path=str(folder/"policy.pt"),family="nominal",training_seed=0)]
    scenarios = [dict(name="nominal",changes={}),dict(name="shift",changes=dict(price_sensitivity_low=.9,price_sensitivity_high=1.5))]
    original = (folder/"policy.pt").read_bytes()
    a = evaluate_grid(market,specs,models,scenarios,[150010,150011],2)
    b = evaluate_grid(market,specs,models,scenarios,[150010,150011],1)
    assert a==b and len(a)==8
    assert (folder/"policy.pt").read_bytes()==original


def test_protocol_initialization_records_prior_market_reuse(tmp_path,monkeypatch):
    source = Path("experiments/configs/robustness.json")
    config = json.loads(source.read_text())
    config["base_config"] = str(Path(config["base_config"]).resolve())
    protocol = tmp_path/"input.json"
    protocol.write_text(json.dumps(config))
    earlier = tmp_path/"results/earlier"
    earlier.mkdir(parents=True)
    (earlier/"config.json").write_text(json.dumps({"evaluation_seeds":config["evaluation_seeds"][:1]}))
    monkeypatch.chdir(tmp_path)
    root = tmp_path/"results/new"
    initialize(root,protocol)
    record = json.loads((root/"protocol-record.json").read_text())
    assert record["test_episodes_seen"]==0
    assert record["prior_protocol_market_seed_collisions"]==["results/earlier/config.json"]
    with pytest.raises(FileExistsError):
        initialize(root,protocol)


def test_domain_rng_is_reproducible_and_only_changes_response_parameters():
    base = MarketConfig(reward_mode="balanced_v1")
    distribution = dict(center_low=.8,center_high=1.2,half_width=.3,rng_offset=9000000)
    a,b = np.random.default_rng(3),np.random.default_rng(3)
    centers = []
    for _ in range(50):
        first,center = sample_market(base,distribution,a)
        second,again = sample_market(base,distribution,b)
        assert first==second and center==again and .8<=center<=1.2
        before,after = base.to_dict(),first.to_dict()
        for field in before.keys()-{"price_sensitivity_low","price_sensitivity_high"}:
            assert before[field]==after[field]
        centers.append(center)
    assert np.std(centers)>.05
    with pytest.raises(ValueError):
        sample_market(base,dict(center_low=.2,center_high=1.2,half_width=.3),a)


def test_constant_domain_matches_standard_sac_training_and_records_actual_domains(tmp_path):
    base = MarketConfig(num_players=8,max_steps=4,shock_step=2,reward_mode="balanced_v1")
    sac = SACConfig(episodes=3,hidden=16,batch_size=4,warmup_steps=4,replay_size=32,control_fee=False)
    train(base,sac,1,tmp_path/"standard")
    distribution = dict(center_low=1.,center_high=1.,half_width=.3,rng_offset=9000000)
    train_randomized(base.to_dict(),asdict(sac),1,distribution,tmp_path/"domain")
    first = torch.load(tmp_path/"standard/policy.pt",weights_only=True)
    second = torch.load(tmp_path/"domain/policy.pt",weights_only=True)
    assert all(torch.equal(v,second["policy"][k]) for k,v in first["policy"].items())
    assert first["gradient_updates"]==second["gradient_updates"]==9
    assert second["training_distribution"]==distribution
    agent,metadata = SACAgent.load(tmp_path/"domain/policy.pt")
    assert agent.env.config==base and metadata["trained_steps"]==12
    rows = list(csv.DictReader((tmp_path/"domain/training.csv").open()))
    assert len(rows)==3 and all(float(r["response_center"])==1. for r in rows)
