"""Checks grouped by maintained responsibility."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch
from rl.config import MarketConfig, SACConfig
from rl.environment import MultiStoreEnv
from rl.sac import SACAgent
from experiments.baselines import CurrencyFeedbackPolicy
from experiments.behavior_market import Behavior, BehaviorMarket, IntegralFeedback
from dataclasses import replace
from rl.config import MarketConfig
from experiments.behavior_market import BehaviorMarket, BEHAVIORS
from experiments.mechanism_market import MechanismMarket, Mechanism, MECHANISMS
from experiments.behavior_runtime import episode,gini


def small_market(**kwargs):
    return MarketConfig(num_players=12, max_steps=6, shock_step=3, reward_mode="balanced_v1",
                        reward_count=.5, reward_volume=.5, reward_cpi=.05, reward_money=1., **kwargs)


def test_original_behavior_is_exact_legacy_transition_and_rng_compatible():
    a, b = MultiStoreEnv(small_market()), BehaviorMarket(small_market())
    obs_a, _ = a.reset(seed=12)
    obs_b, _ = b.reset(seed=12)
    np.testing.assert_array_equal(obs_a, obs_b)
    rng = np.random.default_rng(1)
    for _ in range(6):
        action = rng.uniform(a.action_space.low, a.action_space.high).astype(np.float32)
        oa, ra, ta, xa, ia = a.step(action)
        ob, rb, tb, xb, ib = b.step(action)
        assert ib.pop("consumed_units") == 0
        assert ia == ib and (ra,ta,xa) == (rb,tb,xb)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(a.player_inv, b.player_inv)
        np.testing.assert_array_equal(a.player_gold, b.player_gold)


@pytest.mark.parametrize("separate", [False, True])
def test_consumption_item_and_currency_accounting_and_independent_resources(separate):
    c = small_market()
    a = BehaviorMarket(c, Behavior(separate_quantity=separate, inventory_demand=True))
    b = BehaviorMarket(c)
    a.reset(seed=5); b.reset(seed=5)
    action = np.zeros(a.action_space.shape, dtype=np.float32)
    for _ in range(6):
        before_inv, before_gold = a.player_inv.sum(), a.player_gold.sum()
        *_, info = a.step(action)
        *_, original = b.step(action)
        assert info["drop_rate"] == original["drop_rate"]
        generated = int(np.rint(a.drop_rate).sum())*c.num_players
        expected = before_inv-info["consumed_units"]+generated-sum(info["buy_volume"])+sum(info["sell_volume"])
        assert a.player_inv.sum() == expected
        assert abs(a.player_gold.sum()-before_gold+info["sink"]) < 1e-7
        assert (a.player_inv >= 0).all() and (a.player_gold >= 0).all()
    inv = a.player_inv.copy()
    with pytest.raises(RuntimeError):
        a.step(action)
    np.testing.assert_array_equal(inv, a.player_inv)


def test_separated_quantity_changes_analytical_sell_mean_not_participation_draw():
    c = MarketConfig(num_players=100000, max_steps=1, shock_step=None)
    a, b = BehaviorMarket(c), BehaviorMarket(c, Behavior(separate_quantity=True))
    for env in (a,b):
        env.reset(seed=42)
        env.price_sensitivity.fill(1.)
        env.fee_sensitivity.fill(0.)
        env.store_buy_prices = (env.buy_low+.5*(env.buy_high-env.buy_low))/(1-env.fee_rate)
        env.player_inv.fill(10000)
    legacy, separate = a._sell(), b._sell()
    ratio = 10000/(10000+2*c.avg_qty)
    assert legacy.mean() == pytest.approx(c.avg_qty*.5**2*ratio, abs=.015)
    assert separate.mean() == pytest.approx(c.avg_qty*.5*ratio, abs=.02)


def test_satiation_reduces_purchases_at_identical_prices_and_budget():
    c = MarketConfig(num_players=20000, max_steps=1)
    a, b = BehaviorMarket(c, Behavior(inventory_demand=True)), BehaviorMarket(c, Behavior(inventory_demand=True))
    for env in (a,b):
        env.reset(seed=1)
        env.player_gold.fill(1e6)
        env.store_sell_prices = env.sell_low.copy()
    b.player_inv.fill(1000)
    assert b._buy().sum() < .05*a._buy().sum()


def test_pi_integral_accumulates_clamps_resets_and_obeys_action_limits():
    env = BehaviorMarket(small_market())
    obs, _ = env.reset(seed=3)
    policy = IntegralFeedback(env, .75, .25, 1., .05)
    obs[-2] *= 2
    for _ in range(30):
        action = policy.act(obs)
        assert env.action_space.contains(action) and action[-1] == 0
    assert policy.integral == .5
    policy.reset()
    assert policy.integral == 0
    obs[-2] = 0
    for _ in range(30):
        policy.act(obs)
    assert policy.integral == -.5


def test_wealth_metrics_and_trace_reconstruct_episode_means():
    assert gini([1,1,1,1]) == pytest.approx(0)
    assert gini([0,0,0,4]) == pytest.approx(.75)
    assert gini([0,0]) == 0
    env = BehaviorMarket(small_market())
    policy = CurrencyFeedbackPolicy(env, .75, .25, 1.)
    row, trace, players = episode(env, policy, 4, .2, trace=True)
    assert len(trace) == 6 and len(players) == 12
    assert row["mean_gold_error"] == pytest.approx(np.mean([abs(t["gold_ratio"]-1) for t in trace]))
    assert row["max_gold_error"] == pytest.approx(max(abs(t["gold_ratio"]-1) for t in trace))
    assert row["score_per_step"] == pytest.approx(np.mean([t["score_per_step"] for t in trace]))
    assert row["final_gold_p10"] == pytest.approx(np.quantile(players,.1))
    assert row["final_low_gold_fraction"] == np.mean(np.asarray(players)<200)
    assert row["max_price_change"] <= .050001


@pytest.mark.parametrize('new,old',[('neither','original'),('both','inventory')])
def test_reused_cells_exactly_match_old_markets(new,old):
    c=MarketConfig(max_steps=20,shock_step=10,num_players=12)
    a=MechanismMarket(c,MECHANISMS[new]); b=BehaviorMarket(c,BEHAVIORS[old])
    for seed in (0,51):
        oa,_=a.reset(seed=seed); ob,_=b.reset(seed=seed)
        np.testing.assert_array_equal(oa,ob)
        rng=np.random.default_rng(seed+5)
        for _ in range(c.max_steps):
            action=rng.uniform(a.action_space.low,a.action_space.high).astype(np.float32)
            ra=a.step(action); rb=b.step(action)
            np.testing.assert_array_equal(ra[0],rb[0]); assert ra[1:4]==rb[1:4]
            np.testing.assert_array_equal(a.player_inv,b.player_inv)
            np.testing.assert_array_equal(a.player_gold,b.player_gold)
            assert ra[-1]['consumed_units']==rb[-1]['consumed_units']


def test_consumption_only_removes_units_without_satiating_buy_rule():
    c=MarketConfig(num_players=10,max_steps=3,shock_step=None)
    a=MechanismMarket(c,Mechanism(consumption=True,consumption_rate=1))
    b=MechanismMarket(c,Mechanism())
    a.reset(seed=13); b.reset(seed=13)
    a.player_inv[:]=7
    ra=a.step(np.zeros(a.action_space.shape,dtype=np.float32))
    rb=b.step(np.zeros(b.action_space.shape,dtype=np.float32))
    assert ra[-1]['consumed_units']==210
    np.testing.assert_array_equal(a.player_inv,b.player_inv)
    np.testing.assert_array_equal(ra[0],rb[0])


def test_satiation_only_reduces_purchases_without_consuming_inventory():
    c=MarketConfig(num_players=1000,max_steps=2,shock_step=None)
    a=MechanismMarket(c,MECHANISMS['satiation']); b=MechanismMarket(c,MECHANISMS['neither'])
    a.reset(seed=21); b.reset(seed=21)
    for env in (a,b):
        env.player_inv[:]=10000
        env.store_sell_prices[:]=env.sell_low
    assert a._buy().sum()<b._buy().sum()/20
    step=a.step(np.zeros(a.action_space.shape,dtype=np.float32))
    assert step[-1]['consumed_units']==0


def test_invalid_step_does_not_consume_or_advance_rng():
    env=MechanismMarket(MarketConfig(),MECHANISMS['consumption'])
    env.reset(seed=10); env.player_inv[:]=10
    state=env.consumption_rng.bit_generator.state
    with pytest.raises(ValueError):env.step(np.full(env.action_space.shape,np.nan))
    assert env.consumption_rng.bit_generator.state==state
    assert (env.player_inv==10).all()
