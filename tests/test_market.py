"""Checks grouped by maintained responsibility."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import numpy as np
import pytest
from experiments.market.economy import TaskMarket,TaskConfig,CASES
from experiments.market.controllers import FeedbackController
from experiments.market.economy import TaskMarket,TaskConfig
from experiments.market.inventory import InventoryParameters,InventoryMarket,remove_fifo,age_and_expire
from experiments.market.controllers import InventoryController
from experiments.market.economy import TaskConfig
from experiments.market.inventory import InventoryMarket
from experiments.market.inventory import FactorialMarket,VARIANTS
from experiments.market.controllers import JointController


@pytest.mark.parametrize('case',list(CASES))
def test_every_feature_has_paired_exogenous_streams_and_conserved_ledgers(case):
    c=TaskConfig(max_steps=12);a=TaskMarket(c,case);b=TaskMarket(c,'base')
    oa,_=a.reset(seed=9);ob,_=b.reset(seed=9)
    np.testing.assert_array_equal(a.arrivals,b.arrivals)
    np.testing.assert_array_equal(a.productivity,b.productivity)
    np.testing.assert_array_equal(a.preference,b.preference)
    np.testing.assert_array_equal(a.recipe_schedule,b.recipe_schedule)
    rng=np.random.default_rng(91);arrivals=completed=expired=0.
    for _ in range(12):
        before=pickle.dumps(a)
        with pytest.raises(ValueError):a.step(np.full(4,np.nan,np.float32))
        assert pickle.dumps(a)==before
        action=rng.uniform(-.05,.05,4).astype(np.float32);prices=a.prices.copy()
        obs,reward,term,trunc,info=a.step(action)
        assert np.isfinite(a.encode(obs)).all() and a.encode(np.stack([obs,obs])).shape==(2,a.observation_space.shape[0])
        assert max(abs(a.prices/prices-1))<=.050001
        assert abs(info['cash_residual'])<1e-7 and info['goods_residual']<1e-7
        assert reward==info['reward'] and not term
        arrivals+=info['arrivals'];completed+=info['completed'];expired+=info['expired']
    np.testing.assert_allclose(arrivals-completed-expired,info['pending_tasks'],atol=1e-8)
    assert trunc
    with pytest.raises(RuntimeError):a.step(np.zeros(4,np.float32))


def test_mechanism_checks_and_delayed_quotes_are_honored():
    env=TaskMarket(TaskConfig(max_steps=12),'delay');env.reset(seed=19)
    env.step(np.zeros(4,np.float32));promised=float(env.commitments[-1].sum())
    for _ in range(3):_,_,_,_,info=env.step(np.array([-.05,-.05,0.,0.],np.float32))
    np.testing.assert_allclose(info['payout'],promised)


def test_waiting_defers_but_does_not_erase_rejected_tasks():
    a,b=TaskMarket(TaskConfig(max_steps=12),'waiting'),TaskMarket(TaskConfig(max_steps=12),'base')
    for env in (a,b):env.reset(seed=19);env.acceptance=lambda asks,preference=None:np.zeros(2)
    initial=float(a.tasks.sum());_,_,_,_,ia=a.step(np.zeros(4,np.float32));_,_,_,_,ib=b.step(np.zeros(4,np.float32))
    assert ia['expired']==0 and ia['pending_tasks']==initial and ib['expired']==initial
    a.step(np.zeros(4,np.float32));_,_,_,_,ia=a.step(np.zeros(4,np.float32));assert ia['expired']==initial


def test_public_model_and_nonoracle_rules_cannot_see_hidden_slope_or_future():
    a,b=TaskMarket(case='identification'),TaskMarket(case='identification')
    oa,_=a.reset(seed=19);ob,_=b.reset(seed=19);a.slope=2.;b.slope=12.
    b.arrivals[1:]=99.;b.productivity[:]=.1;b.preference[:]=9.;b.recipe_schedule[1:]=99.
    for method in ('p','pi','predictive','adaptive'):
        np.testing.assert_array_equal(FeedbackController(a,method).act(oa),FeedbackController(b,method).act(ob))
    before=pickle.dumps(a);model=a.public_model(a.frame(),5.);model.step(np.zeros(4,np.float32));assert pickle.dumps(a)==before
    a.slope=2.;b.slope=12.
    np.testing.assert_array_equal(a.acceptance(np.full(2,10.)),b.acceptance(np.full(2,10.)))
    assert np.all(a.acceptance(np.full(2,10.5))>b.acceptance(np.full(2,10.5)))
    _,_,_,_,info=a.step(np.array([0.,0.,.05,.05],np.float32));assert info['price_cost']>0


@pytest.mark.parametrize('change',[dict(lag=0),dict(initial_ask=8.),dict(mean_tasks=0.),dict(history=0)])
def test_invalid_market_rejected(change):
    with pytest.raises(ValueError):replace(TaskConfig(),**change)


def test_fifo_preserves_age_and_expiry_conserves_goods():
    bins=np.array([[4.,3.],[2.,1.],[1.,2.]])
    before=bins.sum(0).copy();taken=remove_fifo(bins,[2.,2.])
    np.testing.assert_array_equal(taken,[[0.,0.],[1.,0.],[1.,2.]])
    np.testing.assert_allclose(before,bins.sum(0)+taken.sum(0))
    following,waste=age_and_expire(bins.copy(),3)
    np.testing.assert_allclose(following.sum(0)+waste,bins.sum(0))
    for _ in range(3):following,w=age_and_expire(following,0)
    np.testing.assert_allclose(following.sum(0),bins.sum(0)-waste)


@pytest.mark.parametrize('case',['capacity','delay_capacity'])
def test_nonperishable_wrapper_preserves_original_market(case):
    c=TaskConfig(max_steps=12);a=InventoryMarket(c,case=case);b=TaskMarket(c,case)
    a.reset(seed=8);b.reset(seed=8);rng=np.random.default_rng(8)
    for _ in range(12):
        action=rng.uniform(-.05,.05,4).astype(np.float32)
        _,ra,_,_,ia=a.step(action);_,rb,_,_,ib=b.step(action)
        np.testing.assert_allclose(a.env.frame(),b.frame(),atol=1e-7)
        np.testing.assert_allclose(ra,rb,atol=1e-10)
        assert ia['waste']==0


@pytest.mark.parametrize('life',[0,3,6])
@pytest.mark.parametrize('case',['capacity','delay_capacity'])
def test_age_inventory_ledgers_and_model_reconstruction(life,case):
    e=InventoryMarket(TaskConfig(max_steps=12),InventoryParameters(buffer_periods=3.,capacity_ratio=.9,shelf_life=life),case)
    obs,_=e.reset(seed=10);rng=np.random.default_rng(11)
    for _ in range(12):
        before=pickle.dumps(e)
        with pytest.raises(ValueError):e.step(np.full(4,np.nan,np.float32))
        assert pickle.dumps(e)==before
        obs,reward,_,_,info=e.step(rng.uniform(-.05,.05,4).astype(np.float32))
        np.testing.assert_allclose(e.shop_ages.sum(0),e.env.stock,atol=1e-7)
        np.testing.assert_allclose(e.house_ages.sum(0),e.env.household,atol=1e-7)
        assert np.isfinite(e.encode(np.stack([obs,obs]))).all()
        assert max(info['goods_residual'],abs(info['cash_residual']))<1e-7
        np.testing.assert_allclose(reward,-sum(info[k] for k in ['money_cost','service_cost','inventory_cost','price_cost','waste_cost']))
    model=e.public_model(e.frame());model.step(np.zeros(4,np.float32))


def test_inflight_goods_do_not_expire_and_both_owner_locations_pay_waste():
    e=InventoryMarket(TaskConfig(max_steps=12),InventoryParameters(shelf_life=1),'delay_capacity');e.reset(seed=19)
    e.step(np.zeros(4,np.float32));first=e.env.pipeline[-1].copy();quote=float(e.env.commitments[-1].sum())
    e.step(np.zeros(4,np.float32));np.testing.assert_allclose(e.env.pipeline[-2],first)
    e.step(np.zeros(4,np.float32));_,_,_,_,info=e.step(np.zeros(4,np.float32));np.testing.assert_allclose(info['payout'],quote)
    e=InventoryMarket(TaskConfig(max_steps=12),InventoryParameters(shelf_life=1),'capacity');e.reset(seed=19)
    e.env.tasks.fill(0.);e.env.household[:]=3.;e.house_ages[0]=3.
    _,_,_,_,info=e.step(np.zeros(4,np.float32))
    assert info['house_waste']==6. and info['shop_waste']>0
    np.testing.assert_allclose(info['waste'],info['house_waste']+info['shop_waste'])


def test_clearance_responds_to_age_not_only_total_stock_and_no_future_peek():
    a=InventoryMarket(parameters=InventoryParameters(shelf_life=3),case='capacity');b=InventoryMarket(parameters=InventoryParameters(shelf_life=3),case='capacity')
    a.reset(seed=19);b.reset(seed=19);b.shop_ages[2]=b.shop_ages[0];b.shop_ages[0]=0.
    for e in (a,b):e.frames.clear();e.frames.extend(e.frame().copy() for _ in range(e.config.history))
    x=InventoryController(a,'clearance',age_gain=1.).act(a.observe());y=InventoryController(b,'clearance',age_gain=1.).act(b.observe())
    assert np.all(y[2:]<=x[2:]) and np.any(y[2:]<x[2:])
    b.reset(seed=19);b.env.arrivals[1:]=99.;b.env.productivity[:]=.01;b.env.preference[:]=9.;b.env.recipe_schedule[1:]=99.
    np.testing.assert_array_equal(InventoryController(a,'predictive').act(a.observe()),InventoryController(b,'predictive').act(b.observe()))


@pytest.mark.parametrize('change',[dict(buffer_periods=0.),dict(capacity_ratio=0.),dict(shelf_life=7),dict(max_age=0)])
def test_invalid_inventory_parameters_rejected(change):
    with pytest.raises(ValueError):replace(InventoryParameters(),**change)


def test_exact_reference_equivalence():
    a=FactorialMarket(TaskConfig(max_steps=12));b=InventoryMarket(TaskConfig(max_steps=12),case='capacity')
    oa,_=a.reset(seed=21);ob,_=b.reset(seed=21);np.testing.assert_array_equal(oa,ob)
    rng=np.random.default_rng(21)
    for _ in range(12):
        action=rng.uniform(-.05,.05,4).astype(np.float32)
        oa,ra,_,_,ia=a.step(action);ob,rb,_,_,ib=b.step(action)
        np.testing.assert_array_equal(oa,ob);assert ra==rb and ia==ib


def test_ablation_preserves_exogenous_draws_and_equal_bid_supply():
    markets=[FactorialMarket(variant=v) for v in VARIANTS]
    for e in markets:e.reset(seed=123)
    for e in markets[1:]:
        for name in ('arrivals','productivity','preference'):np.testing.assert_array_equal(getattr(e.env,name),getattr(markets[0].env,name))
    for e in markets:
        np.testing.assert_allclose(e.env.recipe_schedule.sum(1),3.)
        np.testing.assert_allclose(e.env.supply([9.,9.]),markets[0].env.supply([9.,9.]))
    for e in markets[2:]:np.testing.assert_array_equal(e.env.recipe_schedule,np.full_like(e.env.recipe_schedule,1.5))
    shared,separate=markets[:2];before=shared.env.supply([8.,8.]);after=shared.env.supply([9.,8.])
    assert after[1]<before[1]
    assert separate.env.supply([9.,8.])[1]==separate.env.supply([8.,8.])[1]


@pytest.mark.parametrize('variant',list(VARIANTS))
def test_public_models_are_consistent_and_isolated(variant):
    a=FactorialMarket(TaskConfig(max_steps=8),variant);b=FactorialMarket(TaskConfig(max_steps=8),variant)
    a.reset(seed=7);b.reset(seed=7)
    b.env.arrivals[1:]=100;b.env.productivity[:]=.01;b.env.preference[:]=99;b.env.recipe_schedule[1:]=99
    before=pickle.dumps(a)
    for method in ('joint','predictive','rollout'):
        x=JointController(a,method,lookahead=2).act(a.observe())
        y=JointController(b,method,lookahead=2).act(b.observe())
        np.testing.assert_array_equal(x,y);assert a.action_space.contains(x)
    assert pickle.dumps(a)==before
    model=a.public_model(a.frame());assert model.variant==variant and model.features==a.features
    if variant.endswith('fixed'):np.testing.assert_allclose(model.env.recipe_schedule,1.5)


def test_joint_rule_responds_to_recipe_at_same_stock_and_money():
    e=FactorialMarket();e.reset(seed=7);state=e.unpack(e.frame());state['recipes'][:]=[2.,1.]
    c=JointController(e,'joint',gain=1.,stock_gain=0.)
    x=c.proposal(state);state['recipes'][:]=[1.,2.];y=c.proposal(state)
    assert x[0]>x[1] and y[1]>y[0]
