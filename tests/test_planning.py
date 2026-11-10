"""Checks grouped by maintained responsibility."""
from copy import deepcopy
from dataclasses import asdict
import json
import pickle
import numpy as np
import pytest
from experiments.market.economy import TaskConfig
from experiments.market.inventory import FactorialMarket
from experiments.market.controllers import JointController
from experiments.market.continuing import translate
from experiments.market.continuing import StationaryMarket,Regime
from experiments.market_planner import MarketPlanner
from experiments.market.controllers import AnchoredPI


def test_nominal_physics_and_inputs_match_old_positive_reference():
    a=StationaryMarket();b=FactorialMarket();x,_=a.reset(seed=31);y,_=b.reset(seed=31)
    np.testing.assert_array_equal(x,y);g=np.random.default_rng(8)
    for _ in range(20):
        action=g.uniform(-.05,.05,4).astype(np.float32)
        x,r,_,_,i=a.step(action);y,s,_,_,j=b.step(action)
        np.testing.assert_array_equal(x,y);assert r==s and i==j


def test_stationary_encoding_matches_deployment_translation_and_batches():
    a=StationaryMarket(representation='stationary');b=StationaryMarket();obs,_=a.reset(seed=31);b.reset(seed=31)
    for _ in range(10):obs,*_=a.step(np.zeros(4,np.float32))
    np.testing.assert_array_equal(a.encode(obs),b.encode(translate(a,obs)))
    np.testing.assert_array_equal(a.encode(np.stack([obs,obs])),np.stack([a.encode(obs)]*2))
    np.testing.assert_array_equal(a.observe(),obs)


def test_regimes_preserve_draws_and_stock_capacity_and_holding_scale():
    a=StationaryMarket();b=StationaryMarket(regime=Regime(0.,3.,1.));a.reset(seed=31);b.reset(seed=31)
    for name in ('arrivals','productivity','preference'):np.testing.assert_array_equal(getattr(a.env,name),getattr(b.env,name))
    np.testing.assert_array_equal(b.env.recipe_schedule,np.tile(b.env.recipe_schedule[0],(len(b.env.recipe_schedule),1)))
    assert b.stock_cap.sum()==72 and b.config.production_capacity==24
    a.env.stock[:]=b.env.stock;a.shop_ages[:]=b.shop_ages
    _,_,_,_,x=a.step(np.zeros(4,np.float32));_,_,_,_,y=b.step(np.zeros(4,np.float32))
    assert x['goods_residual']<1e-7 and y['goods_residual']<1e-7


@pytest.mark.parametrize('particles',[1,4])
def test_vector_forecasts_match_scalar_market_physics(particles):
    env=StationaryMarket();env.reset(seed=31)
    for _ in range(7):env.step(np.array([.01,-.02,-.01,.02],np.float32))
    model=env.public_model(env.frame());state=model.env.unpack(model.env.frame())
    planner=MarketPlanner(env,horizon=5,particles=particles);forecast=planner.forecast(state)
    actions=np.array([[.05,0.,-.05,.05],[-.05,.05,0.,0.]],np.float32)
    _,paths=planner.score_actions(state,actions,forecast,True)
    for index,action in enumerate(actions):
        for particle in range(particles):
            for branch in (0,1):
                fork=deepcopy(model);fork.env.arrivals[:6]=np.r_[forecast['demand'][particle],8.]
                fork.env.recipe_schedule[:5]=forecast['recipes'][particle]
                fork.env.productivity[:5]=forecast['productivity'][particle];fork.env.preference[:5]=forecast['preference'][particle]
                feedback=JointController(fork,'joint',gain=3.,stock_gain=.5,integral_gain=.03,allocation_gain=1.)
                for t in range(5):
                    move=action if t==0 else feedback.act(fork.observe()) if branch else np.zeros(4,np.float32)
                    _,reward,_,_,info=fork.step(move)
                    point=paths[t]
                    for name in ('prices','stock','household'):np.testing.assert_allclose(point[name][index,branch,particle],getattr(fork.env,name),atol=2e-5,rtol=1e-7)
                    np.testing.assert_allclose(point['money'][index,branch,particle],fork.env.money,atol=2e-5)
                    np.testing.assert_allclose(point['reward'][index,branch,particle],reward,atol=2e-5)


@pytest.mark.parametrize('particles',[1,4])
def test_planner_has_no_actual_future_access_and_does_not_mutate_market(particles):
    a=StationaryMarket();b=StationaryMarket();x,_=a.reset(seed=31);y,_=b.reset(seed=31)
    b.env.arrivals[1:]=99;b.env.recipe_schedule[1:]=99;b.env.productivity[:]=.001;b.env.preference[:]=99
    original=pickle.dumps(a)
    aa=MarketPlanner(a,horizon=4,particles=particles).act(x);bb=MarketPlanner(b,horizon=4,particles=particles).act(y)
    np.testing.assert_array_equal(aa,bb);assert a.action_space.contains(aa) and pickle.dumps(a)==original


def test_chunked_trace_hydration_checks_parts_and_full_content(tmp_path):
    import hashlib
    from scripts.reproduce import hydrate_trace
    parts=[b'first compressed bytes',b'last compressed bytes'];(tmp_path/'steps-parts').mkdir()
    entries=[]
    for i,block in enumerate(parts):
        name=f'steps-parts/part-{i:05d}.bin';(tmp_path/name).write_bytes(block)
        entries.append(dict(path=name,size=len(block),sha256=hashlib.sha256(block).hexdigest()))
    (tmp_path/'trace-parts.json').write_text(json.dumps(dict(format='lossless-compressed-byte-parts-v1',file='steps.csv.gz',parts=entries,size=sum(map(len,parts)),sha256=hashlib.sha256(b''.join(parts)).hexdigest())))
    hydrate_trace(tmp_path);assert (tmp_path/'steps.csv.gz').read_bytes()==b''.join(parts)
    (tmp_path/'steps.csv.gz').unlink();(tmp_path/entries[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Trace part mismatch'):hydrate_trace(tmp_path)


def test_anchor_control_matches_old_pi_at_original_anchors():
    a=FactorialMarket();b=FactorialMarket();oa,_=a.reset(seed=91);ob,_=b.reset(seed=91)
    old=JointController(a,'pi',gain=3.,stock_gain=0.,integral_gain=.06)
    new=AnchoredPI(b,bid_anchor=8.5,ask_anchor=10.)
    for _ in range(20):
        x=old.act(oa);y=new.act(ob);np.testing.assert_array_equal(x,y)
        oa,ra,_,_,_=a.step(x);ob,rb,_,_,_=b.step(y)
        np.testing.assert_array_equal(oa,ob);assert ra==rb


def test_lower_ask_anchor_changes_service_control_at_zero_money_error():
    e=FactorialMarket();obs,_=e.reset(seed=91)
    old=AnchoredPI(e,8.5,10.).act(obs);new=AnchoredPI(e,8.5,9.).act(obs)
    np.testing.assert_array_equal(old[:2],new[:2]);assert np.all(new[2:]<old[2:])
    assert e.action_space.contains(new)


@pytest.mark.parametrize('money',[6000.,13000.])
def test_ledger_translation_changes_only_bookkeeping_and_keeps_cash_identity(money):
    e=FactorialMarket();e.reset(seed=91);e.env.money=money;e.env.treasury=7000.
    e.env.minted=money+e.env.treasury-12000.
    e.frames.clear();e.frames.extend(e.frame().copy() for _ in range(e.config.history))
    obs=e.observe();before=obs.copy();adjusted=translate(e,obs)
    np.testing.assert_array_equal(obs,before)
    for raw,new in zip(obs.reshape(e.config.history,e.frame_size),adjusted.reshape(e.config.history,e.frame_size)):
        a=e.unpack(raw);b=e.unpack(new)
        assert b['treasury']>=0 and b['minted']>=0
        np.testing.assert_allclose(b['money']+b['treasury']-12000.,b['minted'])
        for key in a:
            if key not in ('treasury','minted'):np.testing.assert_array_equal(a[key],b[key])
