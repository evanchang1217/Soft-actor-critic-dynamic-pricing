"""Frozen quality/compute curves, common-state profiling and training replay."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import groupby
from pathlib import Path
import csv
import gzip
import shutil
import time

import numpy as np
import torch

from rl.artifacts import write_json
from .common import read, sha, csv_read, csv_write, aggregate, compare_tables
from .model_error_study import market, load_agent, train_one, NOMINAL
from .adaptive_planning import (SOURCES as PRIOR_SOURCES, IDENTITY, OnlinePlanner,
    initialize as base_initialize, resolve as base_resolve, check, windows)
from .market.controllers import JointController

SOURCES = PRIOR_SOURCES + ('experiments/compute_tradeoff.py',)


def resolve(raw):
    p = base_resolve(raw)
    names = [x['method'] for x in p['planners']]
    assert names and len(names) == len(set(names)) and not set(names) & {'sac', 'pi'}
    assert all(x['horizon'] >= 1 and x['particles'] >= 2 for x in p['planners'])
    assert p['timing_repetitions'] >= 1 and min(p['audit_steps']) > p['minimum_observations']
    assert max(p['audit_steps']) < p['evaluation_steps']
    assert not set(p['audit_markets']) & set(p['evaluation_seeds'])
    assert any(t % p['update_interval'] == 0 for t in p['audit_steps'])
    assert any(t % p['update_interval'] != 0 for t in p['audit_steps'])
    return p


def initialize(root, config, parent):
    p = resolve(read(config)); parent = Path(parent)
    root = base_initialize(root, config, parent)
    previous = read(parent/'protocol.json')
    assert not set(p['evaluation_seeds']+p['audit_markets']) & set(previous['evaluation_seeds'])
    hashes = read(root/'source-hashes.json'); repo = Path(__file__).resolve().parents[1]
    name = 'experiments/compute_tradeoff.py'; shutil.copyfile(repo/name, root/'source'/name); hashes[name] = sha(repo/name)
    write_json(root/'source-hashes.json', hashes)
    copied = read(root/'parent-hashes.json'); models = read(root/'parent/models.json')
    for identity in p['training_replays']:
        model = next(m for m in models if m['size'] == p['calibration_size'] and m['replicate'] == identity['replicate'] and m['seed'] == identity['seed'])
        name = str(Path(model['path']).parent/'training.csv')
        shutil.copyfile(parent/name, root/'parent'/name); copied[name] = sha(root/'parent'/name)
    write_json(root/'parent-hashes.json', copied)
    return root


def selected_model(root, replicate, seed):
    p = read(root/'protocol.json')
    return next(m for m in read(root/'parent/models.json') if m['size'] == p['calibration_size'] and m['replicate'] == replicate and m['seed'] == seed)


def fitted_parameters(root, replicate):
    p = read(root/'protocol.json')
    return next(f['parameters'] for f in read(root/'parent/fits.json') if f['size'] == p['calibration_size'] and f['replicate'] == replicate)


def controller(root, method, replicate=-1, seed=-1):
    p = read(root/'protocol.json')
    if method == 'sac': return load_agent(root/'parent', selected_model(root, replicate, seed))
    if method == 'pi': return JointController(market(NOMINAL, p['evaluation_steps']), 'pi', **read(root/'parent/protocol.json')['pi'])
    spec = next(x for x in p['planners'] if x['method'] == method)
    return OnlinePlanner(fitted_parameters(root, replicate), p, {k:v for k,v in spec.items() if k != 'method'})


def jobs(root):
    p = read(root/'protocol.json'); work = []
    for case in p['cases']:
        for m in p['evaluation_seeds']:
            work.append(dict(case=case, method='pi', replicate=-1, model_seed=-1, market_seed=m))
            for r in p['calibration_replicates']:
                work.extend(dict(case=case, method=x['method'], replicate=r, model_seed=-1, market_seed=m) for x in p['planners'])
                work.extend(dict(case=case, method='sac', replicate=r, model_seed=s, market_seed=m) for s in p['training_seeds'])
    return work


def evaluate_one(root, identity):
    root = Path(root); p = read(root/'protocol.json'); torch.set_num_threads(1)
    env = market(p['cases'][identity['case']], p['evaluation_steps']); obs, _ = env.reset(seed=identity['market_seed'])
    policy = controller(root, identity['method'], identity['replicate'], identity['model_seed'])
    trace = []; decision = observation = 0.
    for t in range(p['evaluation_steps']):
        start = time.process_time()
        action = policy.act(obs, deterministic=True) if identity['method'] == 'sac' else policy.act(obs)
        decision += time.process_time()-start
        nxt, _, _, _, info = env.step(action)
        start = time.process_time()
        if isinstance(policy, OnlinePlanner): policy.observe(obs, nxt)
        observation += time.process_time()-start
        obs = nxt
        trace.append(dict(**identity, step=t+1, **info, **{f'action{i}':float(v) for i,v in enumerate(action)}))
    rows = [dict(**identity, window=name, **summary) for name,summary in windows(p,trace)]
    updates = [dict(**identity, **u) for u in policy.updates] if isinstance(policy, OnlinePlanner) else []
    return rows, trace, updates, dict(**identity, decisions=p['evaluation_steps'], decision_cpu_seconds=decision, observation_cpu_seconds=observation)


def run(root, workers=1):
    root = Path(root); check(root); work = jobs(root)
    if (root/'evaluation-start.json').exists(): raise FileExistsError('Evaluation exists')
    write_json(root/'evaluation-start.json', {n:sha(root/n) for n in ('protocol.json','source-hashes.json','parent-hashes.json')})
    records = []; updates = []; costs = []
    with ProcessPoolExecutor(max_workers=workers) as pool, gzip.open(root/'steps.csv.gz','wt',newline='') as stream:
        futures = [pool.submit(evaluate_one,str(root),j) for j in work]; writer = None
        for i,future in enumerate(futures):
            rows,trace,changes,timing = future.result(); records.extend(rows); updates.extend(changes); costs.append(timing)
            if writer is None: writer = csv.DictWriter(stream,fieldnames=list(trace[0])); writer.writeheader()
            writer.writerows(trace)
            if (i+1)%20 == 0: print(f'Quality evaluation {i+1}/{len(work)}',flush=True)
    csv_write(root/'episodes.csv',records); csv_write(root/'updates.csv',updates); csv_write(root/'rollout-costs.csv',costs)
    csv_write(root/'summary.csv',aggregate(records,['case','method','window']))
    names = ['protocol.json','source-hashes.json','parent-hashes.json','evaluation-start.json','steps.csv.gz','episodes.csv','updates.csv','rollout-costs.csv','summary.csv']
    write_json(root/'data-hashes.json',{n:sha(root/n) for n in names})


def common_states(root):
    p = read(root/'protocol.json'); states = []
    for case in p['cases']:
        for seed in p['audit_markets']:
            env = market(p['cases'][case],p['evaluation_steps']); obs,_ = env.reset(seed=seed); source = controller(root,'pi')
            observers = {r:OnlinePlanner(fitted_parameters(root,r),p,dict(horizon=1,particles=2)) for r in p['calibration_replicates']}
            for step in range(1,max(p['audit_steps'])+1):
                nxt,*_ = env.step(source.act(obs))
                if step in p['audit_steps']:
                    first = next(iter(observers.values()))
                    states.append(dict(state=len(states),case=case,market_seed=seed,step=step,
                        observation=obs.tolist(),next_observation=nxt.tolist(),history=list(first.history),
                        parameters={str(r):dict(v.parameters) for r,v in observers.items()},
                        scope='PI-visited public state/history; counterfactual planner integral zero and private RNG fresh; timing only.'))
                for v in observers.values(): v.observe(obs,nxt)
                obs = nxt
    return states


def prepared(root, method, state, replicate):
    p = read(root/'protocol.json'); policy = controller(root,method,replicate)
    policy.parameters = dict(state['parameters'][str(replicate)])
    policy.planner.env = market(policy.parameters,p['evaluation_steps'])
    policy.history = deque(deepcopy(state['history']),maxlen=p['rolling_window']); policy.steps = state['step']-1
    return policy


def benchmark(root):
    root = Path(root); check(root); torch.set_num_threads(1); p = read(root/'protocol.json')
    if (root/'benchmark-hashes.json').exists(): raise FileExistsError('Benchmark exists')
    states = common_states(root); write_json(root/'common-states.json',states)
    samples = []; observe_samples = []; reps = p['timing_repetitions']
    methods = ['pi','sac']+[x['method'] for x in p['planners']]
    for method in methods:
        groups = [(-1,-1)] if method == 'pi' else [(r,s) for r in p['calibration_replicates'] for s in p['training_seeds']] if method == 'sac' else [(r,-1) for r in p['calibration_replicates']]
        for r,s in groups:
            fixed = controller(root,method,r,s) if method in ('sac','pi') else None
            for state in states:
                obs = np.asarray(state['observation'],np.float32)
                for repetition in range(reps+1):
                    policy = deepcopy(fixed) if method == 'pi' else fixed if method == 'sac' else prepared(root,method,state,r)
                    start = time.process_time_ns()
                    action = policy.act(obs,deterministic=True) if method == 'sac' else policy.act(obs)
                    elapsed = (time.process_time_ns()-start)/1e6
                    if repetition:
                        samples.append(dict(method=method,state=state['state'],replicate=r,model_seed=s,repetition=repetition,cpu_ms=elapsed,
                            **{f'action{i}':float(v) for i,v in enumerate(action)}))
        print('Common-state decisions:',method,flush=True)
    # Observe/refit implementation is identical for every planner budget; profile
    # it once on the same history instead of duplicating identical workloads.
    first = p['planners'][0]['method']
    for r in p['calibration_replicates']:
        for state in states:
            for repetition in range(reps+1):
                policy = prepared(root,first,state,r)
                before = np.asarray(state['observation'],np.float32); after = np.asarray(state['next_observation'],np.float32)
                start = time.process_time_ns(); policy.observe(before,after); elapsed = (time.process_time_ns()-start)/1e6
                if repetition: observe_samples.append(dict(state=state['state'],replicate=r,repetition=repetition,
                    update_due=int(state['step']%p['update_interval']==0),cpu_ms=elapsed))
    csv_write(root/'decision-samples.csv',samples); csv_write(root/'observation-samples.csv',observe_samples)
    names = ['common-states.json','decision-samples.csv','observation-samples.csv']
    write_json(root/'benchmark-hashes.json',{n:sha(root/n) for n in names})


def timed_training_one(root, identity):
    root = Path(root); p = read(root/'protocol.json'); torch.set_num_threads(1)
    original = selected_model(root,identity['replicate'],identity['seed'])
    fitted = next(f for f in read(root/'parent/fits.json') if f['replicate'] == identity['replicate'] and f['size'] == p['calibration_size'])
    start_cpu = time.process_time(); start_wall = time.perf_counter()
    replay = train_one(root/'training-replay',fitted,identity['seed'])
    cpu = time.process_time()-start_cpu; wall = time.perf_counter()-start_wall
    old = torch.load(root/'parent'/original['path'],map_location='cpu',weights_only=True)
    new = torch.load(root/'training-replay'/replay['path'],map_location='cpu',weights_only=True)
    assert all(torch.equal(value,new['policy'][key]) for key,value in old['policy'].items())
    curve = str(Path(original['path']).parent/'training.csv')
    compare_tables(csv_read(root/'parent'/curve),csv_read(root/'training-replay'/curve))
    return dict(**identity,steps=replay['steps'],updates=replay['updates'],cpu_seconds=cpu,wall_seconds=wall,
        identical_policy=1,identical_curve=1,path=replay['path'],sha256=replay['sha256'],original_path=original['path'],original_sha256=original['sha256'])


def training_cost(root, workers=1):
    root = Path(root); check(root); p = read(root/'protocol.json'); destination = root/'training-replay'
    destination.mkdir(exist_ok=False)
    for name in ('protocol.json','fits.json','source-hashes.json'): shutil.copyfile(root/'parent'/name,destination/name)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(timed_training_one,str(root),x) for x in p['training_replays']]
        rows = [f.result() for f in futures]
    csv_write(root/'training-cost.csv',rows)
    files = ['training-cost.csv']+[str(x.relative_to(root)) for x in destination.rglob('*') if x.is_file()]
    write_json(root/'training-cost-hashes.json',{n:sha(root/n) for n in files})


def tables(root):
    p = read(root/'protocol.json'); records = csv_read(root/'episodes.csv'); timings = csv_read(root/'decision-samples.csv'); observations = csv_read(root/'observation-samples.csv')
    training = csv_read(root/'training-cost.csv'); methods = ['pi','sac']+[x['method'] for x in p['planners']]
    cost = []; due = np.mean([float(r['cpu_ms']) for r in observations if int(r['update_due'])]); other = np.mean([float(r['cpu_ms']) for r in observations if not int(r['update_due'])])
    amortized = (due+(p['update_interval']-1)*other)/p['update_interval']
    offline = float(np.median([float(r['cpu_seconds']) for r in training]))
    for method in methods:
        values = np.array([float(r['cpu_ms']) for r in timings if r['method'] == method]); refit = 0. if method in ('pi','sac') else float(amortized)
        cost.append(dict(method=method,samples=len(values),mean_decision_cpu_ms=float(values.mean()),median_decision_cpu_ms=float(np.median(values)),
            p90_decision_cpu_ms=float(np.quantile(values,.9)),amortized_observe_cpu_ms=refit,mean_total_cpu_ms=float(values.mean()+refit)))
    lookup = {(r['case'],r['method'],int(r['replicate']),int(r['model_seed']),int(r['market_seed']),r['window']):r for r in records}
    rs = p['calibration_replicates']; ss = p['training_seeds']; ms = p['evaluation_seeds']; n = p['bootstrap_resamples']; rng = np.random.default_rng(p['bootstrap_seed'])
    ri = rng.integers(len(rs),size=(n,len(rs))); si = rng.integers(len(ss),size=(n,len(ss))); mi = rng.integers(len(ms),size=(n,len(ms)))
    contrasts = []; frontier = []; break_even = []
    def values(case,method,metric,window):
        return np.array([[[float(lookup[case,method,-1 if method == 'pi' else r,s if method == 'sac' else -1,m,window][metric]) for m in ms] for s in ss] for r in rs])
    for case in p['cases']:
        for base in [x for x in methods if x != 'sac']:
            for window in ('pre_update','adaptation','later','full'):
                for metric in ('mean_reward','mean_fulfillment','mean_money_error'):
                    delta = values(case,'sac',metric,window)-values(case,base,metric,window)
                    draws = delta[ri[:,:,None,None],si[:,None,:,None],mi[:,None,None,:]].mean((1,2,3)); lo,hi = np.quantile(draws,[.025,.975])
                    contrasts.append(dict(case=case,baseline=base,window=window,metric=metric,difference=float(delta.mean()),lower=float(lo),upper=float(hi)))
        scores = {method:float(values(case,method,'mean_reward','later').mean()) for method in methods}
        costs = {r['method']:r['mean_total_cpu_ms'] for r in cost}
        for method in methods:
            dominated = any(costs[m] <= costs[method] and scores[m] >= scores[method] and (costs[m] < costs[method] or scores[m] > scores[method]) for m in methods if m != method)
            frontier.append(dict(case=case,method=method,cpu_ms=costs[method],mean_reward=scores[method],
                mean_fulfillment=float(values(case,method,'mean_fulfillment','later').mean()),mean_money_error=float(values(case,method,'mean_money_error','later').mean()),point_estimate_nondominated=int(not dominated)))
    sac_cpu = next(x['mean_total_cpu_ms'] for x in cost if x['method'] == 'sac')
    for row in cost:
        if row['method'] == 'sac': continue
        saving = (row['mean_total_cpu_ms']-sac_cpu)/1000
        count = offline/saving if saving > 0 else None
        break_even.append(dict(baseline=row['method'],median_training_cpu_seconds=offline,cpu_saving_seconds_per_decision=saving,
            decisions_to_cpu_break_even=count,days_if_one_decision_per_minute=count/1440 if count is not None else None,
            days_if_one_decision_per_hour=count/24 if count is not None else None))
    return cost,contrasts,frontier,break_even


def analyze(root):
    root = Path(root); outputs = tables(root)
    for name,rows in zip(('common-costs.csv','contrasts.csv','frontier.csv','break-even.csv'),outputs): csv_write(root/name,rows)
    write_json(root/'analysis-hashes.json',{n:sha(root/n) for n in ('common-costs.csv','contrasts.csv','frontier.csv','break-even.csv')})


def verify(root, workers=1):
    root = Path(root); check(root); p = resolve(read(root/'protocol.json')); torch.set_num_threads(1)
    for manifest in ('data-hashes.json','benchmark-hashes.json','training-cost-hashes.json','analysis-hashes.json'):
        for name,digest in read(root/manifest).items(): assert sha(root/name) == digest
    work = jobs(root); keys = {tuple(str(j[k]) for k in IDENTITY) for j in work}; records = csv_read(root/'episodes.csv')
    lookup = {tuple(r[k] for k in IDENTITY)+(r['window'],):r for r in records}; assert len(records) == len(lookup) == 5*len(work)
    changes = csv_read(root/'updates.csv'); retained = {}; seen = set()
    chosen = [j for j in work if j['market_seed'] == p['evaluation_seeds'][0] and j['replicate'] in (-1,p['calibration_replicates'][0]) and j['model_seed'] in (-1,p['training_seeds'][0])]
    chosen_keys = {tuple(str(j[k]) for k in IDENTITY) for j in chosen}
    with gzip.open(root/'steps.csv.gz','rt') as stream:
        for key,group in groupby(csv.DictReader(stream),key=lambda r:tuple(r[k] for k in IDENTITY)):
            assert key in keys and key not in seen; seen.add(key)
            trace = [{k:float(v) for k,v in r.items() if k not in IDENTITY} for r in group]
            assert [int(r['step']) for r in trace] == list(range(1,p['evaluation_steps']+1))
            for r in trace:
                assert abs(r['cash_residual']) < 1e-7 and r['goods_residual'] < 1e-7
                assert abs(r['reward']+sum(r[k] for k in ('money_cost','service_cost','inventory_cost','price_cost','waste_cost'))) < 1e-8
            for window,summary in windows(p,trace):
                for name,value in summary.items(): np.testing.assert_allclose(value,float(lookup[key+(window,)][name]),atol=1e-9)
            updates = [r for r in changes if tuple(r[k] for k in IDENTITY) == key]
            if key[1] not in ('pi','sac'):
                assert [int(r['after_step']) for r in updates] == [t for t in range(1,p['evaluation_steps']+1) if t >= p['minimum_observations'] and t%p['update_interval'] == 0]
                assert all(int(r['last_used_step']) == int(r['after_step']) and int(r['n']) <= p['rolling_window'] for r in updates)
            else: assert not updates
            if key in chosen_keys: retained[key] = trace,updates
    assert seen == keys
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one,str(root),j) for j in chosen]
        for identity,future in zip(chosen,futures):
            _,new,updates,_ = future.result(); old,saved = retained[tuple(str(identity[k]) for k in IDENTITY)]
            for a,b in zip(old,new):
                for k in a: np.testing.assert_allclose(a[k],b[k],atol=1e-8,rtol=1e-8)
            compare_tables(saved,updates)
    compare_tables(csv_read(root/'summary.csv'),aggregate(records,['case','method','window']))
    states = common_states(root); assert states == read(root/'common-states.json')
    samples = csv_read(root/'decision-samples.csv'); observe = csv_read(root/'observation-samples.csv')
    groups_per_state = len(p['calibration_replicates'])*(len(p['planners'])+len(p['training_seeds']))+1
    assert len(samples) == len(states)*groups_per_state*p['timing_repetitions']
    assert len(observe) == len(states)*len(p['calibration_replicates'])*p['timing_repetitions']
    for row in samples:
        if row['repetition'] != '1' or row['state'] != '0': continue
        state = states[0]; method = row['method']; r = int(row['replicate']); s = int(row['model_seed'])
        policy = controller(root,method,r,s) if method in ('sac','pi') else prepared(root,method,state,r)
        obs = np.asarray(state['observation'],np.float32); action = policy.act(obs,deterministic=True) if method == 'sac' else policy.act(obs)
        np.testing.assert_allclose(action,[float(row[f'action{i}']) for i in range(4)],atol=1e-8)
    training = csv_read(root/'training-cost.csv')
    assert len(training) == len(p['training_replays'])
    for row in training:
        original = torch.load(root/'parent'/row['original_path'],map_location='cpu',weights_only=True)
        new = torch.load(root/'training-replay'/row['path'],map_location='cpu',weights_only=True)
        assert all(torch.equal(v,new['policy'][k]) for k,v in original['policy'].items())
        assert new['steps'] == int(row['steps']) == original['steps'] and new['updates'] == int(row['updates']) == original['updates']
        curve = str(Path(row['path']).parent/'training.csv'); compare_tables(csv_read(root/'parent'/curve),csv_read(root/'training-replay'/curve))
    for name,rows in zip(('common-costs.csv','contrasts.csv','frontier.csv','break-even.csv'),tables(root)):
        # Optional break-even values are blank in CSV and None in memory.
        if name == 'break-even.csv':
            saved = csv_read(root/name)
            assert len(saved) == len(rows)
            for a,b in zip(saved,rows):
                for k,v in b.items():
                    if v is None: assert a[k] == ''
                    elif isinstance(v,str): assert a[k] == v
                    else: np.testing.assert_allclose(float(a[k]),v)
        else: compare_tables(csv_read(root/name),rows)
    result = dict(evaluation_episodes=len(work),physical_steps=len(work)*p['evaluation_steps'],window_summaries=len(records),complete_replays=len(chosen),
        refits=len(changes),common_states=len(states),timing_samples=len(samples),observation_timing_samples=len(observe),
        timed_training_replays=len(training),timed_replay_interactions=sum(int(r['steps']) for r in training),
        original_training_timing_available=False,performance_policies_changed=False,source_data_models_statistics='passed')
    write_json(root/'verification.json',result); print(result,flush=True); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',choices=['init','run','benchmark','training-cost','analyze','verify'],required=True)
    parser.add_argument('--out',required=True,type=Path); parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--config',default='experiments/configs/compute-tradeoff.json'); parser.add_argument('--parent',default='results/diagnostics/2026-11-09-model-error-study')
    args = parser.parse_args()
    if args.phase == 'init': initialize(args.out,args.config,args.parent)
    elif args.phase in ('run','training-cost','verify'): globals()[args.phase.replace('-','_')](args.out,args.workers)
    else: globals()[args.phase](args.out)


if __name__ == '__main__': main()
