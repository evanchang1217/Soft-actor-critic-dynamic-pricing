"""Shared-data identification, fitted-simulator SAC and frozen-model planning.

No changes to old experiments. All identification uses observable sales/supply,
prices, cash, inventories and current/next recipe, never latent preference,
productivity, rejected demand or the true parameters. Known functional family
and all parameters except three are explicitly shared prior knowledge.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
import csv
import gzip
import shutil

import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from rl.config import SACConfig
from rl.sac import SACAgent, ReplayBuffer
from .market.economy import TaskConfig
from .common import read, sha, csv_read, csv_write, aggregate, compare_tables
from .market.continuing import StationaryMarket, Regime
from .common import MARKET_SOURCES
from .market_planner import MarketPlanner
from .market.controllers import JointController
from .common import summarize

SOURCES = MARKET_SOURCES + ('experiments/market_planner.py','experiments/model_error_study.py')
NOMINAL = dict(base_slope=5., reservation=7.5, switch_probability=.06)
PARAMETERS = tuple(NOMINAL)


def market(parameters, steps=96):
    """Construct from supplied knowledge only. Storage/capacity are known."""
    return StationaryMarket(TaskConfig(max_steps=steps,
        base_slope=parameters['base_slope'], reservation=parameters['reservation']),
        Regime(switch_probability=parameters['switch_probability']), 'stationary')


def resolve(p):
    p = dict(p)
    for key in ('calibration_replicates', 'calibration_sizes', 'training_seeds', 'evaluation_seeds'):
        values = p[key]
        if not values or len(set(values)) != len(values) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError('Invalid independent groups: ' + key)
    if sorted(p['calibration_sizes']) != p['calibration_sizes'] or min(p['calibration_sizes']) < 4:
        raise ValueError('Invalid nested calibration sizes')
    if not 0 < p['early_window'] < p['evaluation_steps']:
        raise ValueError('Invalid evaluation window')
    for spec in p['cases'].values():
        if set(spec) != set(PARAMETERS): raise ValueError('Unknown parameter')
        market(spec, p['evaluation_steps'])
    if p['cases']['nominal'] != NOMINAL: raise ValueError('Retain nominal reference')
    SACConfig(episodes=p['training_episodes'], **p['sac'])
    return p


def initialize(root, config):
    p = resolve(read(config)); root = new_run_directory(root)
    write_json(root/'protocol.json', p)
    write_json(root/'manifest.json', manifest('shared-data-model-error-study'))
    repo = Path(__file__).resolve().parents[1]; hashes = {}
    for name in SOURCES:
        target = root/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, target); hashes[name] = sha(target)
    write_json(root/'source-hashes.json', hashes)
    (root/'calibration').mkdir()
    return root


def check_sources(root, *, require_checkout=True):
    repo = Path(__file__).resolve().parents[1]
    for name, digest in read(root/'source-hashes.json').items():
        if (require_checkout and sha(repo/name) != digest) or sha(root/'source'/name) != digest:
            raise ValueError('Source drift: ' + name)


def collect(replicate, count, parameters=None):
    """Bounded random target-price logging; 96-period independent episodes.

    Only this function uses the actual calibration environment. Replicate
    prefixes are nested by construction, irrespective of requested count.
    """
    parameters = parameters or NOMINAL
    env = market(parameters); rng = np.random.default_rng(220000000 + replicate)
    rows = []; episode = 0
    while len(rows) < count:
        obs, _ = env.reset(seed=221000000 + replicate*10000 + episode)
        for t in range(96):
            state = env.unpack(env.frame())
            if t % 8 == 0:
                target = np.r_[rng.uniform(6.5, 9.5, 2), rng.uniform(9., 13.5, 2)]
            move = np.clip(target/state['prices']-1, -.05, .05).astype(np.float32)
            obs, _, _, _, info = env.step(move); after = env.env
            record = dict(index=len(rows), episode=episode, step=t+1,
                money=float(state['money']), treasury=float(state['treasury']),
                tasks=float(state['tasks'][0]), recipe=float(state['recipes'][0, 0]),
                next_recipe=float(after.recipes[0, 0]))
            for name, values in (('price', after.prices), ('stock', state['stock']),
                                 ('house', state['household']), ('supply', after.last_supply),
                                 ('sales', after.last_sales)):
                record.update({name+str(i): float(x) for i, x in enumerate(values)})
            rows.append(record)
            if len(rows) == count: break
        episode += 1
    return rows


def bounded_minimum(loss, low, high):
    # Coarse global search then deterministic local refinement, no scipy dependency.
    grid = np.linspace(low, high, 161); values = [loss(x) for x in grid]
    i = int(np.argmin(values)); a = grid[max(0, i-1)]; b = grid[min(len(grid)-1, i+1)]
    for _ in range(45):
        left = a+(b-a)/3; right = b-(b-a)/3
        if loss(left) < loss(right): b = right
        else: a = left
    return float((a+b)/2)


def fit(rows):
    """Censored forward least squares on realized public supply/sales.

    Keep actual observed supply in demand fitting; this separates demand
    identification from supply-model residuals. Noise is zero at the fitted
    conditional median, hence this is an approximate estimator, not an exact
    likelihood. Clipping and pooled-cash constraints are included.
    """
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    bids = np.stack([a['price0'], a['price1']], -1)
    asks = np.stack([a['price2'], a['price3']], -1)
    stock = np.stack([a['stock0'], a['stock1']], -1)
    house = np.stack([a['house0'], a['house1']], -1)
    supply = np.stack([a['supply0'], a['supply1']], -1)
    sales = np.stack([a['sales0'], a['sales1']], -1)
    recipe = np.stack([a['recipe'], 3-a['recipe']], -1)
    eligible = np.maximum(0, a['tasks'][:, None]*recipe-house)
    sigmoid = lambda z: 1/(1+np.exp(np.clip(-z, -50, 50)))
    z = bids/1.5; allocation = np.exp(z-z.max(-1, keepdims=True)); allocation /= allocation.sum(-1, keepdims=True)
    def supply_loss(reservation):
        prediction = 36*sigmoid(bids.mean(-1)-reservation)[:, None]*allocation
        prediction = np.minimum(prediction, np.maximum(0, 72-stock))
        cost = (prediction*bids).sum(-1)
        prediction *= np.minimum(1, a['treasury']/np.maximum(cost, 1e-12))[:, None]
        return float(np.mean((prediction-supply)**2))
    def demand_loss(slope):
        prediction = np.minimum(eligible*sigmoid(np.log(3)-slope*np.log(asks/10)), stock+supply)
        cost = (prediction*asks).sum(-1); cash = a['money']+(supply*bids).sum(-1)
        prediction *= np.minimum(1, cash/np.maximum(cost, 1e-12))[:, None]
        return float(np.mean((prediction-sales)**2))
    reservation = bounded_minimum(supply_loss, 5., 10.)
    slope = bounded_minimum(demand_loss, 1., 15.)
    # Recipes are observed after every transition, including final transitions.
    switches = int(np.count_nonzero(a['recipe'] != a['next_recipe']))
    probability = float(np.clip(switches/len(rows), 0., .5))
    return dict(parameters=dict(base_slope=slope, reservation=reservation,
                               switch_probability=probability),
                n=len(rows), switches=switches, supply_mse=supply_loss(reservation),
                sales_mse=demand_loss(slope))


def calibrate(root):
    root = Path(root); check_sources(root); p = read(root/'protocol.json'); fits = []
    if (root/'fits.json').exists(): raise FileExistsError('Calibration already frozen')
    for replicate in p['calibration_replicates']:
        rows = collect(replicate, max(p['calibration_sizes']))
        name = f'calibration/replicate-{replicate}.csv'; csv_write(root/name, rows)
        for n in p['calibration_sizes']:
            fits.append(dict(replicate=replicate, size=n, log=name, log_sha256=sha(root/name),
                             prefix_sha256=__import__('hashlib').sha256(__import__('json').dumps(rows[:n], sort_keys=True).encode()).hexdigest(),
                             **fit(rows[:n])))
    write_json(root/'fits.json', fits)
    print('Frozen fits:', fits, flush=True)


def train_one(root, fitted, seed):
    root = Path(root); p = read(root/'protocol.json'); torch.set_num_threads(1); torch.manual_seed(seed)
    r, n = fitted['replicate'], fitted['size']; out = root/'models'/f'n{n}-rep{r}-seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    env = market(fitted['parameters'], p['training_steps_per_episode'])
    config = SACConfig(episodes=p['training_episodes'], **p['sac']); agent = SACAgent(env, config)
    with torch.no_grad():
        agent.policy.net[-1].weight[4:].zero_(); agent.policy.net[-1].bias[4:].fill_(np.log(.2))
    replay = ReplayBuffer(config.replay_size, env.observation_space.shape[0], 4, seed+1)
    rng = np.random.default_rng(seed+2); steps = updates = 0; logs = []
    for episode in range(p['training_episodes']):
        obs, _ = env.reset(seed=223000000+seed*10000+episode); rewards = []
        for _ in range(env.config.max_steps):
            move = rng.uniform(-.05, .05, 4).astype(np.float32) if steps < config.warmup_steps else agent.act(obs)
            nxt, reward, term, trunc, info = env.step(move)
            replay.push(env.encode(obs), agent.normalize_action(move), reward, env.encode(nxt), term); steps += 1
            if steps >= config.warmup_steps and len(replay) >= config.batch_size:
                metrics = agent.update(replay); updates += 1
            obs = nxt; rewards.append(reward)
        logs.append(dict(episode=episode+1, steps=steps, updates=updates, reward=float(np.mean(rewards))))
        if (episode+1) % 64 == 0: print(f'Train n{n} rep{r} seed{seed}: {steps}', flush=True)
    csv_write(out/'training.csv', logs)
    saved = dict(format='shared-model-error-v1', seed=seed, replicate=r, size=n,
        parameters=fitted['parameters'], steps=steps, updates=updates, sac=asdict(config),
        protocol_sha256=sha(root/'protocol.json'), fits_sha256=sha(root/'fits.json'),
        source_manifest_sha256=sha(root/'source-hashes.json'), policy=agent.policy.state_dict())
    path = out/'policy.pt'; torch.save(saved, path)
    result = {k: saved[k] for k in ('seed', 'replicate', 'size', 'parameters', 'steps', 'updates')}
    result.update(path=str(path.relative_to(root)), sha256=sha(path)); write_json(out/'outcome.json', result)
    return result


def train(root, workers=1):
    root = Path(root); check_sources(root); p = read(root/'protocol.json')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(train_one, str(root), f, s) for f in read(root/'fits.json') for s in p['training_seeds']]
        models = [f.result() for f in futures]
    write_json(root/'models.json', models)


def load_agent(root, model):
    if sha(root/model['path']) != model['sha256']: raise ValueError('Model hash mismatch')
    saved = torch.load(root/model['path'], map_location='cpu', weights_only=True)
    assert saved['format'] == 'shared-model-error-v1'
    for key, name in (('protocol_sha256', 'protocol.json'), ('fits_sha256', 'fits.json'), ('source_manifest_sha256', 'source-hashes.json')):
        assert saved[key] == sha(root/name)
    for key in ('seed', 'replicate', 'size', 'parameters', 'steps', 'updates'): assert saved[key] == model[key]
    agent = SACAgent(market(model['parameters']), SACConfig(**saved['sac']))
    agent.policy.load_state_dict(saved['policy']); return agent


def jobs(root):
    p = read(root/'protocol.json'); models = read(root/'models.json'); output = []
    for case in p['cases']:
        for seed in p['evaluation_seeds']:
            for method in ('oracle_saa8', 'pi'):
                output.append(dict(case=case, market_seed=seed, method=method, size=0, replicate=-1, model_seed=-1))
            for fitted in read(root/'fits.json'):
                base = dict(case=case, market_seed=seed, size=fitted['size'], replicate=fitted['replicate'])
                output.append(dict(**base, method='fitted_saa8', model_seed=-1))
                if fitted['size'] == max(p['calibration_sizes']) and case in ('nominal', 'combined'):
                    output.append(dict(**base, method='fitted_saa64', model_seed=-1))
                for s in p['training_seeds']:
                    output.append(dict(**base, method='fitted_sac', model_seed=s))
            if case in ('nominal', 'combined'):
                output.append(dict(case=case, market_seed=seed, method='oracle_saa64', size=0, replicate=-1, model_seed=-1))
    return output


def evaluate_one(root, identity):
    root = Path(root); p = read(root/'protocol.json'); torch.set_num_threads(1)
    actual = market(p['cases'][identity['case']], p['evaluation_steps'])
    obs, _ = actual.reset(seed=identity['market_seed']); method = identity['method']
    if method.startswith('fitted'):
        fitted = next(f for f in read(root/'fits.json') if f['size'] == identity['size'] and f['replicate'] == identity['replicate'])
        knowledge = fitted['parameters']
    else: knowledge = p['cases'][identity['case']]
    known_env = market(knowledge, p['evaluation_steps'])
    if method == 'fitted_sac':
        model = next(m for m in read(root/'models.json') if (m['size'], m['replicate'], m['seed']) == (identity['size'], identity['replicate'], identity['model_seed']))
        policy = load_agent(root, model)
    elif method == 'pi': policy = JointController(known_env, 'pi', **p['pi'])
    else:
        policy = MarketPlanner(known_env, **p['planner_supplement' if method.endswith('64') else 'planner'])
    trace = []
    for t in range(p['evaluation_steps']):
        move = policy.act(obs, deterministic=True) if method == 'fitted_sac' else policy.act(obs)
        before = actual.unpack(actual.frame()); obs, _, _, _, info = actual.step(move)
        trace.append(dict(**identity, step=t+1, **info,
                          pre_money=float(before['money']), pre_recipe=float(before['recipes'][0, 0]),
                          **{f'action{i}': float(v) for i, v in enumerate(move)}))
    records = [dict(**identity, window=name, **summarize(rows)) for name, rows in
               (('first', trace[:p['early_window']]), ('later', trace[p['early_window']:]), ('full', trace))]
    return records, trace


def evaluate(root, workers=1):
    root = Path(root); check_sources(root); work = jobs(root)
    if (root/'evaluation-start.json').exists(): raise FileExistsError('Evaluation already started')
    write_json(root/'evaluation-start.json', {name: sha(root/name) for name in ('protocol.json', 'fits.json', 'models.json')})
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool, gzip.open(root/'steps.csv.gz', 'wt', newline='') as stream:
        futures = [pool.submit(evaluate_one, str(root), j) for j in work]; writer = None
        for i, future in enumerate(futures):
            rows, trace = future.result(); records.extend(rows)
            if writer is None: writer = csv.DictWriter(stream, fieldnames=list(trace[0])); writer.writeheader()
            writer.writerows(trace)
            if (i+1) % 40 == 0: print(f'Evaluated {i+1}/{len(work)}', flush=True)
    csv_write(root/'episodes.csv', records)
    csv_write(root/'summary.csv', aggregate(records, ['case', 'method', 'size', 'window']))
    paths = ['protocol.json', 'source-hashes.json', 'fits.json', 'models.json', 'evaluation-start.json', 'episodes.csv', 'summary.csv', 'steps.csv.gz']
    paths += [str(f.relative_to(root)) for f in sorted((root/'calibration').glob('*.csv'))]
    paths += [str(f.relative_to(root)) for f in sorted((root/'models').rglob('training.csv'))]
    write_json(root/'data-hashes.json', {name: sha(root/name) for name in paths})


def statistics(root):
    p = read(root/'protocol.json'); rows = csv_read(root/'episodes.csv')
    lookup = {(r['case'], r['method'], int(r['size']), int(r['replicate']), int(r['model_seed']), int(r['market_seed']), r['window']): r for r in rows}
    reps = p['calibration_replicates']; seeds = p['training_seeds']; markets = p['evaluation_seeds']
    rng = np.random.default_rng(p['bootstrap_seed']); n = p['bootstrap_resamples']
    ri = rng.integers(len(reps), size=(n, len(reps))); si = rng.integers(len(seeds), size=(n, len(seeds))); mi = rng.integers(len(markets), size=(n, len(markets)))
    def values(case, method, size, metric):
        return np.array([[[float(lookup[(case, method, size if method.startswith('fitted') else 0,
                    r if method.startswith('fitted') else -1, s if method == 'fitted_sac' else -1, m, 'later')][metric])
                   for m in markets] for s in seeds] for r in reps])
    def interval(delta):
        draws = delta[ri[:, :, None, None], si[:, None, :, None], mi[:, None, None, :]].mean((1, 2, 3))
        lo, hi = np.quantile(draws, [.025, .975])
        return dict(difference=float(delta.mean()), lower=float(lo), upper=float(hi))
    contrasts = []; degradation = []
    for size in p['calibration_sizes']:
        for case in p['cases']:
            for base in ['fitted_saa8', 'oracle_saa8', 'pi'] + (['fitted_saa64', 'oracle_saa64'] if size == max(p['calibration_sizes']) and case in ('nominal', 'combined') else []):
                for metric in ('mean_reward', 'mean_money_error', 'mean_fulfillment'):
                    contrasts.append(dict(size=size, case=case, method='fitted_sac', baseline=base, metric=metric,
                        **interval(values(case, 'fitted_sac', size, metric)-values(case, base, size, metric))))
            if case != 'nominal':
                sac_change = values(case, 'fitted_sac', size, 'mean_reward')-values('nominal', 'fitted_sac', size, 'mean_reward')
                plan_change = values(case, 'fitted_saa8', size, 'mean_reward')-values('nominal', 'fitted_saa8', size, 'mean_reward')
                degradation.append(dict(size=size, case=case, sac_change=float(sac_change.mean()), planner_change=float(plan_change.mean()),
                                        **interval(sac_change-plan_change)))
    return contrasts, degradation


def report(root):
    root = Path(root); contrasts, degradation = statistics(root)
    csv_write(root/'contrasts.csv', contrasts)
    if degradation: csv_write(root/'degradation.csv', degradation)
    else: (root/'degradation.csv').write_text('size,case,sac_change,planner_change,difference,lower,upper\n')
    p = read(root/'protocol.json')
    primary = next(r for r in contrasts if r['case'] == 'nominal' and r['size'] == max(p['calibration_sizes']) and r['baseline'] == 'fitted_saa8' and r['metric'] == 'mean_reward')
    write_json(root/'primary.json', primary)
    write_json(root/'report-hashes.json', {name: sha(root/name) for name in ('contrasts.csv', 'degradation.csv', 'primary.json')})
    print('Primary exploratory contrast:', primary, flush=True)


IDENTITY = ('case', 'market_seed', 'method', 'size', 'replicate', 'model_seed')


def verify(root):
    root = Path(root); check_sources(root); p = resolve(read(root/'protocol.json')); torch.set_num_threads(1)
    for name, digest in read(root/'data-hashes.json').items(): assert sha(root/name) == digest
    for name, digest in read(root/'report-hashes.json').items(): assert sha(root/name) == digest
    for fitted in read(root/'fits.json'):
        rows = csv_read(root/fitted['log']); fresh = collect(fitted['replicate'], max(p['calibration_sizes']))
        compare_tables(rows, fresh)
        regenerated = fit(rows[:fitted['size']])
        for key in PARAMETERS: np.testing.assert_allclose(regenerated['parameters'][key], fitted['parameters'][key], atol=1e-10)
    models = read(root/'models.json')
    assert len(models) == len(p['calibration_replicates'])*len(p['calibration_sizes'])*len(p['training_seeds'])
    for model in models:
        load_agent(root, model)
        assert model['steps'] == p['training_episodes']*p['training_steps_per_episode']
        assert model['updates'] == model['steps']-max(p['sac']['warmup_steps'], p['sac']['batch_size'])+1
    work = jobs(root); keys = {tuple(str(j[k]) for k in IDENTITY) for j in work}; seen = set()
    records = csv_read(root/'episodes.csv'); assert len(records) == 3*len(work)
    lookup = {tuple(str(r[k]) for k in IDENTITY)+(r['window'],): r for r in records}
    replayed = 0
    with gzip.open(root/'steps.csv.gz', 'rt') as stream:
        for key, group in groupby(csv.DictReader(stream), key=lambda r: tuple(r[k] for k in IDENTITY)):
            assert key in keys and key not in seen; seen.add(key)
            trace = [{k: float(v) for k, v in r.items() if k not in IDENTITY} for r in group]
            assert [int(r['step']) for r in trace] == list(range(1, p['evaluation_steps']+1))
            for row in trace:
                assert abs(row['cash_residual']) < 1e-7 and row['goods_residual'] < 1e-7
                assert abs(row['reward']+sum(row[k] for k in ('money_cost', 'service_cost', 'inventory_cost', 'price_cost', 'waste_cost'))) < 1e-8
            for name, subset in (('first', trace[:p['early_window']]), ('later', trace[p['early_window']:]), ('full', trace)):
                for metric, v in summarize(subset).items(): np.testing.assert_allclose(v, float(lookup[key+(name,)][metric]), atol=1e-9)
            identity = next(j for j in work if tuple(str(j[k]) for k in IDENTITY) == key)
            eligible = identity['market_seed'] == p['evaluation_seeds'][0] and (identity['replicate'] in (-1, p['calibration_replicates'][0])) and identity['model_seed'] in (-1, p['training_seeds'][0])
            if eligible:
                _, regenerated = evaluate_one(root, identity)
                for old, new in zip(trace, regenerated):
                    for name in old: np.testing.assert_allclose(old[name], new[name], atol=1e-8)
                replayed += 1
    assert seen == keys and len(lookup) == len(records)
    compare_tables(csv_read(root/'summary.csv'), aggregate(records, ['case', 'method', 'size', 'window']))
    contrasts, degradation = statistics(root)
    compare_tables(csv_read(root/'contrasts.csv'), contrasts); compare_tables(csv_read(root/'degradation.csv'), degradation)
    result = dict(calibrations=len(read(root/'fits.json')), checkpoints=len(models), evaluation_episodes=len(work),
                  window_summaries=len(records), physical_evaluation_steps=len(work)*p['evaluation_steps'], complete_replays=replayed,
                  source_data_models_calibration_statistics='passed')
    write_json(root/'verification.json', result); print(result, flush=True); return result


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--phase', choices=['init', 'calibrate', 'train', 'evaluate', 'report', 'verify'], required=True)
    a.add_argument('--out', required=True, type=Path); a.add_argument('--workers', type=int, default=1)
    a.add_argument('--config', default='experiments/configs/model-error-study.json')
    args = a.parse_args()
    if args.phase == 'init': initialize(args.out, args.config)
    elif args.phase in ('train', 'evaluate'): globals()[args.phase](args.out, args.workers)
    else: globals()[args.phase](args.out)


if __name__ == '__main__': main()
