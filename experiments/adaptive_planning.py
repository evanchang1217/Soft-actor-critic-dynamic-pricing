"""Prospective online identification versus frozen planning on fresh markets."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path
import csv
import gzip
import shutil
import time

import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from .common import read, sha, csv_read, csv_write, aggregate, compare_tables
from .model_error_study import SOURCES as PRIOR_SOURCES, market, fit, load_agent
from .market_planner import MarketPlanner
from .market.controllers import JointController
from .common import summarize, check_parent_manifest, require, evaluation_run

SOURCES = PRIOR_SOURCES + ('experiments/adaptive_planning.py',)
IDENTITY = ('case', 'method', 'replicate', 'model_seed', 'market_seed')


def public_transition(model, before, after):
    """Build estimator input from two observed frames, with no simulator info."""
    a = model.unpack(np.asarray(before)[-model.frame_size:])
    b = model.unpack(np.asarray(after)[-model.frame_size:])
    row = dict(money=float(a['money']), treasury=float(a['treasury']),
               tasks=float(a['tasks'][0]), recipe=float(a['recipes'][0, 0]),
               next_recipe=float(b['recipes'][0, 0]))
    for name, values in [('price', b['prices']), ('stock', a['stock']), ('house', a['household']),
                         ('supply', b['last_supply']), ('sales', b['last_sales'])]:
        row.update({name+str(i):float(x) for i, x in enumerate(values)})
    return row


def identifying_counts(rows):
    """Public uncensored, cash-unconstrained price-response opportunities."""
    supply_points = demand_points = 0
    for r in rows:
        bid = np.array([r['price0'], r['price1']]); ask = np.array([r['price2'], r['price3']])
        stock = np.array([r['stock0'], r['stock1']]); house = np.array([r['house0'], r['house1']])
        supply = np.array([r['supply0'], r['supply1']]); sales = np.array([r['sales0'], r['sales1']])
        supply_cash = r['treasury']-float(supply@bid) > .001
        demand_cash = r['money']+float(supply@bid)-float(sales@ask) > .001
        supply_points += int(np.count_nonzero((supply > .001) & (stock+supply < 72-.001))) if supply_cash else 0
        eligible = np.maximum(0, r['tasks']*np.array([r['recipe'], 3-r['recipe']])-house)
        demand_points += int(np.count_nonzero((eligible > .001) & (sales > .001) &
            (sales < stock+supply-.001) & (np.abs(np.log(ask/10)) > .01))) if demand_cash else 0
    return supply_points, demand_points


class OnlinePlanner:
    def __init__(self, parameters, plan, specification=None):
        self.parameters = dict(parameters); self.plan = plan
        self.planner = MarketPlanner(market(self.parameters, plan['evaluation_steps']), **(specification or plan['planner']))
        self.history = deque(maxlen=plan['rolling_window']); self.steps = 0; self.updates = []
        self.refit_cpu_seconds = 0.

    def act(self, observation): return self.planner.act(observation)

    def observe(self, before, after):
        self.history.append(public_transition(self.planner.env, before, after)); self.steps += 1
        if self.steps < self.plan['minimum_observations'] or self.steps % self.plan['update_interval']: return
        start = time.process_time(); rows = list(self.history)
        result = fit(rows); supply_points, demand_points = identifying_counts(rows)
        new = dict(self.parameters); threshold = self.plan['minimum_identifying_points']
        if supply_points >= threshold: new['reservation'] = result['parameters']['reservation']
        if demand_points >= threshold: new['base_slope'] = result['parameters']['base_slope']
        new['switch_probability'] = result['parameters']['switch_probability']
        # Preserve integral state and private forecast RNG; change supplied model only.
        self.planner.env = market(new, self.plan['evaluation_steps']); self.parameters = new
        elapsed = time.process_time()-start; self.refit_cpu_seconds += elapsed
        self.updates.append(dict(after_step=self.steps, first_used_step=self.steps-len(rows)+1,
            last_used_step=self.steps, n=len(rows), supply_points=supply_points, demand_points=demand_points,
            reservation_accepted=int(supply_points >= threshold), slope_accepted=int(demand_points >= threshold),
            **new, supply_mse=result['supply_mse'], sales_mse=result['sales_mse']))


def resolve(p):
    p = dict(p)
    for key in ('calibration_replicates', 'training_seeds', 'evaluation_seeds'):
        v = p[key]
        if not v or len(v) != len(set(v)) or any(type(x) is not int or x < 0 for x in v): raise ValueError(key)
    if not 0 < p['minimum_observations'] < p['early_window'] < p['evaluation_steps']: raise ValueError('Window order')
    if not p['minimum_observations'] <= p['rolling_window'] or p['update_interval'] <= 0: raise ValueError('Invalid schedule')
    if p['minimum_identifying_points'] <= 0: raise ValueError('Invalid information gate')
    for spec in p['cases'].values(): market(spec, p['evaluation_steps'])
    if not set(p['supplement_cases']) <= set(p['cases']): raise ValueError('Invalid supplement')
    return p


def initialize(root, config, parent):
    p = resolve(read(config)); parent = Path(parent).resolve()
    if not (parent/'source-hashes.json').exists() or not (parent/'source').is_dir():
        raise FileNotFoundError('Restore full evidence with scripts/research_archive.py restore --out runs/evidence, then pass its model-error record with --parent')
    from .model_error_study import check_sources as check_parent
    check_parent(parent, require_checkout=False)
    root = new_run_directory(root); (root/'parent').mkdir()
    for name in ('protocol.json', 'fits.json', 'models.json', 'source-hashes.json'):
        shutil.copyfile(parent/name, root/'parent'/name)
    parent_files = {name:sha(root/'parent'/name) for name in ('protocol.json', 'fits.json', 'models.json', 'source-hashes.json')}
    for model in read(parent/'models.json'):
        require(sha(parent/model['path']) == model['sha256'], 'Saved evidence hash mismatch')
        target = root/'parent'/model['path']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent/model['path'], target); parent_files[model['path']] = sha(target)
    parent_plan = read(parent/'protocol.json')
    require(not set(p['evaluation_seeds']) & set(parent_plan['evaluation_seeds']+parent_plan.get('validation_seeds', [])),
            'Invalid parent models or overlapping evaluation seeds')
    fits = read(root/'parent/fits.json')
    for r in p['calibration_replicates']:
        require(any(f['size'] == p['calibration_size'] and f['replicate'] == r for f in fits),
                'Invalid parent models or overlapping evaluation seeds')
        for s in p['training_seeds']:
            require(any(m['size'] == p['calibration_size'] and m['replicate'] == r and m['seed'] == s for m in read(root/'parent/models.json')),
                    'Invalid parent models or overlapping evaluation seeds')
    write_json(root/'parent-hashes.json', parent_files)
    write_json(root/'protocol.json', p); write_json(root/'manifest.json', manifest('online-refitting-versus-frozen-price-planning'))
    repo = Path(__file__).resolve().parents[1]; hashes = {}
    for name in SOURCES:
        target = root/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, target); hashes[name] = sha(target)
    write_json(root/'source-hashes.json', hashes)
    return root


def check(root):
    repo = Path(__file__).resolve().parents[1]
    for name, digest in read(root/'source-hashes.json').items():
        require(sha(repo/name) == sha(root/'source'/name) == digest,
                'Saved evidence hash mismatch')
    for name, digest in read(root/'parent-hashes.json').items():
        require(sha(root/'parent'/name) == digest,
                'Saved evidence hash mismatch')
    check_parent_manifest(read(root/'parent/source-hashes.json'),read(root/'source-hashes.json'))


def jobs(root):
    p = read(root/'protocol.json'); output = []
    for case in p['cases']:
        for m in p['evaluation_seeds']:
            for method in ['oracle8', 'pi']+(['oracle64'] if case in p['supplement_cases'] else []):
                output.append(dict(case=case, method=method, replicate=-1, model_seed=-1, market_seed=m))
            for r in p['calibration_replicates']:
                for method in ['frozen8', 'online8']+(['frozen64', 'online64'] if case in p['supplement_cases'] else []):
                    output.append(dict(case=case, method=method, replicate=r, model_seed=-1, market_seed=m))
                for s in p['training_seeds']:
                    output.append(dict(case=case, method='sac', replicate=r, model_seed=s, market_seed=m))
    return output


def windows(p, trace):
    first = p['minimum_observations']; early = p['early_window']
    return [(name, summarize(rows)) for name, rows in [('pre_update', trace[:first]),
        ('adaptation', trace[first:early]), ('later', trace[early:]), ('first', trace[:early]), ('full', trace)]]


def evaluate_one(root, identity):
    root = Path(root); p = read(root/'protocol.json'); torch.set_num_threads(1)
    method = identity['method']; actual = market(p['cases'][identity['case']], p['evaluation_steps'])
    obs, _ = actual.reset(seed=identity['market_seed'])
    if identity['replicate'] >= 0:
        fitted = next(f for f in read(root/'parent/fits.json') if f['size'] == p['calibration_size'] and f['replicate'] == identity['replicate'])
        knowledge = fitted['parameters']
    else: knowledge = p['cases'][identity['case']]
    if method == 'sac':
        model = next(m for m in read(root/'parent/models.json') if m['size'] == p['calibration_size'] and m['replicate'] == identity['replicate'] and m['seed'] == identity['model_seed'])
        controller = load_agent(root/'parent', model)
    elif method == 'pi': controller = JointController(market(knowledge), 'pi', **read(root/'parent/protocol.json')['pi'])
    else:
        specification = p['planner_supplement' if method.endswith('64') else 'planner']
        controller = OnlinePlanner(knowledge, p, specification) if method.startswith('online') else MarketPlanner(market(knowledge, p['evaluation_steps']), **specification)
    trace = []; cpu = 0.
    for t in range(p['evaluation_steps']):
        parameters = controller.parameters if method.startswith('online') else knowledge
        start = time.process_time()
        move = controller.act(obs, deterministic=True) if method == 'sac' else controller.act(obs)
        cpu += time.process_time()-start
        nxt, _, _, _, info = actual.step(move)
        row = dict(**identity, step=t+1, **info, **{f'action{i}':float(v) for i, v in enumerate(move)},
                   **{f'estimate_{k}':float(v) for k, v in parameters.items()})
        if method.startswith('online'): controller.observe(obs, nxt)
        obs = nxt; trace.append(row)
    records = [dict(**identity, window=name, **summary) for name, summary in windows(p, trace)]
    updates = [dict(**identity, **u) for u in controller.updates] if method.startswith('online') else []
    timing = dict(**identity, decisions=p['evaluation_steps'], decision_cpu_seconds=cpu,
                  refit_cpu_seconds=controller.refit_cpu_seconds if method.startswith('online') else 0.)
    return records, trace, updates, timing


@evaluation_run(
    inputs=('protocol.json', 'parent-hashes.json', 'source-hashes.json'),
    outputs=('steps.csv.gz', 'episodes.csv', 'updates.csv', 'timings.csv',
             'summary.csv', 'per-fit.csv', 'data-hashes.json'))
def run(root, workers=1):
    root = Path(root); check(root); work = jobs(root)
    records = []; updates = []; times = []
    with ProcessPoolExecutor(max_workers=workers) as pool, gzip.open(root/'steps.csv.gz', 'wt', newline='') as stream:
        futures = [pool.submit(evaluate_one, str(root), j) for j in work]; writer = None
        for i, future in enumerate(futures):
            rows, trace, fitted, timing = future.result(); records.extend(rows); updates.extend(fitted); times.append(timing)
            if writer is None: writer = csv.DictWriter(stream, fieldnames=list(trace[0])); writer.writeheader()
            writer.writerows(trace)
            if (i+1) % 30 == 0: print(f'Evaluated {i+1}/{len(work)}', flush=True)
    csv_write(root/'episodes.csv', records); csv_write(root/'updates.csv', updates); csv_write(root/'timings.csv', times)
    csv_write(root/'summary.csv', aggregate(records, ['case', 'method', 'window']))
    csv_write(root/'per-fit.csv', aggregate(records, ['case', 'method', 'replicate', 'model_seed', 'window']))
    names = ['protocol.json', 'parent-hashes.json', 'source-hashes.json', 'evaluation-start.json', 'steps.csv.gz', 'episodes.csv', 'updates.csv', 'timings.csv', 'summary.csv', 'per-fit.csv']
    write_json(root/'data-hashes.json', {n:sha(root/n) for n in names})


def statistics(root):
    p = read(root/'protocol.json'); rows = csv_read(root/'episodes.csv')
    lookup = {(r['case'], r['method'], int(r['replicate']), int(r['model_seed']), int(r['market_seed']), r['window']):r for r in rows}
    reps = p['calibration_replicates']; seeds = p['training_seeds']; markets = p['evaluation_seeds']; n = p['bootstrap_resamples']
    rng = np.random.default_rng(p['bootstrap_seed']); ri = rng.integers(len(reps), size=(n, len(reps))); si = rng.integers(len(seeds), size=(n, len(seeds))); mi = rng.integers(len(markets), size=(n, len(markets)))
    def values(case, method, metric, window):
        return np.array([[[float(lookup[case, method, -1 if method.startswith('oracle') or method == 'pi' else r,
                    s if method == 'sac' else -1, m, window][metric]) for m in markets] for s in seeds] for r in reps])
    output = []
    for case in p['cases']:
        pairs = [('online8', 'frozen8'), ('online8', 'oracle8'), ('sac', 'frozen8'), ('sac', 'online8'), ('sac', 'oracle8'), ('sac', 'pi')]
        if case in p['supplement_cases']: pairs += [('online64', 'frozen64'), ('online64', 'oracle64'), ('sac', 'online64'), ('sac', 'frozen64')]
        for method, base in pairs:
            for window in ('pre_update', 'adaptation', 'later', 'full'):
                for metric in ('mean_reward', 'mean_fulfillment', 'mean_money_error'):
                    delta = values(case, method, metric, window)-values(case, base, metric, window)
                    draws = delta[ri[:, :, None, None], si[:, None, :, None], mi[:, None, None, :]].mean((1, 2, 3)); lo, hi = np.quantile(draws, [.025, .975])
                    output.append(dict(case=case, method=method, baseline=base, window=window, metric=metric,
                                       difference=float(delta.mean()), lower=float(lo), upper=float(hi)))
    return output


def report(root):
    root = Path(root); output = statistics(root); csv_write(root/'contrasts.csv', output)
    primary = next(r for r in output if r['case'] == 'combined' and r['method'] == 'online8' and r['baseline'] == 'frozen8' and r['metric'] == 'mean_reward' and r['window'] == 'later')
    write_json(root/'primary.json', primary)
    write_json(root/'report-hashes.json', {n:sha(root/n) for n in ('contrasts.csv', 'primary.json')})
    print('Primary exploratory comparison:', primary, flush=True)


def verify(root, workers=1):
    root = Path(root); check(root); p = resolve(read(root/'protocol.json')); torch.set_num_threads(1)
    for filename in ('data-hashes.json', 'report-hashes.json'):
        for name, digest in read(root/filename).items():
            require(sha(root/name) == digest,
                    'Saved evidence hash mismatch')
    work = jobs(root); keys = {tuple(str(j[k]) for k in IDENTITY) for j in work}
    records = csv_read(root/'episodes.csv'); lookup = {tuple(r[k] for k in IDENTITY)+(r['window'],):r for r in records}
    require(len(records) == len(lookup) == 5*len(work), 'Saved research evidence is inconsistent')
    for identity in work:
        if not identity['method'].startswith('online'): continue
        key = tuple(str(identity[k]) for k in IDENTITY)+('pre_update',)
        frozen = dict(identity, method=identity['method'].replace('online', 'frozen'))
        other = tuple(str(frozen[k]) for k in IDENTITY)+('pre_update',)
        for metric in ('mean_reward', 'mean_fulfillment', 'mean_money_error'):
            np.testing.assert_allclose(float(lookup[key][metric]), float(lookup[other][metric]), atol=1e-12)
    update_rows = csv_read(root/'updates.csv'); seen = set(); retained = {}; all_updates = 0
    chosen = [j for j in work if j['market_seed'] == p['evaluation_seeds'][0] and j['replicate'] in (-1, p['calibration_replicates'][0]) and j['model_seed'] in (-1, p['training_seeds'][0])]
    replay_keys = {tuple(str(j[k]) for k in IDENTITY) for j in chosen}
    with gzip.open(root/'steps.csv.gz', 'rt') as stream:
        for key, group in groupby(csv.DictReader(stream), key=lambda r:tuple(r[k] for k in IDENTITY)):
            require(key in keys and key not in seen, 'Saved research evidence is inconsistent'); seen.add(key)
            trace = [{k:float(v) for k, v in r.items() if k not in IDENTITY} for r in group]
            require([int(r['step']) for r in trace] == list(range(1, p['evaluation_steps']+1)),
                    'Saved research evidence is inconsistent')
            for r in trace:
                require(abs(r['cash_residual']) < 1e-7 and r['goods_residual'] < 1e-7,
                        'Saved research evidence is inconsistent')
                require(abs(r['reward']+sum(r[k] for k in ('money_cost', 'service_cost', 'inventory_cost', 'price_cost', 'waste_cost'))) < 1e-8,
                        'Saved research evidence is inconsistent')
            for window, summary in windows(p, trace):
                for metric, v in summary.items(): np.testing.assert_allclose(v, float(lookup[key+(window,)][metric]), atol=1e-9)
            fitted = [r for r in update_rows if tuple(r[k] for k in IDENTITY) == key]
            if key[1].startswith('online'):
                expected = [t for t in range(1, p['evaluation_steps']+1) if t >= p['minimum_observations'] and t % p['update_interval'] == 0]
                require([int(r['after_step']) for r in fitted] == expected, 'Saved research evidence is inconsistent')
                require(all(int(r['last_used_step']) == int(r['after_step']) and int(r['n']) <= p['rolling_window'] for r in fitted),
                        'Saved research evidence is inconsistent')
            else: require(not fitted, 'Saved research evidence is inconsistent')
            all_updates += len(fitted)
            if key in replay_keys: retained[key] = trace, fitted
    require(seen == keys, 'Saved research evidence is inconsistent')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, str(root), j) for j in chosen]
        for identity, future in zip(chosen, futures):
            _, replay, updates, _ = future.result(); key = tuple(str(identity[k]) for k in IDENTITY); old, fitted = retained[key]
            for a, b in zip(old, replay):
                for k in a: np.testing.assert_allclose(a[k], b[k], atol=1e-8, rtol=1e-8)
            compare_tables(fitted, updates)
    for name, groups in [('summary.csv', ['case', 'method', 'window']), ('per-fit.csv', ['case', 'method', 'replicate', 'model_seed', 'window'])]:
        compare_tables(csv_read(root/name), aggregate(records, groups))
    compare_tables(csv_read(root/'contrasts.csv'), statistics(root))
    result = dict(evaluation_episodes=len(work), physical_steps=len(work)*p['evaluation_steps'], window_summaries=len(records),
        complete_replays=len(chosen), online_refits=all_updates, source_models_data_statistics='passed', new_training_steps=0)
    write_json(root/'verification.json', result); print(result, flush=True); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase', choices=['init', 'run', 'report', 'verify'], required=True)
    parser.add_argument('--out', required=True, type=Path); parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--config', default='experiments/configs/adaptive-planning.json')
    parser.add_argument('--parent', default='results/diagnostics/2026-11-09-model-error-study')
    parser.add_argument('--retry', action='store_true', help='Retry a recorded failed evaluation, preserving partial outputs')
    args = parser.parse_args()
    if args.retry and args.phase != 'run': parser.error('--retry requires --phase run')
    if args.phase == 'init': initialize(args.out, args.config, args.parent)
    elif args.phase == 'report': report(args.out)
    elif args.phase == 'run': run(args.out, args.workers, retry=args.retry)
    else: verify(args.out, args.workers)


if __name__ == '__main__': main()
