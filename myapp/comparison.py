"""Read-only inference from the saved behavior-audit policies and frozen controls."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import torch
from django.conf import settings
from experiments.behavior_runtime import read_json,read_rows,sha,behavior_for,episode,make_policy
from experiments.behavior_market import BehaviorMarket
from rl.config import MarketConfig
from rl.sac import verify_checkpoint_updates, SACAgent
from .tasks import runner

RECORD = 'results/confirmation/2026-11-06-behavior-audit'
SCENARIOS = {
    'original': dict(label='Original market · 100 steps',target='original',suite='behavior',scenario='nominal',changes={}),
    'persistent': dict(label='Persistent original market · 1,000 steps',target='original',suite='stress',scenario='long_1000',changes={'max_steps':1000}),
    'inventory': dict(label='Consumption + satiation · 100 steps',target='inventory',suite='behavior',scenario='nominal',changes={}),
}


def options():
    c=read_json(Path(settings.BASE_DIR)/RECORD/'protocol.json')
    return dict(scenarios=[dict(value=k,label=v['label']) for k,v in SCENARIOS.items()],
                markets=c['evaluation_seeds'],models=c['training_seeds'])


def compare(scenario, market_seed, model_seed):
    allowed=options()
    if scenario not in SCENARIOS or market_seed not in allowed['markets'] or model_seed not in allowed['models']:
        raise ValueError('Choose a listed scenario, test market and model seed.')
    if not runner.compute_lock.acquire(blocking=False):
        raise RuntimeError('A training run or comparison is active. Please retry when it finishes.')
    try:
        return _compare(str(settings.BASE_DIR),scenario,market_seed,model_seed)
    finally:
        runner.compute_lock.release()


@lru_cache(maxsize=12)
def _compare(base,scenario,market_seed,model_seed):
    torch.set_num_threads(1)
    root=Path(base)/RECORD
    c=read_json(root/'protocol.json'); f=read_json(root/'frozen-selection.json')
    if sha(root/'protocol.json')!=f['protocol_sha256'] or sha(root/'validation.csv')!=f['validation_sha256']:
        raise ValueError('The saved study inputs failed verification.')
    # Inference must use the same implementation as the saved study.
    source=read_json(root/'research-source-hashes.json')
    for name in ('rl/config.py','rl/environment.py','rl/rewards.py','rl/sac.py',
                 'experiments/behavior_market.py','experiments/baselines.py'):
        if sha(Path(base)/name)!=source[name]:
            raise ValueError('The study implementation has changed; verify or update the demo adapter.')
    runtime=read_json(Path(base)/'myapp/comparison-runtime.json')
    if (runtime['original_source_sha256']!=source['experiments/behavior_audit.py'] or
            sha(Path(base)/'experiments/behavior_runtime.py')!=runtime['runtime_sha256']):
        raise ValueError('Saved-demo runtime provenance mismatch.')
    choice=SCENARIOS[scenario]; target=choice['target']
    r=next(r for r in read_json(root/'models.json') if r['family']==target and r['training_seed']==model_seed)
    path=root/r['path']
    if sha(path)!=r['sha256']: raise ValueError('Saved model failed verification.')
    agent,meta=SACAgent.load(path)
    verify_checkpoint_updates(meta,path)
    if meta['market']!=c['market'] or meta['trained_steps']!=c['training_steps']:
        raise ValueError('Saved model configuration mismatch.')
    if target!='original' and meta.get('behavior')!=behavior_for(c,target).to_dict():
        raise ValueError('Saved behavior metadata mismatch.')
    market=replace(MarketConfig.from_dict(c['market']),**choice['changes'])
    specs=[dict(kind='sac',method=f'sac_{target}',path=str(path)),
           {**f['choices'][target]['feedback'],'method':'matched_feedback'},
           {**f['choices'][target]['pi'],'method':'matched_pi'}]
    controllers=[]
    batch=read_rows(root/'report/summary.csv')
    for label,spec in zip(('SAC','P','PI'),specs):
        env=BehaviorMarket(market,behavior_for(c,target))
        policy=make_policy(env,spec)
        summary,steps,_=episode(env,policy,market_seed,c['low_gold_fraction'],trace=True)
        b=next(x for x in batch if x['suite']==choice['suite'] and x['target']==target and
               x['scenario']==choice['scenario'] and x['method']==spec['method'])
        controllers.append(dict(name=label,summary=summary,steps=steps,
            parameters=spec.get('parameters'),batch={k:float(b[k]) for k in
            ('score_per_step','mean_gold_error','max_gold_error','volume_per_step','mean_low_gold_fraction')}))
    return dict(scenario=scenario,label=choice['label'],market_seed=market_seed,model_seed=model_seed,
        steps=market.max_steps,products=[p.name for p in market.products],threshold=c['low_gold_fraction'],
        controllers=controllers,record=RECORD,model_sha256=r['sha256'],training_steps=c['training_steps'],
        protocol_sha256=sha(root/'protocol.json'),behavior=behavior_for(c,target).to_dict(),
        batch_markets=len(c['evaluation_seeds']),batch_models=len(c['training_seeds']))
