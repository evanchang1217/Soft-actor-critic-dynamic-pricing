"""Generate the refactor reference using an explicitly supplied frozen source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def generate(source, output):
    sys.path.insert(0, str(Path(source).resolve()))
    import numpy as np
    from experiments.feature_market import FeatureConfig
    from experiments.stationary_market import StationaryMarket, Regime
    from experiments.advantage_market import AdvantageController
    from experiments.advantage_anchor import AnchoredPI
    regimes = [{}, {'switch_probability': 0.}, {'switch_probability': .15},
               {'buffer_periods': 3.}, {'capacity_ratio': 1.},
               {'switch_probability': .15, 'buffer_periods': 3., 'capacity_ratio': 1.}]
    arrays = {}; cases = []
    for regime in regimes:
        for representation in ('raw', 'stationary'):
            key = f'case{len(cases)}'; seed = 711
            env = StationaryMarket(FeatureConfig(max_steps=32), Regime(**regime), representation)
            observation, _ = env.reset(seed=seed)
            controls = [AdvantageController(env, m, gain=3., stock_gain=0., integral_gain=.06, lookahead=2)
                        for m in ('p', 'pi', 'joint', 'predictive')]
            controls.append(AnchoredPI(env, 8.5, 10.))
            rng = np.random.default_rng(713)
            observations = [observation]; encodings = [env.encode(observation)]; actions = []; rewards = []; proposals = []; infos = []
            for _ in range(32):
                proposals.append([c.act(observation) for c in controls])
                action = rng.uniform(-.05, .05, 4).astype(np.float32)
                observation, reward, _, _, info = env.step(action)
                observations.append(observation); encodings.append(env.encode(observation))
                actions.append(action); rewards.append(reward); infos.append(info)
            for name, values in [('observations', observations), ('encodings', encodings), ('actions', actions),
                                 ('rewards', rewards), ('proposals', proposals)]:
                arrays[f'{key}_{name}'] = np.asarray(values)
            fields = sorted(infos[0]); arrays[key+'_info'] = np.asarray([[row[k] for k in fields] for row in infos])
            cases.append(dict(key=key, regime=regime, representation=representation, seed=seed, fields=fields))
    output = Path(output)
    np.savez_compressed(output, **arrays)
    names = ['experiments/feature_market.py', 'experiments/inventory_market.py', 'experiments/feature_screen.py',
             'experiments/advantage_market.py', 'experiments/advantage_anchor.py', 'experiments/stationary_market.py']
    metadata = dict(description='Generated before refactoring from the retained stationary study source snapshot; no training or test selection.',
                    controls=['p', 'pi', 'joint', 'predictive', 'anchored_pi'], cases=cases,
                    source_hashes={name:hashlib.sha256((Path(source)/name).read_bytes()).hexdigest() for name in names},
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Frozen reference: {len(cases)} markets, 384 physical steps, 1,920 controller decisions')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); generate(a.source, a.out)
