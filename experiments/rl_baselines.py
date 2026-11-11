"""Optional SAC/TD3/PPO/DQN comparison matching the original 30k study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from rl.config import MarketConfig, SACConfig, integer
from rl.environment import MultiStoreEnv
from rl.sac import SACAgent
from rl.training import train
from .baselines import CurrencyFeedbackPolicy
from .behavior_market import IntegralFeedback
from .common import read, sha, snapshot_research
from .evaluate import control_rollout, control_summary, paired_interval, write_csv

METHODS = ('sac_custom', 'sac_sb3', 'td3', 'ppo', 'dqn')


class PricingEnv(gym.Wrapper):
    """Common semantic observations and normalized fixed-fee price actions."""
    def __init__(self, market, *, discrete=False, training_seed=None):
        super().__init__(MultiStoreEnv(market))
        self.training_seed, self.episodes = training_seed, 0
        dimensions = 2*len(market.products)
        self.discrete = discrete
        self.grid = None
        if discrete:
            if 3**dimensions > 10000:
                raise ValueError('DQN joint grid is too large; use a separate action-space study')
            self.grid = np.array(list(product((-1., 0., 1.), repeat=dimensions)), dtype=np.float32)
            self.action_space = Discrete(len(self.grid))
        else:
            self.action_space = Box(-1., 1., (dimensions,), dtype=np.float32)
        self.observation_space = Box(-1., 1., self.env.observation_space.shape, dtype=np.float32)

    def physical_action(self, action):
        if self.discrete:
            value = np.asarray(action)
            if value.shape != () or not np.issubdtype(value.dtype, np.integer):
                raise ValueError('DQN action must be a scalar integer')
            index = int(value.item())
            if not self.action_space.contains(index):
                raise ValueError('DQN action is outside the joint grid')
            normalized = self.grid[index]
        else:
            normalized = np.asarray(action, dtype=np.float32)
            if not self.action_space.contains(normalized):
                raise ValueError('Continuous action must be inside [-1, 1]')
        return np.r_[normalized*self.env.config.price_offset, 0.].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if self.training_seed is not None:
            self.episodes += 1
            seed = self.training_seed+self.episodes*10000
        observation, info = super().reset(seed=seed, options=options)
        return self.env.encode(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(self.physical_action(action))
        return self.env.encode(observation), reward, terminated, truncated, info


def load_config(path):
    p = read(path)
    market = MarketConfig.from_dict(p['market'])
    if market.reward_mode != 'balanced_v1':
        raise ValueError('Pilot requires the balanced common reward')
    if not p['methods'] or len(set(p['methods'])) != len(p['methods']) or set(p['methods'])-set(METHODS):
        raise ValueError('Choose distinct supported methods')
    sac = SACConfig(**p['sac'])
    for name in ('training_steps', 'ppo_rollout_steps', 'ppo_batch_size', 'bootstrap_resamples'):
        integer(p[name], name)
    if (p['training_steps'] % market.max_steps or p['training_steps'] % p['ppo_rollout_steps']
            or p['ppo_rollout_steps'] % p['ppo_batch_size'] or sac.warmup_steps >= p['training_steps']
            or sac.episodes*market.max_steps != p['training_steps'] or sac.control_fee):
        raise ValueError('Budgets must contain complete market episodes and PPO rollouts/minibatches')
    if 'dqn' in p['methods'] and 3**(2*len(market.products)) > 10000:
        raise ValueError('DQN joint grid is too large; use a separate action-space study')
    for key in ('training_seeds', 'evaluation_seeds'):
        values = p[key]
        if len(values) < 2 or len(set(values)) != len(values):
            raise ValueError('At least two distinct seeds per independent group are required')
        for seed in values:
            integer(seed, key, 0)
    used = {s+ep*10000 for s in p['training_seeds']
            for ep in range(p['training_steps']//market.max_steps+2)}
    if used & set(p['evaluation_seeds']):
        raise ValueError('Training and evaluation seeds overlap')
    names = set()
    for scenario in p['scenarios']:
        if scenario['name'] in names or set(scenario['changes'])-{
                'price_sensitivity_low', 'price_sensitivity_high', 'shock_multiplier'}:
            raise ValueError('Invalid scenario or changed observation/action semantics')
        replace(market, **scenario['changes'])
        names.add(scenario['name'])
    if not p['scenarios'] or p['scenarios'][0] != dict(name='nominal', changes={}):
        raise ValueError('First scenario must be nominal')
    integer(p['bootstrap_seed'], 'bootstrap_seed', 0)
    return market, p


def build_model(method, env, p, seed):
    try:
        from stable_baselines3 import SAC, TD3, PPO, DQN
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as error:
        raise RuntimeError('Install optional requirements-baselines.txt in a separate environment') from error
    sac = SACConfig(**p['sac'])
    shared = dict(policy='MlpPolicy', env=env, seed=seed, device='cpu', verbose=0,
                  gamma=sac.gamma, learning_rate=sac.learning_rate,
                  policy_kwargs=dict(net_arch=[sac.hidden]*2, activation_fn=torch.nn.ReLU))
    if method == 'ppo':
        return PPO(**shared, n_steps=p['ppo_rollout_steps'], batch_size=p['ppo_batch_size'], n_epochs=10)
    shared.update(buffer_size=sac.replay_size, learning_starts=sac.warmup_steps,
                  batch_size=sac.batch_size, tau=sac.tau)
    if method == 'sac_sb3':
        return SAC(**shared, ent_coef=f'auto_{sac.initial_alpha}', train_freq=1, gradient_steps=1)
    if method == 'td3':
        noise = NormalActionNoise(np.zeros(env.action_space.shape), .1*np.ones(env.action_space.shape))
        return TD3(**shared, action_noise=noise, train_freq=1, gradient_steps=1)
    if method == 'dqn':
        shared['learning_rate'] = 1e-4
        shared['tau'] = 1.0
        return DQN(**shared, exploration_fraction=.5, train_freq=4, gradient_steps=1,
                   target_update_interval=500)
    raise ValueError('Unknown library method')


def train_one(root, p, method, seed):
    root = Path(root)
    torch.set_num_threads(1)
    market = MarketConfig.from_dict(p['market'])
    output = root/'models'/method/f'seed-{seed}'
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    if method == 'sac_custom':
        config = SACConfig(**p['sac'])
        def progress(row, info):
            if row['episode'] % 16 == 0:
                print(f'{method} seed {seed}: {row["steps"]} steps', flush=True)
        result = train(market, config, seed, output, callback=progress)
        path, steps, updates = output/'policy.pt', result['steps'], result['updates']
    else:
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        class Progress(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % 1600 == 0:
                    print(f'{method} seed {seed}: {self.num_timesteps} steps', flush=True)
                return True
        output.mkdir(parents=True, exist_ok=False)
        env = PricingEnv(market, discrete=method == 'dqn', training_seed=seed)
        model = build_model(method, Monitor(env, str(output/'monitor.csv')), p, seed)
        model.learn(total_timesteps=p['training_steps'], callback=Progress())
        path, steps, updates = output/'policy.zip', model.num_timesteps, model._n_updates
        model.save(path)
        env.close()
    if steps != p['training_steps'] or updates <= 0:
        raise ValueError('Training did not meet the declared interaction budget or perform updates')
    metadata = dict(method=method, training_seed=seed, steps=steps, update_counter=updates,
                    cpu_seconds=time.process_time()-start_cpu,
                    wall_seconds=time.perf_counter()-start_wall,
                    path=path.relative_to(root).as_posix(), sha256=sha(path))
    write_json(output/'comparison-metadata.json', metadata)
    return metadata


class SavedPolicy:
    def __init__(self, root, metadata, env):
        path = Path(root)/metadata['path']
        if sha(path) != metadata['sha256']:
            raise ValueError('Comparison model hash mismatch')
        self.method = metadata['method']
        if self.method == 'sac_custom':
            self.model = SACAgent.load(path)[0]
            self.model.env = env
        else:
            from stable_baselines3 import SAC, TD3, PPO, DQN
            self.model = dict(sac_sb3=SAC, td3=TD3, ppo=PPO, dqn=DQN)[self.method].load(path, device='cpu')
            self.adapter = PricingEnv(env.config, discrete=self.method == 'dqn')

    def act(self, observation):
        if self.method == 'sac_custom':
            return self.model.act(observation, deterministic=True)
        action, _ = self.model.predict(self.adapter.env.encode(observation), deterministic=True)
        return self.adapter.physical_action(action)


def evaluate_one(root, p, metadata):
    torch.set_num_threads(1)
    rows = []
    market = MarketConfig.from_dict(p['market'])
    for scenario in p['scenarios']:
        env = MultiStoreEnv(replace(market, **scenario['changes']))
        policy = SavedPolicy(root, metadata, env)
        for seed in p['evaluation_seeds']:
            rows.append(dict(scenario=scenario['name'], method=metadata['method'],
                             training_seed=metadata['training_seed'], evaluation_seed=seed,
                             **control_rollout(env, policy, seed)))
        env.close()
    return rows


def feedback_results(market, p):
    record = Path(p['feedback_record'])
    protocol, frozen = read(record/'protocol.json'), read(record/'frozen-selection.json')
    if (market.to_dict() != protocol['market'] or sha(record/'protocol.json') != frozen['protocol_sha256']
            or sha(record/'validation.csv') != frozen['validation_sha256']):
        raise ValueError('Frozen feedback parameters require their original market and verified protocol')
    choices, rows = frozen['choices']['original'], []
    for scenario in p['scenarios']:
        env = MultiStoreEnv(replace(market, **scenario['changes']))
        for method, cls, key in [('p', CurrencyFeedbackPolicy, 'feedback'), ('pi', IntegralFeedback, 'pi')]:
            for seed in p['evaluation_seeds']:
                policy = cls(env, **choices[key]['parameters'])
                rows.append(dict(scenario=scenario['name'], method=method, training_seed='',
                                 evaluation_seed=seed, **control_rollout(env, policy, seed)))
        env.close()
    return rows, dict(source=p['feedback_record'], sha256=sha(record/'frozen-selection.json'), choices=choices)


def check_historical_settings(market, p):
    record = Path(p['feedback_record'])
    original = read(record/'protocol.json')
    expected = dict(market=market.to_dict(), sac=p['sac'], training_steps=p['training_steps'],
                    training_seeds=p['training_seeds'], evaluation_seeds=p['evaluation_seeds'])
    for key, value in expected.items():
        if original[key] != value:
            raise ValueError(f'Comparison does not match historical {key}')
    return dict(record=p['feedback_record'], protocol_sha256=sha(record/'protocol.json'),
                matched_fields=list(expected), test_markets_previously_used=True)


def report(root, p, rows, models):
    summary = control_summary(rows)
    write_csv(root/'summary.csv', summary)
    rng, contrasts = np.random.default_rng(p['bootstrap_seed']), []
    for scenario in p['scenarios']:
        selected = [row for row in rows if row['scenario'] == scenario['name']]
        for method in ('sac_custom', 'sac_sb3'):
            if method not in p['methods']:
                continue
            proposed = [row for row in selected if row['method'] == method]
            for baseline in [*p['methods'], 'p', 'pi']:
                if baseline == method:
                    continue
                reference = [row for row in selected if row['method'] == baseline]
                contrasts.append(dict(scenario=scenario['name'], method=method, baseline=baseline,
                    **paired_interval(proposed, reference, 'common_return', rng, p['bootstrap_resamples'])))
    if contrasts:
        write_csv(root/'contrasts.csv', contrasts)
    lines = ['# RL baseline comparison: historical 30k settings', '',
             f'{len(p["training_seeds"])} training seeds, {p["training_steps"]:,} interactions per policy; '
             f'{len(p["evaluation_seeds"])} held-out markets per scenario, {market_goods(p)} goods.', '',
             'This retrospective extension matches the original-market behavioral study: market, custom '
             'SAC configuration, training seeds and evaluation markets. These test markets have been used '
             'before; this is not a fresh confirmatory study. P/PI parameters remain frozen from validation. '
             'RL presets receive no hyperparameter search, and results do not establish convergence.', '',
             'All methods use the same semantic observations, reward, fixed fee, episode seeds and price bounds. '
             'PPO, TD3 and SAC use continuous six-dimensional actions in the three-good case. '
             'DQN uses the Cartesian {-1,0,+1} grid: 729 joint actions for three goods, a different action resolution.', '',
             'Custom SAC and SB3 SAC are separate implementations. The update counters have algorithm-specific '
             'meanings; environment interactions are matched, not gradient steps or CPU cost.', '',
             f'Custom SAC uses its full recorded configuration in protocol.json. PPO uses '
             f'{p["ppo_rollout_steps"]}-step rollouts and {p["ppo_batch_size"]}-sample minibatches so the '
             'interaction budget is exact; its other presets and the library presets are recorded in the '
             'source snapshot and saved models.', '',
             '| Scenario | Method | Common score ↑ | Currency error % ↓ | Units traded | Sell-price change % ↓ |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append(f'| {row["scenario"]} | {row["method"]} | {row["common_return_mean"]:.3f} | '
                     f'{100*row["mean_gold_deviation_mean"]:.3f} | {row["total_volume_mean"]:.0f} | '
                     f'{100*row["mean_item_price_change_mean"]:.3f} |')
    lines += ['', '## Training cost', '', '| Method | Mean measured CPU seconds |', '| --- | ---: |']
    for method in p['methods']:
        cost = np.mean([model['cpu_seconds'] for model in models if model['method'] == method])
        lines.append(f'| {method} | {cost:.2f} |')
    lines += ['', 'CPU includes per-job initialization, training and saving. See models.json for steps, '
              'algorithm-specific update counts and measured wall times. All inference is deterministic.', '',
              'contrasts.csv contains exploratory crossed model/market bootstrap intervals without multiplicity '
              'adjustment. A fixed budget and untuned presets do not establish a definitive algorithm ranking.', '']
    (root/'README.md').write_text('\n'.join(lines), encoding='utf-8')


def market_goods(p):
    return len(MarketConfig.from_dict(p['market']).products)


def run(config, output, workers=3):
    import stable_baselines3, pandas
    market, p = load_config(config)
    integer(workers, 'workers')
    reference = check_historical_settings(market, p)
    root = new_run_directory(output)
    write_json(root/'protocol.json', p)
    write_json(root/'historical-settings.json', reference)
    runtime = manifest('historical-30k-rl-baseline-comparison')
    runtime['optional_packages'] = dict(stable_baselines3=stable_baselines3.__version__,
                                        pandas=pandas.__version__, gymnasium=gym.__version__)
    write_json(root/'manifest.json', runtime)
    snapshot_research(root)
    for name in ('requirements.txt', 'requirements-baselines.txt'):
        shutil.copyfile(Path(__file__).resolve().parents[1]/name, root/name)
    rows, feedback = feedback_results(market, p)
    write_json(root/'feedback.json', feedback)
    models = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        work = [pool.submit(train_one, str(root), p, method, seed)
                for method in p['methods'] for seed in p['training_seeds']]
        for future in as_completed(work):
            metadata = future.result()
            models.append(metadata)
            write_json(root/'models.json', sorted(models, key=lambda m: (m['method'], m['training_seed'])))
            print(f'Trained {metadata["method"]} seed {metadata["training_seed"]}: '
                  f'{metadata["steps"]} steps, {metadata["cpu_seconds"]:.1f} CPU seconds', flush=True)
        evaluation = [pool.submit(evaluate_one, str(root), p, model) for model in models]
        for future in as_completed(evaluation):
            rows.extend(future.result())
    order = {s['name']: i for i, s in enumerate(p['scenarios'])}
    rows.sort(key=lambda r: (order[r['scenario']], r['method'], str(r['training_seed']), r['evaluation_seed']))
    write_csv(root/'episodes.csv', rows)
    report(root, p, rows, models)
    write_json(root/'output-hashes.json', {path.relative_to(root).as_posix(): sha(path)
               for path in sorted(root.rglob('*')) if path.is_file()})
    print(f'Comparison complete: {root}/README.md', flush=True)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/configs/rl-baselines.json')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    run(args.config, args.out, args.workers)
