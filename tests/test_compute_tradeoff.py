from pathlib import Path
import json

import numpy as np
import pytest

from experiments.compute_tradeoff import resolve, common_states, prepared
from experiments.common import read


def test_budget_protocol_requires_stochastic_forecasts_and_due_nondue_audits():
    p = read('experiments/configs/compute-tradeoff.json'); resolve(p)
    p['planners'][0]['particles'] = 1
    with pytest.raises(AssertionError): resolve(p)
    p = read('experiments/configs/compute-tradeoff.json'); p['audit_steps'] = [48,144]
    with pytest.raises(AssertionError): resolve(p)


def test_complete_tiny_quality_cost_training_replay_and_archive(tmp_path):
    from experiments.model_error_study import initialize as parent_initialize, calibrate, train, NOMINAL
    from experiments.compute_tradeoff import initialize, run, benchmark, training_cost, analyze, verify, tables
    parent_plan = read('experiments/configs/model-error-study.json')
    parent_plan.update(calibration_replicates=[0], calibration_sizes=[4,8], training_seeds=[50],
        training_episodes=4,training_steps_per_episode=4,evaluation_steps=8,early_window=4,
        evaluation_seeds=[224000010],cases=dict(nominal=NOMINAL),bootstrap_resamples=20)
    parent_plan['sac'].update(hidden=12,batch_size=8,replay_size=128,warmup_steps=8)
    path = tmp_path/'parent.json'; path.write_text(json.dumps(parent_plan))
    parent = parent_initialize(tmp_path/'parent',path); calibrate(parent); train(parent)
    p = read('experiments/configs/compute-tradeoff.json')
    p.update(calibration_size=8,calibration_replicates=[0],training_seeds=[50],evaluation_seeds=[228000010],
        evaluation_steps=8,early_window=4,minimum_observations=2,update_interval=2,rolling_window=4,
        minimum_identifying_points=1,audit_markets=[229000010],audit_steps=[4,5],timing_repetitions=1,
        training_replays=[dict(replicate=0,seed=50)],bootstrap_resamples=20,
        planners=[dict(method='h1_f2',horizon=1,particles=2),dict(method='h2_f2',horizon=2,particles=2)])
    path = tmp_path/'plan.json'; path.write_text(json.dumps(p))
    root = initialize(tmp_path/'record',path,parent)
    run(root); benchmark(root); training_cost(root); analyze(root); result = verify(root,2)
    assert result['evaluation_episodes'] == 8 and result['window_summaries'] == 40
    assert result['common_states'] == 4 and result['timing_samples'] == 16
    assert result['timed_replay_interactions'] == 16 and not result['performance_policies_changed']
    assert result['refits'] == 16
    costs,contrasts,frontier,crossovers = tables(root)
    assert len(costs) == 4 and len(contrasts) == 72 and len(frontier) == 8
    assert all(row['mean_total_cpu_ms'] > 0 for row in costs)
    assert all(r['cpu_ms'] > 0 for r in frontier)
    states = common_states(root); assert states == read(root/'common-states.json')
    assert all(len(s['history']) <= p['rolling_window'] for s in states)
    obj = prepared(root,'h1_f2',states[0],0)
    assert obj.steps == states[0]['step']-1 and obj.planner.integral == 0
    assert obj.parameters == states[0]['parameters']['0']
    # Future observations are present for measuring observe(), never for act().
    from copy import deepcopy
    changed = deepcopy(states[0]); changed['next_observation'] = [999.]*len(changed['next_observation'])
    obs = np.asarray(states[0]['observation'],np.float32)
    np.testing.assert_array_equal(obj.act(obs),prepared(root,'h1_f2',changed,0).act(obs))
    from scripts.reproduce import reproduce_compute as reproduce
    assert reproduce(root,workers=2)['result'] == result
