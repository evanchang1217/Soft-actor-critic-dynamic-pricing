from copy import deepcopy
import json
import pickle

import numpy as np

from experiments.model_error_study import (
    NOMINAL, collect, fit, market, read, resolve, initialize, calibrate, train,
    evaluate, report, verify, csv_read, load_agent, jobs, bounded_minimum,
)
from experiments.market_planner import MarketPlanner


def test_logs_are_nested_and_only_contain_observable_quantities():
    small = collect(1, 48); large = collect(1, 384)
    assert small == large[:48]
    assert not {'hidden_slope', 'preference', 'productivity', 'accepted', 'parameters'} & set(small[0])
    assert set(small[0]) == {'index', 'episode', 'step', 'money', 'treasury', 'tasks', 'recipe', 'next_recipe',
        'price0', 'price1', 'price2', 'price3', 'stock0', 'stock1', 'house0', 'house1',
        'supply0', 'supply1', 'sales0', 'sales1'}


def test_forward_estimator_recovers_unseen_behavior_parameters_from_logs():
    actual = dict(base_slope=8., reservation=8.2, switch_probability=.15)
    estimate = fit(collect(2, 768, actual))['parameters']
    assert abs(estimate['base_slope']-actual['base_slope']) < .35
    assert abs(estimate['reservation']-actual['reservation']) < .06
    assert abs(estimate['switch_probability']-actual['switch_probability']) < .05


def test_estimate_has_no_external_truth_argument_and_uses_observed_next_recipe():
    rows = collect(0, 48); changed = deepcopy(rows)
    for r in changed: r['next_recipe'] = r['recipe']
    a, b = fit(rows), fit(changed)
    assert b['parameters']['switch_probability'] == 0
    assert a['parameters']['base_slope'] == b['parameters']['base_slope']
    assert a['parameters']['reservation'] == b['parameters']['reservation']


def test_fitted_planning_cannot_read_actual_parameters_or_future():
    actual = market(dict(base_slope=9., reservation=8.4, switch_probability=.2), 16)
    obs, _ = actual.reset(seed=3)
    known = market(NOMINAL, 16)
    original = pickle.dumps(actual)
    first = MarketPlanner(known, horizon=4, particles=4).act(obs)
    actual.env.productivity[:] = .001; actual.env.preference[:] = 99
    actual.env.arrivals[1:] = 99; actual.env.recipe_schedule[1:] = 99
    second = MarketPlanner(market(NOMINAL, 16), horizon=4, particles=4).act(obs)
    np.testing.assert_array_equal(first, second)
    assert known.config.base_slope == 5 and actual.config.base_slope == 9
    assert actual.action_space.contains(first)


def test_fitting_censoring_and_optimizer_remain_finite_at_edge_states():
    rows = collect(0, 8)
    for row in rows:
        row.update(treasury=0., money=0., tasks=0., stock0=72., stock1=72., supply0=0., supply1=0., sales0=0., sales1=0.)
    result = fit(rows)
    assert all(np.isfinite(x) for x in result['parameters'].values())
    assert abs(bounded_minimum(lambda x: (x-3.125)**2, 1, 8)-3.125) < 1e-6


def test_prediction_diagnostic_matches_zero_noise_physical_transitions():
    from experiments.model_error_prediction import predict
    parameters = dict(base_slope=8., reservation=8.2, switch_probability=.15)
    env = market(parameters); obs, _ = env.reset(seed=19)
    env.env.productivity[:] = 1.; env.env.preference[:] = 0.
    for _ in range(24):
        state = env.unpack(env.frame())
        obs, *_ = env.step(np.array([-.01, .02, -.01, .03], np.float32))
        row = dict(money=state['money'], treasury=state['treasury'], tasks=state['tasks'][0], recipe=state['recipes'][0, 0])
        for name, values in [('price', env.env.prices), ('stock', state['stock']), ('house', state['household'])]:
            row.update({name+str(i): float(x) for i, x in enumerate(values)})
        supply, sales = predict([row], parameters)
        np.testing.assert_allclose(supply[0], env.env.last_supply, atol=2e-5)
        np.testing.assert_allclose(sales[0], env.env.last_sales, atol=2e-5)


def test_complete_tiny_shared_data_study_and_replay(tmp_path):
    p = read('experiments/configs/model-error-study.json')
    p.update(calibration_replicates=[0], calibration_sizes=[4, 8], training_seeds=[50], training_episodes=4,
             training_steps_per_episode=4, evaluation_steps=8, early_window=4,
             evaluation_seeds=[224000010], cases=dict(nominal=NOMINAL), bootstrap_resamples=20,
             planner=dict(horizon=2, particles=2), planner_supplement=dict(horizon=3, particles=4))
    p['sac'].update(hidden=12, batch_size=8, replay_size=128, warmup_steps=8)
    config = tmp_path/'plan.json'; config.write_text(json.dumps(p))
    root = initialize(tmp_path/'record', config); calibrate(root); train(root); evaluate(root); report(root)
    result = verify(root)
    assert result['checkpoints'] == 2 and result['evaluation_episodes'] == 8
    assert result['window_summaries'] == 24
    assert len(jobs(root)) == 8
    for model in read(root/'models.json'):
        assert model['updates'] == 9
        agent = load_agent(root, model)
        assert agent.env.representation == 'stationary'
    from scripts.reproduce import reproduce_model_error as reproduce
    reproduce(root, save_verification=True)
    assert read(root/'archive-verification.json')['result'] == result
    from scripts.report import planner_gaps
    gaps = planner_gaps(root)
    assert len(gaps) == 9 and all(np.isfinite(r['difference']) for r in gaps)
    from scripts.report import render_model_error
    render_model_error(root)
    text = (root/'README.md').read_text()
    assert 'nested 4/8 public transitions' in text
    assert '16 synthetic interactions each' in text
    assert '1 cases, 1 fresh paired evaluation markets, 8 periods each' in text
    assert 'periods 5–8' in text and '2 final checkpoints' in text
    assert 'at 8 calibration transitions' in text
    assert 'missing `/usr/bin/python`' not in text
