from copy import deepcopy
import json

import numpy as np

from experiments.model_error_study import NOMINAL, market, read
from experiments.market_planner import MarketPlanner
from experiments.adaptive_planning import OnlinePlanner, public_transition, identifying_counts


def plan(): return read('experiments/configs/adaptive-planning.json')


def test_actions_match_frozen_planning_until_first_observed_update():
    p = plan(); actual = market(p['cases']['combined'], 48); obs, _ = actual.reset(seed=91)
    spec = dict(horizon=2, particles=2)
    online = OnlinePlanner(NOMINAL, p, spec); frozen = MarketPlanner(market(NOMINAL, 384), **spec)
    for _ in range(32):
        a = online.act(obs); b = frozen.act(obs); np.testing.assert_array_equal(a, b)
        nxt, *_ = actual.step(a); online.observe(obs, nxt); obs = nxt
    assert len(online.updates) == 1 and online.updates[0]['after_step'] == 32
    assert online.updates[0]['last_used_step'] == 32
    assert online.planner.integral == frozen.integral
    assert online.planner.rng.bit_generator.state == frozen.rng.bit_generator.state
    assert abs(online.parameters['reservation']-8.2) < .1


def test_unidentifiable_public_data_retain_behavioral_prior():
    p = plan(); online = OnlinePlanner(NOMINAL, p, dict(horizon=2, particles=2))
    env = market(NOMINAL); obs, _ = env.reset(seed=1)
    for _ in range(96): online.observe(obs, obs)
    assert online.parameters['reservation'] == NOMINAL['reservation']
    assert online.parameters['base_slope'] == NOMINAL['base_slope']
    assert online.updates[-1]['n'] == 64 and online.updates[-1]['first_used_step'] == 33
    assert not online.updates[-1]['reservation_accepted'] and not online.updates[-1]['slope_accepted']


def test_transition_extraction_reads_only_public_frames_and_latest_flows():
    env = market(NOMINAL); before, _ = env.reset(seed=1)
    after, *_ = env.step(np.zeros(4, np.float32))
    row = public_transition(market(NOMINAL), before, after)
    state = env.unpack(after[-env.frame_size:])
    assert row['supply0'] == state['last_supply'][0] and row['sales1'] == state['last_sales'][1]
    assert not set(row) & {'hidden_slope', 'productivity', 'preference', 'true_parameters'}
    env.env.preference[:] = 99.; env.env.productivity[:] = .001
    assert row == public_transition(market(NOMINAL), before, after)


def test_online_updates_and_forecasts_are_independent_of_actual_future():
    p = plan(); actual = market(p['cases']['combined'], 48); obs, _ = actual.reset(seed=91)
    a = OnlinePlanner(NOMINAL, p, dict(horizon=2, particles=2)); b = deepcopy(a)
    for _ in range(32):
        nxt, *_ = actual.step(np.zeros(4, np.float32)); a.observe(obs, nxt); b.observe(obs, nxt); obs = nxt
    actual.env.arrivals[:] = 999; actual.env.recipe_schedule[:] = 999
    actual.env.preference[:] = 999; actual.env.productivity[:] = .001
    assert a.parameters == b.parameters and a.updates == b.updates
    np.testing.assert_array_equal(a.act(obs), b.act(obs))


def test_complete_tiny_adaptive_study_and_archive(tmp_path):
    from experiments.model_error_study import initialize as initial_parent, calibrate, train
    from experiments.adaptive_planning import initialize, run, report, verify
    parent_plan = read('experiments/configs/model-error-study.json')
    parent_plan.update(calibration_replicates=[0], calibration_sizes=[4, 8], training_seeds=[50], training_episodes=4,
        training_steps_per_episode=4, evaluation_steps=8, early_window=4, evaluation_seeds=[224000010], cases=dict(nominal=NOMINAL), bootstrap_resamples=20)
    parent_plan['sac'].update(hidden=12, batch_size=8, replay_size=128, warmup_steps=8)
    config = tmp_path/'parent.json'; config.write_text(json.dumps(parent_plan))
    parent = initial_parent(tmp_path/'parent', config); calibrate(parent); train(parent)
    p = plan(); p.update(calibration_replicates=[0], training_seeds=[50], calibration_size=8,
        evaluation_seeds=[226000010], evaluation_steps=8, early_window=4, minimum_observations=2,
        update_interval=2, rolling_window=4, minimum_identifying_points=1,
        cases={c:p['cases'][c] for c in ('nominal', 'combined')},
        planner=dict(horizon=2, particles=2), planner_supplement=dict(horizon=3, particles=4), bootstrap_resamples=20)
    path = tmp_path/'plan.json'; path.write_text(json.dumps(p))
    root = initialize(tmp_path/'record', path, parent); run(root); report(root); result = verify(root, 2)
    assert result['evaluation_episodes'] == 16 and result['window_summaries'] == 80
    assert result['online_refits'] == 16 and result['new_training_steps'] == 0
    from scripts.report import render_adaptive as render
    render(root)
    from scripts.reproduce import reproduce_adaptive as reproduce
    assert reproduce(root, workers=2)['result'] == result
