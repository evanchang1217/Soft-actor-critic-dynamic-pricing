"""Check the moved runtime against outputs frozen before the refactor."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.market.economy import TaskConfig
from experiments.market.continuing import Regime, StationaryMarket
from experiments.market.controllers import JointController, AnchoredPI

FIXTURE = Path(__file__).parent / 'fixtures/market-reference.npz'
REFERENCE = json.loads(FIXTURE.with_suffix('.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: c['key'])
def test_market_encoding_accounting_and_controllers_match_frozen_runtime(case):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == REFERENCE['fixture_sha256']
    env = StationaryMarket(TaskConfig(max_steps=32), Regime(**case['regime']), case['representation'])
    observation, _ = env.reset(seed=case['seed'])
    controls = [JointController(env, m, gain=3., stock_gain=0., integral_gain=.06, lookahead=2)
                for m in ('p', 'pi', 'joint', 'predictive')]
    controls.append(AnchoredPI(env, 8.5, 10.))
    with np.load(FIXTURE, allow_pickle=False) as data:
        prefix = case['key'] + '_'
        np.testing.assert_array_equal(observation, data[prefix+'observations'][0])
        np.testing.assert_array_equal(env.encode(observation), data[prefix+'encodings'][0])
        for t, action in enumerate(data[prefix+'actions']):
            np.testing.assert_array_equal([c.act(observation) for c in controls], data[prefix+'proposals'][t])
            observation, reward, _, truncated, info = env.step(action)
            np.testing.assert_array_equal(observation, data[prefix+'observations'][t+1])
            np.testing.assert_array_equal(env.encode(observation), data[prefix+'encodings'][t+1])
            np.testing.assert_array_equal([info[k] for k in case['fields']], data[prefix+'info'][t])
            assert reward == data[prefix+'rewards'][t]
        assert truncated
