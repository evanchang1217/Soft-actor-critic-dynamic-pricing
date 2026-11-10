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


def assert_frozen_numeric(actual, expected):
    """Allow rounding differences across NumPy CPU kernels, not model drift."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    if not np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_array_equal(actual, expected)
        return
    # float32 encodings can differ by a few rounding units in SIMD functions.
    # float64 accounting also has cancellation near zero; its absolute bound
    # scales with the recorded ledger magnitude, rather than a loose constant.
    precision = np.finfo(expected.dtype)
    scale = max(1., float(np.max(np.abs(expected))))
    rtol = (4 if expected.dtype == np.float32 else 32) * precision.eps
    atol = 32 * np.finfo(np.float64).eps * scale
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_reference_tolerance_accepts_rounding_but_rejects_changed_values(dtype):
    expected = np.array([0., .5, 1000.], dtype=dtype)
    assert_frozen_numeric(np.nextafter(expected, dtype(np.inf)), expected)
    changed = expected.copy()
    changed[1] += .001
    with pytest.raises(AssertionError):
        assert_frozen_numeric(changed, expected)


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
        assert_frozen_numeric(observation, data[prefix+'observations'][0])
        assert_frozen_numeric(env.encode(observation), data[prefix+'encodings'][0])
        for t, action in enumerate(data[prefix+'actions']):
            assert_frozen_numeric([c.act(observation) for c in controls], data[prefix+'proposals'][t])
            observation, reward, _, truncated, info = env.step(action)
            assert_frozen_numeric(observation, data[prefix+'observations'][t+1])
            assert_frozen_numeric(env.encode(observation), data[prefix+'encodings'][t+1])
            assert_frozen_numeric([info[k] for k in case['fields']], data[prefix+'info'][t])
            assert_frozen_numeric(reward, data[prefix+'rewards'][t])
            # Exogenous Poisson task counts must remain exactly the same.
            assert info['arrivals'] == data[prefix+'info'][t, case['fields'].index('arrivals')]
        assert truncated
