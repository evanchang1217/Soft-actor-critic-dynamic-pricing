"""Failure recovery and validation behavior shared by the research runners."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module_name,entrypoint,check_name,inputs', [
    ('model_error_study', 'evaluate', 'check_sources',
     ('protocol.json', 'fits.json', 'models.json', 'source-hashes.json')),
    ('adaptive_planning', 'run', 'check', ('protocol.json', 'source-hashes.json', 'parent-hashes.json')),
    ('compute_tradeoff', 'run', 'check', ('protocol.json', 'source-hashes.json', 'parent-hashes.json')),
])
def test_failed_evaluation_requires_explicit_retry_and_preserves_outputs(
        tmp_path, monkeypatch, module_name, entrypoint, check_name, inputs):
    module = importlib.import_module('experiments.'+module_name)
    run = getattr(module, entrypoint)
    for name in inputs:
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr(module, check_name, lambda root: None)
    monkeypatch.setattr(module, 'jobs', lambda root: [{}])
    fail = True

    class Future:
        def result(self):
            if fail:
                raise RuntimeError('Interrupted worker')
            records = [dict(case='nominal', method='sac', size=8, replicate=0,
                            model_seed=0, market_seed=1, window='full', mean_reward=1.)]
            result = (records, [dict(step=1, reward=1.)])
            return result if entrypoint == 'evaluate' else (*result, [dict(after_step=1)], dict(decisions=1))

    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, *args): return Future()

    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    with pytest.raises(RuntimeError, match='Interrupted worker'):
        run(tmp_path)
    assert json.loads((tmp_path/'evaluation-failed.json').read_text())['error'] == 'RuntimeError'
    partial_trace = (tmp_path/'steps.csv.gz').read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path)
    failure = (tmp_path/'evaluation-failed.json').read_bytes()
    (tmp_path/'evaluation-failed.json').unlink()
    with pytest.raises(FileExistsError):
        run(tmp_path, retry=True)
    (tmp_path/'evaluation-failed.json').write_bytes(failure)
    (tmp_path/'protocol.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='inputs changed'):
        run(tmp_path, retry=True)
    assert not (tmp_path/'failed-attempts').exists()
    (tmp_path/'protocol.json').write_text('{}')
    fail = False
    run(tmp_path, retry=True)
    assert (tmp_path/'data-hashes.json').exists()
    assert not (tmp_path/'evaluation-failed.json').exists()
    backup, = (tmp_path/'failed-attempts').iterdir()
    assert (backup/'steps.csv.gz').read_bytes() == partial_trace
    assert (backup/'evaluation-failed.json').is_file()
    for retry in (False, True):
        with pytest.raises(FileExistsError):
            run(tmp_path, retry=retry)


def test_validation_remains_active_with_optimized_python(tmp_path):
    code = '''
import json
from pathlib import Path
from experiments.common import compare_tables
from experiments.compute_tradeoff import resolve
from experiments.adaptive_planning import check
from scripts.reproduce import hydrate_trace

def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise RuntimeError('Validation was skipped')

p = json.loads(Path('experiments/configs/compute-tradeoff.json').read_text())
p['planners'][0]['particles'] = 1
rejected(lambda: resolve(p))
rejected(lambda: compare_tables([{'method':'wrong'}], [{'method':'expected'}]))
root = Path(__import__('sys').argv[1])
(root/'source/rl').mkdir(parents=True)
(root/'source/rl/config.py').write_text('changed')
(root/'source-hashes.json').write_text(json.dumps({'rl/config.py':'0'*64}))
rejected(lambda: check(root))
(root/'steps.csv.gz').write_bytes(b'changed')
(root/'trace-parts.json').write_text(json.dumps({
    'format':'lossless-compressed-byte-parts-v1', 'file':'steps.csv.gz', 'sha256':'0'*64}))
rejected(lambda: hydrate_trace(root))
'''
    result = subprocess.run([sys.executable, '-O', '-c', code, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1],
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
