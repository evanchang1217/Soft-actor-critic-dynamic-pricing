"""Independent fixed-model prediction audit on new exploratory market states.

True parameters appear only in this diagnostic reference, never in a fitted
controller. Conditional-median discrepancies are not long-run value errors.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np

from .model_error_study import collect, NOMINAL
from .common import read, sha, csv_write
from rl.artifacts import write_json


def predict(rows, parameters):
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    bids = np.stack([a['price0'], a['price1']], -1)
    asks = np.stack([a['price2'], a['price3']], -1)
    stock = np.stack([a['stock0'], a['stock1']], -1)
    house = np.stack([a['house0'], a['house1']], -1)
    recipe = np.stack([a['recipe'], 3-a['recipe']], -1)
    sigmoid = lambda z: 1/(1+np.exp(np.clip(-z, -50, 50)))
    z = bids/1.5; allocation = np.exp(z-z.max(-1, keepdims=True)); allocation /= allocation.sum(-1, keepdims=True)
    supply = 36*sigmoid(bids.mean(-1)-parameters['reservation'])[:, None]*allocation
    supply = np.minimum(supply, np.maximum(0, 72-stock))
    cost = (supply*bids).sum(-1)
    supply *= np.minimum(1, a['treasury']/np.maximum(cost, 1e-12))[:, None]
    eligible = np.maximum(0, a['tasks'][:, None]*recipe-house)
    sales = np.minimum(eligible*sigmoid(np.log(3)-parameters['base_slope']*np.log(asks/10)), stock+supply)
    cost = (sales*asks).sum(-1); money = a['money']+(supply*bids).sum(-1)
    sales *= np.minimum(1, money/np.maximum(cost, 1e-12))[:, None]
    return supply, sales


def audit(root):
    root = Path(root); p = read(root/'protocol.json'); fits = read(root/'fits.json')
    destination = root/'prediction-audit'; destination.mkdir(exist_ok=False)
    # Fixed new exploratory replicates, never used to fit, tune or train.
    plan = dict(replicates=[10, 11], transitions_per_replicate=384,
                scope='Exploratory public states, not policy-visited distributions. Structural median reference, not actual long-run counterfactual policy values.',
                source_sha256=sha(Path(__file__)), protocol_sha256=sha(root/'protocol.json'), fits_sha256=sha(root/'fits.json'))
    write_json(destination/'plan.json', plan); output = []
    for case, actual in p['cases'].items():
        for replicate in plan['replicates']:
            rows = collect(replicate, plan['transitions_per_replicate'], actual)
            csv_write(destination/f'{case}-{replicate}.csv', rows)
            reference = predict(rows, actual)
            realized = [np.array([[r[name+str(i)] for i in range(2)] for r in rows]) for name in ('supply', 'sales')]
            for fitted in fits:
                estimated = predict(rows, fitted['parameters'])
                record = dict(case=case, audit_replicate=replicate, size=fitted['size'], calibration_replicate=fitted['replicate'],
                              switch_probability_absolute_error=abs(fitted['parameters']['switch_probability']-actual['switch_probability']))
                for i, name in enumerate(('supply', 'sales')):
                    record[name+'_structural_rmse'] = float(np.sqrt(np.mean((estimated[i]-reference[i])**2)))
                    record[name+'_realized_rmse'] = float(np.sqrt(np.mean((estimated[i]-realized[i])**2)))
                    record[name+'_correct_model_realized_rmse'] = float(np.sqrt(np.mean((reference[i]-realized[i])**2)))
                output.append(record)
    csv_write(destination/'prediction.csv', output); shutil.copyfile(__file__, destination/'source.py')
    write_json(destination/'hashes.json', {str(f.relative_to(destination)):sha(f) for f in destination.iterdir() if f.is_file()})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--record', required=True, type=Path)
    audit(parser.parse_args().record)
