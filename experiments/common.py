"""Shared record I/O and aggregate-market statistics; no experiment orchestration."""
import csv
import hashlib
import json
import zipfile
from rl.artifacts import write_json
from pathlib import Path
import numpy as np

MARKET_SOURCES = ('experiments/common.py', 'experiments/market/__init__.py',
    'experiments/market/economy.py', 'experiments/market/inventory.py',
    'experiments/market/controllers.py', 'experiments/market/continuing.py',
    'rl/sac.py', 'rl/config.py', 'rl/environment.py', 'rl/rewards.py', 'rl/artifacts.py')

def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def csv_read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def aggregate(rows,keys):
    identities=list(dict.fromkeys(tuple(r[k] for k in keys) for r in rows))
    metrics=[k for k in rows[0] if k.startswith(('mean_','late_','final_'))]
    return [dict(**dict(zip(keys,identity)),**{k:float(np.mean([float(r[k]) for r in rows if tuple(r[f] for f in keys)==identity])) for k in metrics}) for identity in identities]


def compare_tables(saved,computed):
    assert len(saved)==len(computed)
    for a,b in zip(saved,computed):
        assert set(a)==set(b)
        for k,v in b.items():
            if isinstance(v,str):assert a[k]==v
            else:np.testing.assert_allclose(float(a[k]),v,rtol=1e-8,atol=1e-8)


def summarize(rows):
    output={}
    for prefix,subset in [('mean',rows),('late',rows[len(rows)//3:])]:
        for k in ('reward','money_error','money_cost','service_cost','inventory_cost','price_cost','price_change','waste','waste_cost'):
            output[f'{prefix}_{k}']=float(np.mean([r[k] for r in subset]))
        total=max(1e-12,sum(r['arrivals'] for r in subset));output[f'{prefix}_fulfillment']=sum(r['completed'] for r in subset)/total
        output[f'{prefix}_expired_fraction']=sum(r['expired'] for r in subset)/total
    for k in ('pending_tasks','pending_goods','pending_cash','household','stock'):
        output[f'final_{k}']=float(rows[-1][k])
    return output


# Recognized pre-refactor parent; physics equivalence is covered by the frozen reference.
LEGACY_PARENT_SOURCES = {'experiments/advantage_market.py': '1cc44b86f328e229c883622be27ae99b62341793369899520ebb0caa68038d3e', 'experiments/advantage_study.py': 'aac89d511286991bcddc878edb1543b643a28936a26b2359077981e74a5efce3', 'experiments/inventory_market.py': 'f3d6d0c51ccd42b571b2fe7c70d4da552e5392a2921bfa6d7f6f8ac407ed0db6', 'experiments/inventory_study.py': '7cbfc5430ee5435a189e270ebe0bea08969e25df3199efbd168d87389d2b52af', 'experiments/feature_market.py': 'e6d321a3a70b1b40ba1dcbb215ddf034176aafbf701ac941e9b908ddbd49de91', 'experiments/feature_screen.py': '2d907f2da76fbc91a2b03109cfa6ba68541682fa7590a89f12bd4468b738e18d', 'rl/sac.py': 'c2bc647770a72f4c69d11238ca26c96db79d6240f554784492a8f6bcbe0227c2', 'rl/config.py': 'a103466df2fe8f16ad0aa320f5127ccde503e5ddc4ee67817b8cbb288b777f01', 'rl/environment.py': 'e85214894121cbae07187a543a2a6a0d406b023684425f354d2e9d7dac265e2c', 'rl/rewards.py': '9dff87afbb1bf0bf83329c8e8cd4270c09bb954ebea0be65e9864c174dc6c30b', 'rl/artifacts.py': '1fc637b02809bffb972f1f25c27f8ea8d18d46ceb54f1575f6a5b8c750abaad9', 'experiments/advantage_anchor.py': 'a7f177dd0b47c55a8a10175d0a60df68ebd88ff4abb9a1490171729f01cfcc19', 'experiments/advantage_ledger_probe.py': '09ffb0fcb15f71104e327718c3c0461a641a8bcd7bafab4318dd0ecc2443b189', 'experiments/stationary_market.py': '234713240c3fa55a556c4abb40c89ae34344b3952c16d14976ca6e9f16bd6a2b', 'experiments/market_planner.py': '819806ed7e07c491ed3928beff6360f7164202e0b5e24ff1de8ba2e8363a8f11', 'experiments/stationary_study.py': 'a4500743cf771b2a7653acd49b3f20ec8cb4891572699b2d938bc3af48090e1b', 'experiments/model_error_study.py': 'e7e2369561f5e37829b01c0ea86a15987fbe171ec7775e4e793b2e4a3905c037'}

def check_parent_manifest(parent, current):
    if parent == LEGACY_PARENT_SOURCES:
        for name, digest in parent.items():
            if name.startswith('rl/') and current.get(name) != digest:
                raise ValueError('Legacy parent requires the unchanged learning core')
        return
    if any(current.get(name) != digest for name, digest in parent.items()):
        raise ValueError('Parent source version is incompatible; verify or regenerate the parent study')


def snapshot_research(output):
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("rl/*.py"), *root.glob("experiments/**/*.py"), *root.glob("experiments/configs/*.json")])
    paths.append(root / "requirements-lock.txt")
    digests = {}
    with zipfile.ZipFile(output / "research-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative, content = path.relative_to(root).as_posix(), path.read_bytes()
            archive.writestr(relative, content)
            digests[relative] = hashlib.sha256(content).hexdigest()
    write_json(output / "research-source-hashes.json", digests)
    return root, digests